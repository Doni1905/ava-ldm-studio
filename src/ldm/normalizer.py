"""
src/ldm/normalizer.py
======================
Linguistic Normalization Pipeline for AVA LDM Studio.

Purpose
-------
Convert informal/dialectal/code-mixed user language into normalized
semantic English WITHOUT inventing new meaning.

    Input:  "Dei nalaiku assignment submit panna remind pannu"
    Output: "Remind me to submit my assignment tomorrow."

The normalizer explicitly guarantees:
  - Intent is preserved (no new semantic content added)
  - Named entities, times, quantities, negation, places, apps are preserved
  - Discourse fillers are dropped
  - Tanglish phrases are glossed to English equivalents
  - Dialect-specific forms are adapted before general glossing

Architecture
------------

    ASR transcript
        │
        ▼
    [SemanticPreserver.extract]       ← lock down what must survive
        │
        ▼
    [DialectAdapter.adapt]            ← dialect-specific filler removal
        │
        ▼
    [SlangMapper.apply_phrases]       ← multi-word Tanglish → English
        │
        ▼
    [SlangMapper.apply_token_glosses] ← single-token Tanglish → English
        │
        ▼
    [_reconstruct]                    ← clean whitespace, capitalize, terminate
        │
        ▼
    [SemanticPreserver.verify]        ← confirm anchors survived
        │
        ▼
    NormalizationResult

Constraints
-----------
- This module is NOT a chatbot.
- It does NOT call any LLM or generative model.
- Unknown tokens are passed through unchanged (not hallucinated).
- Only explicitly listed transformations are applied.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .dialect_adapter import DialectAdapter, DialectAdaptation
from .semantic_preserver import (
    ExtractionResult,
    PreservationReport,
    SemanticPreserver,
)
from .slang_mapper import PhraseMatch, SlangMapper, TokenGloss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class NormalizationTrace:
    """
    Complete audit trail of every transformation applied.
    """
    input_text:      str
    dialect:         str
    after_dialect:   str
    after_phrases:   str
    after_tokens:    str
    final_output:    str
    phrase_matches:  List[PhraseMatch]       = field(default_factory=list)
    token_glosses:   List[TokenGloss]        = field(default_factory=list)
    dialect_changes: List[DialectAdaptation] = field(default_factory=list)


@dataclass
class NormalizationResult:
    """
    Result of the linguistic normalization pipeline.

    Attributes
    ----------
    input_text : str
        Original ASR transcript.
    normalized_text : str
        The normalized output. This is the only content this module produces.
    dialect : str
        Dialect label used (from M6 or "Standard" if unavailable).
    language : str
        Language label (from M5 or "unknown").
    preservation : PreservationReport
        Which semantic anchors were verified as preserved.
    extraction : ExtractionResult
        All semantic anchors extracted from the input.
    trace : NormalizationTrace
        Step-by-step transformation log.
    """
    input_text:       str
    normalized_text:  str
    dialect:          str
    language:         str
    preservation:     PreservationReport
    extraction:       ExtractionResult
    trace:            NormalizationTrace

    def to_dict(self) -> Dict[str, Any]:
        """Return the public-facing output dict."""
        return {
            "input":              self.input_text,
            "normalized":         self.normalized_text,
            "dialect":            self.dialect,
            "language":           self.language,
            "all_entities_preserved": self.preservation.all_preserved,
            "warnings":           self.preservation.warnings,
        }


# ---------------------------------------------------------------------------
# LinguisticNormalizer
# ---------------------------------------------------------------------------

class LinguisticNormalizer:
    """
    Deterministic, rule-based linguistic normalizer.

    Parameters
    ----------
    config : dict | None
        Parsed ``configs/ldm.yaml``.
    slang_mapper : SlangMapper | None
        Pre-constructed mapper (for reuse / testing).
    dialect_adapter : DialectAdapter | None
        Pre-constructed adapter (for reuse / testing).
    semantic_preserver : SemanticPreserver | None
        Pre-constructed preserver (for reuse / testing).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        slang_mapper: Optional[SlangMapper] = None,
        dialect_adapter: Optional[DialectAdapter] = None,
        semantic_preserver: Optional[SemanticPreserver] = None,
    ) -> None:
        self._cfg       = config or {}
        norm_cfg        = self._cfg.get("normalizer", {})

        self._capitalize = norm_cfg.get("capitalize_output", True)
        self._terminator = norm_cfg.get("sentence_terminator", ".")
        self._drop_unk   = norm_cfg.get("drop_unknown_tamil_tokens", False)

        self._slang   = slang_mapper       or SlangMapper(config=config)
        self._dialect = dialect_adapter    or DialectAdapter(config=config)
        self._sem     = semantic_preserver or SemanticPreserver(config=config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        text: str,
        dialect: str = "Standard",
        language: str = "unknown",
    ) -> NormalizationResult:
        """
        Normalize *text* and return a full NormalizationResult.

        Parameters
        ----------
        text : str
            Raw ASR transcript.
        dialect : str
            Dialect label from M6, e.g. "Chennai", "Madurai".
            Defaults to "Standard".
        language : str
            Language label from M5, e.g. "ta", "tanglish", "ta-en".
        """
        if not text or not text.strip():
            return self._empty_result(text, dialect, language)

        # Step 1 — Extract semantic anchors BEFORE any transformation
        extraction = self._sem.extract(text)

        # Step 2 — Dialect-specific adaptation
        after_dialect, dialect_changes = self._dialect.adapt(text, dialect)

        # Step 3 — Phrase-level slang mapping
        after_phrases, phrase_matches = self._slang.apply_phrases(after_dialect)

        # Step 4 — Token-level gloss substitution
        after_tokens, token_glosses = self._slang.apply_token_glosses(after_phrases)

        # Step 5 — Reconstruct / clean
        final_output = self._reconstruct(after_tokens)

        # Step 6 — Verify semantic preservation
        preservation = self._sem.verify(extraction, final_output)

        # Build trace
        trace = NormalizationTrace(
            input_text=text,
            dialect=dialect,
            after_dialect=after_dialect,
            after_phrases=after_phrases,
            after_tokens=after_tokens,
            final_output=final_output,
            phrase_matches=phrase_matches,
            token_glosses=token_glosses,
            dialect_changes=dialect_changes,
        )

        if not preservation.all_preserved:
            logger.warning(
                f"[Normalizer] Preservation issues for: {text!r}\n"
                + "\n".join(f"  - {w}" for w in preservation.warnings)
            )

        return NormalizationResult(
            input_text=text,
            normalized_text=final_output,
            dialect=dialect,
            language=language,
            preservation=preservation,
            extraction=extraction,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(self, text: str) -> str:
        """
        Clean up the glossed text and apply final formatting.

        Specifically:
        1. Collapse multiple whitespace into single spaces.
        2. Remove leading/trailing whitespace.
        3. Remove repeated words caused by double application of glosses.
        4. Capitalize first letter.
        5. Ensure sentence ends with punctuation.
        """
        # Collapse whitespace
        result = re.sub(r"\s{2,}", " ", text).strip()

        # Remove any lingering Tamil-script characters if drop_unknown enabled
        if self._drop_unk:
            result = re.sub(r"[\u0B80-\u0BFF]+", "", result).strip()
            result = re.sub(r"\s{2,}", " ", result)

        # Remove duplicate consecutive words (artifact of double-glossing)
        result = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", result, flags=re.IGNORECASE)

        # Capitalize
        if self._capitalize and result:
            result = result[0].upper() + result[1:]

        # Terminate
        if result and result[-1] not in ".!?":
            result += self._terminator

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, text: str, dialect: str, language: str) -> NormalizationResult:
        empty_extraction    = ExtractionResult()
        empty_preservation  = PreservationReport(all_preserved=True)
        empty_trace = NormalizationTrace(
            input_text=text, dialect=dialect, after_dialect="",
            after_phrases="", after_tokens="", final_output="",
        )
        return NormalizationResult(
            input_text=text,
            normalized_text="",
            dialect=dialect,
            language=language,
            preservation=empty_preservation,
            extraction=empty_extraction,
            trace=empty_trace,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def normalize_batch(
        self,
        texts: List[str],
        dialect: str = "Standard",
        language: str = "unknown",
    ) -> List[NormalizationResult]:
        """Normalize a list of texts and return one result per item."""
        return [self.normalize(t, dialect, language) for t in texts]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "LinguisticNormalizer":
        """Load config from YAML and return a ready normalizer."""
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "LinguisticNormalizer":
        """Return a normalizer with built-in defaults."""
        return cls(config=None)
