"""
src/linguistic/code_mix_detector.py
=====================================
Tamil-English code-mixing analysis for AVA LDM Studio.

This module builds on ``LanguageDetector`` to provide richer code-mixing
metrics at both the **token** and **phrase** level.

Key metrics computed
--------------------

1. **Code Mixing Index (CMI)** [0.0, 1.0]
   Based on the formula from Das & Gambäck (2014):
       CMI = 1 - (max(n_lang_i) / N)
   where N = total classified (non-OTHER) tokens and n_lang_i is the
   token count for language i. A CMI of 0.0 means monolingual; 1.0 means
   maximally mixed (equal split between all languages).

2. **Switch Point Count**
   Number of language boundaries in the token sequence.
   High switch-point count indicates fine-grained interleaving.

3. **M-index** (Multilingual Index)
   Based on entropy of language distribution across tokens.
   Normalised to [0.0, 1.0].

4. **Dominant Language per Span**
   Which language dominates each contiguous span of tokens.

Output schema
-------------
::

    {
        "language":       str,          # same as LanguageDetector
        "code_mixed":     bool,
        "tamil_ratio":    float,
        "english_ratio":  float,
        "confidence":     float,
        # Additional code-mix fields:
        "tanglish_ratio": float,
        "cmi":            float,        # Code Mixing Index
        "switch_points":  int,
        "m_index":        float,
        "dominant_spans": list[dict]
    }

Design note
-----------
This module deliberately does **not** generate or modify any text.
It only analyses the input token sequence and returns descriptive metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .language_detector import (
    LanguageDetectionResult,
    LanguageDetector,
    TaggedToken,
    TokenTag,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpanInfo:
    """A contiguous run of tokens sharing the same dominant language."""
    start_idx:  int
    end_idx:    int       # exclusive
    language:   str       # "ta" | "en" | "tanglish" | "mixed" | "other"
    tokens:     List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class CodeMixResult:
    """
    Complete code-mixing analysis result.

    Fields matching the requested JSON output schema are marked (★).
    """
    # --- Requested output fields ---
    language:       str           # ★
    code_mixed:     bool          # ★
    tamil_ratio:    float         # ★
    english_ratio:  float         # ★
    confidence:     float         # ★

    # --- Extended code-mix metrics ---
    tanglish_ratio: float         = 0.0
    cmi:            float         = 0.0    # Code Mixing Index (Das & Gambäck)
    switch_points:  int           = 0      # language boundary count
    m_index:        float         = 0.0    # Multilingual Index (entropy-based)
    dominant_spans: List[SpanInfo]= field(default_factory=list)
    token_details:  List[Dict]    = field(default_factory=list)
    evidence:       Dict[str, Any]= field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the public-facing output dict (requested schema)."""
        return {
            "language":       self.language,
            "code_mixed":     self.code_mixed,
            "tamil_ratio":    round(self.tamil_ratio, 4),
            "english_ratio":  round(self.english_ratio, 4),
            "confidence":     round(self.confidence, 4),
            "tanglish_ratio": round(self.tanglish_ratio, 4),
            "cmi":            round(self.cmi, 4),
            "switch_points":  self.switch_points,
            "m_index":        round(self.m_index, 4),
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Return the full result including token-level details and spans."""
        base = self.to_dict()
        base["token_details"] = self.token_details
        base["dominant_spans"] = [
            {
                "start": s.start_idx,
                "end":   s.end_idx,
                "language": s.language,
                "tokens":   s.tokens,
            }
            for s in self.dominant_spans
        ]
        base["evidence"] = self.evidence
        return base


# ---------------------------------------------------------------------------
# Code Mix Detector
# ---------------------------------------------------------------------------

class CodeMixDetector:
    """
    Analyse Tamil-English code-mixing in ASR transcripts.

    Parameters
    ----------
    config : dict | None
        Parsed contents of ``configs/linguistic.yaml`` (optional).
    detector : LanguageDetector | None
        Pre-constructed language detector (reuse for efficiency).
        If None, a default detector is created.

    Examples
    --------
    >>> cd = CodeMixDetector()
    >>> result = cd.detect("நான் school போகிறேன் because exams irukku")
    >>> result.code_mixed
    True
    >>> result.cmi
    0.5  # approximate
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        detector: Optional[LanguageDetector] = None,
    ) -> None:
        self.config = config or {}
        self.detector = detector or LanguageDetector(config=config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> CodeMixResult:
        """
        Full code-mix analysis of *text*.

        Parameters
        ----------
        text : str
            Raw ASR transcript. May contain Tamil Unicode and/or Roman chars.

        Returns
        -------
        CodeMixResult
        """
        lang_result: LanguageDetectionResult = self.detector.detect(text)
        tagged: List[TaggedToken] = lang_result.tokens

        # Build token sequence with normalised language tags
        classified = [t for t in tagged if t.tag != TokenTag.OTHER]

        if not classified:
            return self._empty_result(lang_result)

        # --- Code Mixing Index ---
        cmi = self._compute_cmi(classified)

        # --- Switch points ---
        switch_pts = self._count_switch_points(classified)

        # --- M-Index ---
        m_index = self._compute_m_index(classified)

        # --- Dominant spans ---
        spans = self._extract_spans(tagged)

        # --- Token detail records ---
        token_details = [
            {
                "token":         t.text,
                "tag":           t.tag.value,
                "tanglish_score": round(t.tanglish_score, 3),
            }
            for t in tagged
        ]

        evidence = {
            **lang_result.evidence,
            "cmi":          cmi,
            "switch_points":switch_pts,
            "m_index":      m_index,
        }

        return CodeMixResult(
            language=lang_result.language,
            code_mixed=lang_result.code_mixed,
            tamil_ratio=lang_result.tamil_ratio,
            english_ratio=lang_result.english_ratio,
            tanglish_ratio=lang_result.tanglish_ratio,
            confidence=lang_result.confidence,
            cmi=cmi,
            switch_points=switch_pts,
            m_index=m_index,
            dominant_spans=spans,
            token_details=token_details,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Code Mixing Index (Das & Gambäck 2014)
    # ------------------------------------------------------------------

    def _compute_cmi(self, classified: List[TaggedToken]) -> float:
        """
        Compute the Code Mixing Index.

        CMI = 1 - (max_lang_count / N)

        For a perfectly monolingual text: CMI = 0.0.
        For a 50/50 two-language text:    CMI = 0.5.

        MIXED tokens are split evenly between TA and EN for counting purposes.
        """
        n = len(classified)
        if n == 0:
            return 0.0

        counts: Dict[str, float] = {}
        for t in classified:
            if t.tag == TokenTag.MIXED:
                counts["ta"] = counts.get("ta", 0.0) + 0.5
                counts["en"] = counts.get("en", 0.0) + 0.5
            else:
                key = t.tag.value
                counts[key] = counts.get(key, 0.0) + 1.0

        if not counts:
            return 0.0

        dominant_count = max(counts.values())
        cmi = 1.0 - dominant_count / n
        return round(max(0.0, min(1.0, cmi)), 4)

    # ------------------------------------------------------------------
    # Switch points
    # ------------------------------------------------------------------

    def _count_switch_points(self, classified: List[TaggedToken]) -> int:
        """
        Count language switch points in the token sequence.

        A switch point occurs whenever adjacent tokens have different
        language tags (MIXED tokens create a switch on both sides).
        """
        if len(classified) < 2:
            return 0

        switches = 0
        prev_tag = classified[0].tag
        for t in classified[1:]:
            if t.tag != prev_tag:
                switches += 1
            prev_tag = t.tag
        return switches

    # ------------------------------------------------------------------
    # M-Index (entropy-based multilingual index)
    # ------------------------------------------------------------------

    def _compute_m_index(self, classified: List[TaggedToken]) -> float:
        """
        Compute the entropy-based Multilingual Index.

        M-index = -sum(p_i * log2(p_i)) / log2(K)

        where p_i is the proportion of tokens in language i and K is the
        number of languages present. Result is in [0.0, 1.0].
        A value of 0.0 is monolingual; 1.0 is maximally mixed.
        """
        n = len(classified)
        if n == 0:
            return 0.0

        counts: Dict[str, int] = {}
        for t in classified:
            key = t.tag.value
            counts[key] = counts.get(key, 0) + 1

        k = len(counts)
        if k <= 1:
            return 0.0

        entropy = -sum(
            (c / n) * math.log2(c / n)
            for c in counts.values()
            if c > 0
        )
        return round(entropy / math.log2(k), 4)

    # ------------------------------------------------------------------
    # Span extraction
    # ------------------------------------------------------------------

    def _extract_spans(self, tagged: List[TaggedToken]) -> List[SpanInfo]:
        """
        Extract contiguous spans of tokens sharing the same tag.

        Collapses consecutive tokens with the same language tag into
        one SpanInfo object (including OTHER spans).
        """
        if not tagged:
            return []

        spans: List[SpanInfo] = []
        cur_tag  = tagged[0].tag.value
        cur_start= 0
        cur_toks : List[str] = [tagged[0].text]

        for i, t in enumerate(tagged[1:], start=1):
            if t.tag.value == cur_tag:
                cur_toks.append(t.text)
            else:
                spans.append(SpanInfo(cur_start, i, cur_tag, cur_toks[:]))
                cur_tag   = t.tag.value
                cur_start = i
                cur_toks  = [t.text]

        spans.append(SpanInfo(cur_start, len(tagged), cur_tag, cur_toks))
        return spans

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "CodeMixDetector":
        """Load config from YAML and return a configured detector."""
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "CodeMixDetector":
        """Return a detector with built-in default configuration."""
        return cls(config=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(lang_result: LanguageDetectionResult) -> CodeMixResult:
        return CodeMixResult(
            language=lang_result.language,
            code_mixed=lang_result.code_mixed,
            tamil_ratio=lang_result.tamil_ratio,
            english_ratio=lang_result.english_ratio,
            confidence=lang_result.confidence,
        )
