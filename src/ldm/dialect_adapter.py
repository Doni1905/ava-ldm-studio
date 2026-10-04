"""
src/ldm/dialect_adapter.py
===========================
Dialect-specific text adaptation for LDM normalization.

Responsibility
--------------
Each Tamil dialect has characteristic discourse particles and morphological
patterns that must be handled *before* the general slang mapper runs.
This module applies dialect-specific preprocessing so that downstream
components see cleaner text.

Dialect-specific rules are driven by the dialect label produced by M6
(``src/dialect/``).  When no dialect label is available, a "Standard"
fallback is applied.

Mapping between dialects and their markers
------------------------------------------
This module uses the exact same dialect → marker mapping already present in:
  - ``src/lib/ldm/processor.ts`` (DIALECT_MARKERS array)
  - ``configs/dialect.yaml`` (dialect_to_district, dialect marker sets)

Adaptation rules per dialect
-----------------------------
Chennai (Madras Bashai):
  - "vaada" → drop (masculine address marker)
  - "scene" → drop (context marker without content)
  - "semma" → "great"

Madurai:
  - "aama" → "yes"
  - "ennanga" → "what" (polite)
  - "thambi" used as address marker → drop (context: when at start of utterance)

Kongu (Coimbatore):
  - "la" (sentence-final) → drop
  - "ayya" (respectful address) → drop

Nellai (Tirunelveli):
  - "pa" / "ppa" (sentence-final) → drop
  - "ille" → "no" (when followed by "?" or at end)

Note: These are dialect-level adaptations on top of the general slang
mapper.  The slang mapper handles all dialects equally; this module handles
dialect-SPECIFIC forms that might have different meanings in other dialects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


# ---------------------------------------------------------------------------
# Dialect constants (same labels as M6)
# ---------------------------------------------------------------------------

DIALECT_CHENNAI  = "Chennai"
DIALECT_MADURAI  = "Madurai"
DIALECT_KONGU    = "Kongu"
DIALECT_NELLAI   = "Nellai"
DIALECT_STANDARD = "Standard"
DIALECT_UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# Dialect rule definitions
# ---------------------------------------------------------------------------

# Each rule is (pattern_str, replacement, description)
_DIALECT_RULES: Dict[str, List[Tuple[str, str, str]]] = {

    DIALECT_CHENNAI: [
        # Address markers that carry no content
        (r"\b(?:vaada|vada)\b",                 "",     "male address marker"),
        (r"\b(?:machi|machan)\b",               "",     "peer address marker"),
        (r"\b(?:dei|da)\b",                     "",     "informal address"),
        (r"\b(?:bruh|bro)\b",                   "",     "slang address"),
        # Scene marker
        (r"\bscene\s+theriyuma\b",              "do you know",  "scene inquiry"),
        (r"\bscene\b",                          "",     "scene filler"),
        # Semma as intensifier
        (r"\bsemma\b",                          "great",  "intensifier"),
    ],

    DIALECT_MADURAI: [
        # Polite address / agreement
        (r"\baama\b",                           "yes",   "affirmative"),
        (r"\bennanga\b",                        "what",  "polite inquiry"),
        (r"\bthambi\b(?=\s+ku|\s+kku|\s+\w)",  "brother", "address as noun"),
        (r"\bthambi\b",                         "",     "address filler"),
        # Question-final marker
        (r"\bnga\b",                            "",     "politeness marker"),
    ],

    DIALECT_KONGU: [
        # Sentence-final la (distinct from general 'la' filler)
        (r"\bla\b(?=\s*[?!.]|$)",              "",     "sentence-final particle"),
        (r"\bayya\b",                           "",     "respectful address"),
        (r"\byov\b",                            "",     "address marker"),
        # Kongu-specific verb forms
        (r"\birukku\b",                         "is",   "copula"),
        (r"\birukkaa\b",                        "is there",   "existence query"),
    ],

    DIALECT_NELLAI: [
        # Sentence-final pa / ppa
        (r"\bppa\b",                            "",     "sentence-final particle"),
        (r"\bpa\b(?=\s*[?!.]|$)",              "",     "sentence-final particle"),
        (r"\bpa\b",                             "",     "address particle"),
        # Nellai negation
        (r"\bille\b(?=\s*[?!.]|$)",            "no",   "negation/question"),
        (r"\bille\b",                           "no",   "negation"),
        # NOTE: 'kudu' (give) is intentionally NOT adapted here because
        # 'phone pottu kudu' / 'call pottu kudu' are multi-word phrase patterns
        # handled by SlangMapper and would be disrupted if 'kudu' is replaced first.
    ],

    DIALECT_STANDARD: [
        # Nothing dialect-specific for Standard; general slang mapper handles it
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DialectAdaptation:
    """Record of a single dialect-specific substitution."""
    dialect:     str
    description: str
    original:    str
    replacement: str


# ---------------------------------------------------------------------------
# DialectAdapter
# ---------------------------------------------------------------------------

class DialectAdapter:
    """
    Apply dialect-specific text adaptations before general normalization.

    Parameters
    ----------
    config : dict | None
        Parsed ``configs/ldm.yaml``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        dialect_cfg = (config or {}).get("dialect", {})

        # Build compiled rule tables per dialect
        self._compiled: Dict[str, List[Tuple[re.Pattern, str, str]]] = {}
        for dialect, rules in _DIALECT_RULES.items():
            compiled_rules = []
            for pat_str, replacement, desc in rules:
                compiled_rules.append((
                    re.compile(pat_str, re.IGNORECASE),
                    replacement,
                    desc,
                ))
            self._compiled[dialect] = compiled_rules

        # Additional filler lists from config
        self._config_fillers: Dict[str, List[str]] = {
            DIALECT_CHENNAI:  dialect_cfg.get("chennai_fillers",  []),
            DIALECT_MADURAI:  dialect_cfg.get("madurai_fillers",  []),
            DIALECT_KONGU:    dialect_cfg.get("kongu_fillers",    []),
            DIALECT_NELLAI:   dialect_cfg.get("nellai_fillers",   []),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        text: str,
        dialect: str,
    ) -> Tuple[str, List[DialectAdaptation]]:
        """
        Apply dialect-specific adaptations to *text*.

        Parameters
        ----------
        text : str
            Input text (may still contain Tanglish).
        dialect : str
            One of: "Chennai", "Madurai", "Kongu", "Nellai", "Standard",
            "unknown".  Unrecognized dialects fall through with no changes.

        Returns
        -------
        (adapted_text, list_of_adaptations)
        """
        adaptations: List[DialectAdaptation] = []

        # Normalize dialect name
        dialect_key = self._resolve_dialect(dialect)

        rules = self._compiled.get(dialect_key, [])
        result = text

        for pattern, replacement, desc in rules:
            matches = list(pattern.finditer(result))
            if matches:
                for m in matches:
                    adaptations.append(DialectAdaptation(
                        dialect=dialect_key,
                        description=desc,
                        original=m.group(0),
                        replacement=replacement,
                    ))
                result = pattern.sub(replacement, result)

        # Also remove any additional config-listed fillers for this dialect
        for filler in self._config_fillers.get(dialect_key, []):
            pat = re.compile(r"\b" + re.escape(filler) + r"\b", re.IGNORECASE)
            if pat.search(result):
                adaptations.append(DialectAdaptation(
                    dialect=dialect_key,
                    description="config filler",
                    original=filler,
                    replacement="",
                ))
                result = pat.sub("", result)

        # Clean up extra whitespace
        result = re.sub(r"\s{2,}", " ", result).strip()

        return result, adaptations

    def list_dialects(self) -> List[str]:
        """Return the list of supported dialect labels."""
        return list(_DIALECT_RULES.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dialect(dialect: str) -> str:
        """Normalize a dialect string to one of the known keys."""
        d = dialect.strip().title()
        # Handle variants like "Chennai (Madras Bashai)"
        for key in _DIALECT_RULES:
            if d.startswith(key) or key in d:
                return key
        return DIALECT_STANDARD

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DialectAdapter":
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "DialectAdapter":
        return cls(config=None)
