"""
src/ldm/semantic_preserver.py
==============================
Semantic anchor extraction and preservation verification.

Responsibility
--------------
Before normalization, this module extracts "semantic anchors" — entities,
temporal expressions, quantities, names, and negation markers — that MUST
survive the normalization step unchanged.

After normalization, it verifies that all extracted anchors appear in the
output, optionally raising a warning for any missing anchors.

Design principles
-----------------
- Only extracts; never generates.
- Entity extraction is deterministic (rule-based lookup against word lists).
- Negation is detected by scanning for negation tokens in both the input
  and the output — if negation was present in input but absent in output,
  a preservation error is raised.
- Quantities (digits, Tanglish number words) are extracted as strings
  and checked for presence in the output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity type constants
# ---------------------------------------------------------------------------
ET_TIME    = "time"
ET_DATE    = "date"
ET_PERSON  = "person"
ET_PLACE   = "location"
ET_APP     = "app"
ET_DEVICE  = "device_setting"
ET_TASK    = "task"
ET_QTY     = "quantity"
ET_NEG     = "negation"
ET_NAME    = "name"


# ---------------------------------------------------------------------------
# Default word lists (override via config)
# ---------------------------------------------------------------------------

_DEFAULT_TIME_WORDS: List[str] = [
    "tomorrow", "today", "tonight", "now", "morning", "evening",
    "night", "afternoon", "midnight",
    # Tanglish time words (after slang mapping they become English)
    "naalaikku", "nalaiku", "inniku", "innaiku", "ipo", "ippo",
    "raathiri", "kaalaila",
]

_DEFAULT_NEGATION: List[str] = [
    "illai", "illya", "illa", "matteen", "matten",
    "no", "not", "never", "don't", "dont", "won't", "wont",
    "cannot", "can't", "didn't", "won't", "mateen",
]

_DEFAULT_APPS: List[str] = [
    "whatsapp", "camera", "youtube", "instagram", "gallery",
    "spotify", "maps", "gmail", "telegram", "netflix",
]

_DEFAULT_PLACES: List[str] = [
    "chennai", "coimbatore", "bangalore", "madurai", "office",
    "home", "college", "school", "hospital", "airport",
    "tirunelveli", "salem", "erode",
]

_DEFAULT_DEVICE: List[str] = [
    "volume", "brightness", "wifi", "bluetooth",
    "torch", "flashlight", "hotspot", "data",
]

_DEFAULT_PEOPLE: Dict[str, str] = {
    "amma":   "mother",
    "appa":   "father",
    "thambi": "brother",
    "akka":   "sister",
    "anna":   "brother",
    "friend": "friend",
    "sister": "sister",
    "brother":"brother",
    "wife":   "wife",
    "husband":"husband",
    "mother": "mother",
    "father": "father",
}

# Tanglish Tamil number words → digits/English
_TAMIL_NUMBERS: Dict[str, str] = {
    "onnu":    "1",
    "rendu":   "2",
    "moonu":   "3",
    "naalu":   "4",
    "anju":    "5",
    "aaru":    "6",
    "ezhu":    "7",
    "ettu":    "8",
    "ombodu":  "9",
    "pathu":   "10",
    "pathinardu": "14",
    "irupathu": "20",
    "muppadu": "30",
}

# Regex for clock times — requires either HH:MM format or am/pm suffix
# Bare digits (e.g. "5" in "buy 5 books") are NOT matched.
_CLOCK_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b"         # HH:MM (optional am/pm)
    r"|"
    r"\b\d{1,2}\s*(?:am|pm)\b",                    # H am/pm (requires am/pm)
    re.IGNORECASE
)
# Regex for standalone digit quantities (not clock times)
_DIGIT_RE = re.compile(r"\b\d+\b")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SemanticAnchor:
    """A single extracted semantic anchor from the input text."""
    entity_type: str     # one of the ET_* constants
    value:       str     # the extracted value (normalized English form)
    raw_form:    str     # original text fragment
    preserved:   Optional[bool] = None   # set after verification


@dataclass
class ExtractionResult:
    """All semantic anchors extracted from a text."""
    anchors:      List[SemanticAnchor]  = field(default_factory=list)
    has_negation: bool                  = False
    quantities:   List[str]             = field(default_factory=list)
    time_refs:    List[str]             = field(default_factory=list)

    @property
    def entity_types_present(self) -> Set[str]:
        return {a.entity_type for a in self.anchors}

    @property
    def all_values(self) -> List[str]:
        return [a.value for a in self.anchors]


@dataclass
class PreservationReport:
    """Result of checking whether all anchors survived normalization."""
    all_preserved:  bool
    missing_anchors: List[SemanticAnchor]  = field(default_factory=list)
    warnings:        List[str]             = field(default_factory=list)


# ---------------------------------------------------------------------------
# SemanticPreserver
# ---------------------------------------------------------------------------

class SemanticPreserver:
    """
    Extracts and verifies semantic anchors in LDM normalization.

    Parameters
    ----------
    config : dict | None
        Parsed ``configs/ldm.yaml``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        sem_cfg = (config or {}).get("semantic", {})

        self._time_words  = set(w.lower() for w in sem_cfg.get("time_words",  _DEFAULT_TIME_WORDS))
        self._neg_markers = set(
            str(w).lower() for w in sem_cfg.get("negation_markers", _DEFAULT_NEGATION)
            if w is not None and w is not True  # filter out YAML bool misparses
        )
        self._apps        = set(w.lower() for w in sem_cfg.get("apps",         _DEFAULT_APPS))
        self._places      = set(w.lower() for w in sem_cfg.get("places",       _DEFAULT_PLACES))
        self._device      = set(w.lower() for w in sem_cfg.get("device_targets", _DEFAULT_DEVICE))
        self._people      = {k.lower(): v for k, v in sem_cfg.get("people", _DEFAULT_PEOPLE).items()}

        norm_cfg = (config or {}).get("normalizer", {})
        self._strict = norm_cfg.get("strict_entity_preservation", True)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all semantic anchors from *text*.

        Works on both raw Tanglish and post-slang-mapping text.
        """
        lower  = text.lower()
        tokens = re.split(r"[\s.,!?;:\"'()]+", lower)
        tokens = [t for t in tokens if t]

        result = ExtractionResult()

        # --- Negation ---
        for neg in self._neg_markers:
            if neg in tokens or neg in lower:
                result.has_negation = True
                result.anchors.append(SemanticAnchor(
                    entity_type=ET_NEG,
                    value="NEGATION",
                    raw_form=neg,
                ))
                break  # one negation flag is enough

        # --- Clock times ---
        for m in _CLOCK_RE.finditer(text):
            val = m.group(0).strip()
            result.anchors.append(SemanticAnchor(ET_TIME, val, val))
            result.time_refs.append(val)

        # --- Time words ---
        for tok in tokens:
            if tok in self._time_words and tok not in [a.raw_form for a in result.anchors]:
                english = self._time_english(tok)
                result.anchors.append(SemanticAnchor(ET_DATE, english, tok))
                result.time_refs.append(english)

        # --- Digit quantities ---
        existing_vals = {a.value for a in result.anchors}
        for m in _DIGIT_RE.finditer(text):
            val = m.group(0)
            if val not in existing_vals:
                result.anchors.append(SemanticAnchor(ET_QTY, val, val))
                result.quantities.append(val)
                existing_vals.add(val)

        # --- Tamil number words ---
        for tok in tokens:
            if tok in _TAMIL_NUMBERS:
                eng_num = _TAMIL_NUMBERS[tok]
                result.anchors.append(SemanticAnchor(ET_QTY, eng_num, tok))
                result.quantities.append(eng_num)

        # --- People ---
        for tok in tokens:
            if tok in self._people:
                eng = self._people[tok]
                result.anchors.append(SemanticAnchor(ET_PERSON, eng, tok))

        # --- Apps ---
        for tok in tokens:
            if tok in self._apps:
                result.anchors.append(SemanticAnchor(ET_APP, tok, tok))

        # --- Places ---
        for tok in tokens:
            if tok in self._places:
                result.anchors.append(SemanticAnchor(ET_PLACE, tok, tok))

        # --- Device settings ---
        for tok in tokens:
            if tok in self._device:
                result.anchors.append(SemanticAnchor(ET_DEVICE, tok, tok))

        # --- Task keywords (hardcoded high-value tasks) ---
        for task_kw in ("assignment", "medicine", "meeting", "bill", "electricity"):
            if task_kw in lower:
                result.anchors.append(SemanticAnchor(ET_TASK, task_kw, task_kw))

        return result

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        extraction: ExtractionResult,
        normalized_text: str,
    ) -> PreservationReport:
        """
        Check that all semantic anchors in *extraction* appear in
        *normalized_text*.

        Parameters
        ----------
        extraction : ExtractionResult
            Anchors extracted from the original input.
        normalized_text : str
            The output of the normalization pipeline.

        Returns
        -------
        PreservationReport
        """
        norm_lower = normalized_text.lower()
        missing    = []
        warnings   = []

        for anchor in extraction.anchors:
            # Negation: check that the output also signals negation
            if anchor.entity_type == ET_NEG:
                neg_in_output = any(neg in norm_lower for neg in self._neg_markers) or \
                                any(w in norm_lower for w in ["not", "no", "never", "without"])
                if not neg_in_output:
                    anchor.preserved = False
                    missing.append(anchor)
                    warnings.append(
                        f"Negation present in input but absent in output: '{anchor.raw_form}'"
                    )
                else:
                    anchor.preserved = True
                continue

            # All other anchors: check value appears in output
            val_lower = anchor.value.lower()
            if val_lower in norm_lower:
                anchor.preserved = True
            else:
                anchor.preserved = False
                missing.append(anchor)
                msg = (
                    f"Entity '{anchor.entity_type}' value '{anchor.value}' "
                    f"(from '{anchor.raw_form}') not found in output."
                )
                warnings.append(msg)
                if self._strict:
                    logger.warning(msg)

        return PreservationReport(
            all_preserved=len(missing) == 0,
            missing_anchors=missing,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_english(self, tanglish_time: str) -> str:
        """Map a Tanglish time word to its English equivalent."""
        mapping = {
            "naalaikku": "tomorrow", "nalaiku": "tomorrow", "naalaiku": "tomorrow",
            "inniku": "today", "innaiku": "today",
            "ipo": "now", "ippo": "now",
            "raathiri": "tonight",
            "kaalaila": "in the morning",
        }
        return mapping.get(tanglish_time, tanglish_time)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SemanticPreserver":
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "SemanticPreserver":
        return cls(config=None)
