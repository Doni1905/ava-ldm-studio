"""
src/linguistic/language_detector.py
=====================================
Lightweight, rule-based language detector for Tamil, English, and Tanglish.

Architecture
------------
This module works purely with Unicode character ranges and a curated
Tanglish lexicon. It requires **no external ML model** and runs in O(n)
time proportional to input length.

Detection pipeline
------------------
1. **Character-level script scan** (deterministic, zero ambiguity)
   - Tamil Unicode block: U+0B80–U+0BFF
   - ASCII alpha characters: U+0041-U+005A, U+0061-U+007A
   - All other printable characters: tracked but not classified

2. **Token-level tagging**
   Each whitespace-delimited token is assigned one of:
   - ``TA``       — contains Tamil script characters
   - ``EN``       — ASCII-only, matches English vocab heuristic
   - ``TANGLISH`` — ASCII-only, matches Tanglish vocabulary/affixes
   - ``MIXED``    — contains both Tamil script and ASCII (e.g. "ROM55")
   - ``OTHER``    — digits, punctuation, symbols

3. **Ratio computation**
   - ``tamil_ratio``   = Tamil-script characters / all classified alpha chars
   - ``english_ratio`` = English tokens / all classified tokens
   - ``tanglish_score``= Tanglish indicator hits weighted by affix/vocab type

4. **Language label assignment** (ordered priority):
   - ``ta``         Tamil script dominant (>= threshold)
   - ``ta-en``      Significant Tamil + significant English/Tanglish tokens
   - ``tanglish``   Romanized Tamil dominant
   - ``en``         English dominant
   - ``unknown``    No clear signal

5. **Confidence score** [0.0, 1.0]
   Reflects certainty based on:
   - Degree of script dominance (how far above threshold)
   - Size of evidence (penalises very short texts)
   - Ambiguity (mixed signals reduce confidence)

Tanglish detection strategy
----------------------------
Tanglish is Tamil written in Roman script. It is detected using:
  a) A **curated word list** of common Tamil words and function words
     frequently written in Roman script.
  b) **Morphological affix matching**: Tamil grammar suffixes applied to
     Roman-script words (e.g., "-nga", "-la", "-nu", "-le", "-kku").
  c) Affixes carry double weight vs vocabulary hits.

No semantic content is generated — this module only classifies.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Tamil Unicode block: U+0B80–U+0BFF
# This covers the full range of Tamil script characters including:
# vowels, consonants, vowel signs, virama, numerics, and symbols.
# ---------------------------------------------------------------------------
_TAMIL_START = 0x0B80
_TAMIL_END   = 0x0BFF

# ---------------------------------------------------------------------------
# Tanglish Lexicon
# ---------------------------------------------------------------------------
# Core Tamil function words / common words written in Roman script.
# Grouped by part-of-speech / frequency for maintainability.
# All entries must be lowercase.
_TANGLISH_CORE_WORDS: frozenset[str] = frozenset({
    # Pronouns
    "naan", "nee", "avan", "aval", "nanga", "unga", "inga", "avanga",
    "avangaluku", "enakku", "unakku", "avanukkku", "avanukku",
    # Common verbs
    "sollu", "paaru", "vaa", "po", "ponga", "vandhaan", "vandha",
    "vandhen", "vandhutu", "irukku", "illai", "irukkaan", "irukken",
    "irukka", "pannunga", "pannu", "pandu", "solla", "paakka", "vaanga",
    "solren", "paakren", "kekkaren",
    # Adjectives / adverbs
    "romba", "konjam", "nalla", "kettavanu", "periya", "chinna",
    "azhaga", "nannaa", "bayangara", "appadiye", "ippadiye",
    # Discourse markers / particles
    "seri", "da", "di", "la", "le", "nu", "nga", "ponga",
    "macha", "machan", "dei", "thambi", "anna", "akka", "amma", "appa",
    # Question words
    "enna", "yenna", "epdi", "eppadi", "eppo", "yeppo", "enga", "yenga",
    "yaaru", "yaen", "yean", "ethu",
    # Common nouns
    "veetu", "padam", "ooru", "naadu", "kadai", "school", "college",
    "kaasu", "paiyan", "ponna", "pasanga", "payyan",
    # Connectives / particles
    "thaan", "ellam", "mattum", "kuda", "irundha", "aana", "aanaa",
    "appo", "ippo", "vera", "veru",
    # Verbalised particles / complements
    "vittaa", "vidu", "vittaen", "theriyum", "theriyathu", "puriyuthu",
    "puriyala", "sollitten", "ketten",
    # Intensifiers
    "super", "mass", "boss", "guru",  # borrowed but with Tamil suffix patterns
})

# Tamil grammar affixes applied to Roman-script word stems.
# These are highly diagnostic for Tanglish.
_TANGLISH_AFFIXES: Tuple[str, ...] = (
    # Dative / locative suffixes
    "kku", "ukku", "la", "le", "liye",
    # Plural / honorific
    "nga", "anga", "inga",
    # Verb tense suffixes
    "ren", "reen", "ran", "raan", "tten", "dhen", "chen",
    "kitten", "kitten", "ndhen", "ndhan",
    # Conditional / participle
    "na", "naa", "tha", "thaa", "ndhaa",
    # Negative
    "la", "matteen", "matten",
    # Quotative
    "nu", "nnu",
    # Aspectual
    "vutten", "tutten", "duten",
)

# Compiled regex: word that ends with a Tamil affix (min 4-char word to avoid false positives)
_AFFIX_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(r"^[a-z]{2,}" + re.escape(sfx) + r"$")
    for sfx in _TANGLISH_AFFIXES
)

# Regex for Tamil Unicode range (single compiled pattern)
_TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")

# Regex to split tokens on whitespace + punctuation (preserves Tamil punctuation ranges)
_TOKEN_SPLIT_RE = re.compile(r"[\s\u0964\u0965.,!?;:\"'()\[\]{}<>]+")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TokenTag(str, Enum):
    TA       = "ta"        # Tamil script
    EN       = "en"        # English (ASCII, not Tanglish)
    TANGLISH = "tanglish"  # Romanized Tamil
    MIXED    = "mixed"     # Both Tamil script and ASCII in same token
    OTHER    = "other"     # Digits, punctuation, symbols


@dataclass
class TaggedToken:
    """A single token with its language tag and supporting evidence."""
    text:              str
    tag:               TokenTag
    tamil_chars:       int    = 0
    ascii_alpha_chars: int    = 0
    tanglish_score:    float  = 0.0   # 0.0 = no signal, >0 = Tanglish evidence


@dataclass
class LanguageDetectionResult:
    """
    Complete detection result for a piece of text.

    Attributes
    ----------
    language : str
        One of: "ta" | "en" | "ta-en" | "tanglish" | "unknown"
    code_mixed : bool
        True when significant content appears in more than one language.
    tamil_ratio : float
        Fraction of alphabetic characters that are Tamil-script [0.0, 1.0].
    english_ratio : float
        Fraction of classified tokens that are English [0.0, 1.0].
    tanglish_ratio : float
        Fraction of classified tokens that are Tanglish [0.0, 1.0].
    confidence : float
        Detection confidence [0.0, 1.0].
    tokens : list[TaggedToken]
        Per-token classification details.
    evidence : dict
        Raw counts and intermediate scores for debugging/audit.
    """
    language:       str
    code_mixed:     bool
    tamil_ratio:    float
    english_ratio:  float
    tanglish_ratio: float
    confidence:     float
    tokens:         List[TaggedToken] = field(default_factory=list)
    evidence:       Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the public-facing output dict."""
        return {
            "language":      self.language,
            "code_mixed":    self.code_mixed,
            "tamil_ratio":   round(self.tamil_ratio, 4),
            "english_ratio": round(self.english_ratio, 4),
            "confidence":    round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Core Detector
# ---------------------------------------------------------------------------

class LanguageDetector:
    """
    Lightweight, rule-based detector for Tamil, English, and Tanglish text.

    Parameters
    ----------
    config : dict | None
        Parsed contents of ``configs/linguistic.yaml`` (optional).
        Defaults are applied for all missing keys.

    Examples
    --------
    >>> detector = LanguageDetector()
    >>> result = detector.detect("நான் school போகிறேன்")
    >>> result.language
    'ta-en'
    >>> result = detector.detect("romba nalla irukku da")
    >>> result.language
    'tanglish'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("detection", config or {})

        self.tamil_dom_thr    : float = cfg.get("tamil_dominant_threshold",   0.70)
        self.english_dom_thr  : float = cfg.get("english_dominant_threshold", 0.70)
        self.tanglish_thr     : float = cfg.get("tanglish_detection_threshold", 0.35)
        self.code_mix_min     : float = cfg.get("code_mix_minority_threshold", 0.20)
        self.script_w         : float = cfg.get("script_confidence_weight",   0.70)
        self.heuristic_w      : float = cfg.get("heuristic_confidence_weight",0.30)

        tng_cfg = (config or {}).get("tanglish", {})
        self.affix_weight : float = tng_cfg.get("affix_weight", 2.0)
        self.vocab_weight : float = tng_cfg.get("vocab_weight",  1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language(s) in *text*.

        Parameters
        ----------
        text : str
            Raw ASR transcript (may be Tamil script, Roman, or mixed).

        Returns
        -------
        LanguageDetectionResult
        """
        if not text or not text.strip():
            return self._empty_result()

        tokens = self._tokenize(text)
        tagged = [self._tag_token(tok) for tok in tokens if tok]

        # Aggregate character-level counts
        total_alpha  = sum(t.tamil_chars + t.ascii_alpha_chars for t in tagged)
        tamil_chars  = sum(t.tamil_chars for t in tagged)
        ascii_chars  = sum(t.ascii_alpha_chars for t in tagged)

        tamil_char_ratio = tamil_chars / max(total_alpha, 1)

        # Aggregate token-level counts (exclude OTHER tokens from ratios)
        classified = [t for t in tagged if t.tag != TokenTag.OTHER]
        n_cls       = max(len(classified), 1)
        n_ta        = sum(1 for t in classified if t.tag == TokenTag.TA)
        n_en        = sum(1 for t in classified if t.tag == TokenTag.EN)
        n_tng       = sum(1 for t in classified if t.tag == TokenTag.TANGLISH)
        n_mixed     = sum(1 for t in classified if t.tag == TokenTag.MIXED)

        token_tamil_ratio    = (n_ta + n_mixed) / n_cls
        token_english_ratio  = n_en / n_cls
        token_tanglish_ratio = n_tng / n_cls

        # Tanglish aggregate score per token
        avg_tanglish_score = (
            sum(t.tanglish_score for t in tagged) / max(len(tagged), 1)
        )

        evidence = {
            "n_tokens":           len(tagged),
            "n_classified":       len(classified),
            "n_ta":               n_ta,
            "n_en":               n_en,
            "n_tanglish":         n_tng,
            "n_mixed":            n_mixed,
            "tamil_chars":        tamil_chars,
            "ascii_chars":        ascii_chars,
            "total_alpha":        total_alpha,
            "tamil_char_ratio":   tamil_char_ratio,
            "token_tamil_ratio":  token_tamil_ratio,
            "token_english_ratio":token_english_ratio,
            "token_tanglish_ratio":token_tanglish_ratio,
            "avg_tanglish_score": avg_tanglish_score,
        }

        language, code_mixed = self._classify(
            tamil_char_ratio=tamil_char_ratio,
            token_tamil_ratio=token_tamil_ratio,
            token_english_ratio=token_english_ratio,
            token_tanglish_ratio=token_tanglish_ratio,
            avg_tanglish_score=avg_tanglish_score,
        )

        confidence = self._compute_confidence(
            language=language,
            tamil_char_ratio=tamil_char_ratio,
            token_tamil_ratio=token_tamil_ratio,
            token_english_ratio=token_english_ratio,
            token_tanglish_ratio=token_tanglish_ratio,
            n_tokens=len(tagged),
            avg_tanglish_score=avg_tanglish_score,
        )

        return LanguageDetectionResult(
            language=language,
            code_mixed=code_mixed,
            tamil_ratio=tamil_char_ratio,
            english_ratio=token_english_ratio,
            tanglish_ratio=token_tanglish_ratio,
            confidence=confidence,
            tokens=tagged,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Split text into word-level tokens, stripping empty entries."""
        parts = _TOKEN_SPLIT_RE.split(text.strip())
        return [p for p in parts if p]

    # ------------------------------------------------------------------
    # Token tagging
    # ------------------------------------------------------------------

    def _tag_token(self, token: str) -> TaggedToken:
        """Classify a single token and return a TaggedToken."""
        # Count character types
        tamil_chars     = sum(1 for c in token if _TAMIL_START <= ord(c) <= _TAMIL_END)
        ascii_alpha     = sum(1 for c in token if c.isascii() and c.isalpha())
        non_alpha_count = len(token) - tamil_chars - ascii_alpha

        # Pure Tamil script
        if tamil_chars > 0 and ascii_alpha == 0:
            return TaggedToken(text=token, tag=TokenTag.TA,
                               tamil_chars=tamil_chars, ascii_alpha_chars=0)

        # Pure ASCII alpha
        if ascii_alpha > 0 and tamil_chars == 0:
            tng_score = self._tanglish_score(token.lower())
            tag = TokenTag.TANGLISH if tng_score > 0 else TokenTag.EN
            return TaggedToken(text=token, tag=tag,
                               tamil_chars=0, ascii_alpha_chars=ascii_alpha,
                               tanglish_score=tng_score)

        # Mixed: both Tamil script and ASCII alpha in same token
        if tamil_chars > 0 and ascii_alpha > 0:
            return TaggedToken(text=token, tag=TokenTag.MIXED,
                               tamil_chars=tamil_chars,
                               ascii_alpha_chars=ascii_alpha)

        # Digits, punctuation, symbols only
        return TaggedToken(text=token, tag=TokenTag.OTHER)

    # ------------------------------------------------------------------
    # Tanglish scoring
    # ------------------------------------------------------------------

    def _tanglish_score(self, word: str) -> float:
        """
        Return a non-negative score indicating how strongly *word* (lowercase)
        is Tanglish. Zero means no evidence.

        Scoring:
        - Exact match in core vocabulary:           vocab_weight
        - Matches a Tamil morphological affix:      affix_weight
        (Both can apply for a max score of vocab_weight + affix_weight)
        """
        score = 0.0
        if word in _TANGLISH_CORE_WORDS:
            score += self.vocab_weight
        for pat in _AFFIX_PATTERNS:
            if pat.match(word):
                score += self.affix_weight
                break  # one affix hit is enough
        return score

    # ------------------------------------------------------------------
    # Language classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        *,
        tamil_char_ratio: float,
        token_tamil_ratio: float,
        token_english_ratio: float,
        token_tanglish_ratio: float,
        avg_tanglish_score: float,
    ) -> Tuple[str, bool]:
        """
        Apply threshold rules to assign a language label and code_mixed flag.

        Priority order:
        1. Tamil-dominant script → "ta"
        2. Code-mixed (both Tamil + English/Tanglish significant) → "ta-en"
        3. Tanglish (Roman-script Tamil dominant) → "tanglish"
        4. English dominant → "en"
        5. Unknown
        """
        # 1. Tamil-script dominant
        if tamil_char_ratio >= self.tamil_dom_thr:
            minor = token_english_ratio + token_tanglish_ratio
            code_mixed = minor >= self.code_mix_min
            return "ta", code_mixed

        # 2. Code-mixed: Tamil script present AND significant English/Tanglish
        if tamil_char_ratio >= self.code_mix_min:
            roman_contribution = token_english_ratio + token_tanglish_ratio
            if roman_contribution >= self.code_mix_min:
                return "ta-en", True

        # 3. Tanglish: Roman-script dominant with Tamil vocabulary/affixes
        if token_tanglish_ratio >= self.tanglish_thr or avg_tanglish_score >= self.vocab_weight:
            # Even if some English words present, Tanglish dominates
            code_mixed = token_english_ratio >= self.code_mix_min
            return "tanglish", code_mixed

        # 4. English dominant
        if token_english_ratio >= self.english_dom_thr:
            return "en", False

        # 5. Low evidence — attempt best guess
        if token_tanglish_ratio > 0 and token_tanglish_ratio > token_english_ratio:
            return "tanglish", False
        if token_english_ratio > 0:
            return "en", False
        if tamil_char_ratio > 0:
            return "ta", False

        return "unknown", False

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        *,
        language: str,
        tamil_char_ratio: float,
        token_tamil_ratio: float,
        token_english_ratio: float,
        token_tanglish_ratio: float,
        n_tokens: int,
        avg_tanglish_score: float,
    ) -> float:
        """
        Compute a confidence score in [0.0, 1.0].

        Strategy:
        - Script confidence: how far the dominant ratio exceeds threshold.
        - Heuristic confidence: based on Tanglish score magnitude.
        - Length penalty: very short texts are less reliable.
        - Code-mix ambiguity penalty: mixed texts are harder to classify.
        """
        # --- Script-based confidence ---
        if language == "ta":
            script_conf = min(1.0, tamil_char_ratio / max(self.tamil_dom_thr, 1e-9))
        elif language == "ta-en":
            # Confidence in code-mix = how evenly split it is (peak at 50/50)
            split_balance = 1.0 - abs(tamil_char_ratio - 0.5) * 2
            script_conf = max(0.3, split_balance)
        elif language == "en":
            script_conf = min(1.0, token_english_ratio / max(self.english_dom_thr, 1e-9))
        elif language == "tanglish":
            script_conf = min(1.0, token_tanglish_ratio / max(self.tanglish_thr, 1e-9))
        else:  # unknown
            script_conf = 0.1

        # --- Heuristic confidence (Tanglish evidence strength) ---
        if language in ("tanglish", "ta-en"):
            # Normalize: avg score of ~2.0 → full heuristic confidence
            heuristic_conf = min(1.0, avg_tanglish_score / 2.0)
        else:
            heuristic_conf = script_conf  # script is both signals

        # --- Combine ---
        raw_conf = self.script_w * script_conf + self.heuristic_w * heuristic_conf

        # --- Length penalty: penalise texts with < 3 tokens ---
        if n_tokens < 3:
            length_penalty = 0.6 + 0.13 * (n_tokens - 1)
            raw_conf *= length_penalty

        return round(min(1.0, max(0.0, raw_conf)), 4)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "LanguageDetector":
        """Load config from YAML file and return a configured detector."""
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "LanguageDetector":
        """Return a detector with built-in default configuration."""
        return cls(config=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> LanguageDetectionResult:
        return LanguageDetectionResult(
            language="unknown",
            code_mixed=False,
            tamil_ratio=0.0,
            english_ratio=0.0,
            tanglish_ratio=0.0,
            confidence=0.0,
            tokens=[],
            evidence={"note": "empty input"},
        )
