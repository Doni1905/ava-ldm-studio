"""
src/ldm/slang_mapper.py
========================
Tanglish slang and phrase mapping for AVA LDM normalization.

Responsibility
--------------
This module converts Tanglish/Tamil slang words and multi-word phrases
into their English equivalents WITHOUT generating new semantic content.

It does exactly two things:
  1. **Phrase substitution** — replaces multi-word Tanglish patterns
     (e.g. "remind pannu" → "remind me") using regex rules.
  2. **Token glossing** — maps single Tanglish tokens to English
     (e.g. "nalaiku" → "tomorrow") or drops discourse fillers
     (e.g. "dei" → "").

Design principles
-----------------
- Every substitution maps a *known* input form to a *known* output.
- Unknown tokens are never generated.
- The phrase list is ordered longest-match first to avoid partial matches.
- Rules are loaded from ``configs/ldm.yaml`` so they can be updated
  without touching Python source.

Source
------
The phrase patterns and word glosses are adapted (not copied) from the
TypeScript processor in ``src/lib/ldm/processor.ts``, which is the
ground-truth reference for this project.  The Python version is
refactored for testability: every transformation is a named, testable
function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Built-in phrase patterns (regex, replacement)
# Ordered longest → shortest to avoid partial matches.
# ---------------------------------------------------------------------------

_BUILTIN_PHRASES: List[Tuple[str, str]] = [
    # Reminder
    (r"\bremind\s+pannunga\b",       "remind me"),
    (r"\bremind\s+pannu\b",          "remind me"),
    (r"\bnyabagam\s+padutthu\b",     "remind me"),
    # Call / phone
    (r"\bcall\s+pottu\s+kudu\b",     "call"),
    (r"\bphone\s+pottu\s+kudu\b",    "call"),
    (r"\bcall\s+pannu\b",            "call"),
    (r"\bphone\s+pannu\b",           "call"),
    # Message
    (r"\bmessage\s+anuppu\b",        "send message to"),
    (r"\bmessage\s+podu\b",          "send message to"),
    # Alarm
    (r"\balarm\s+vai(?:kka|kku)?\b", "set alarm"),
    (r"\balarm\s+set\s+pannu\b",     "set alarm"),
    # Navigation
    (r"\bvazhi\s+kaatu\b",           "directions to"),
    (r"\broute\s+sollu\b",           "route to"),
    # Device control
    (r"\bkammi\s+pannu\b",           "reduce"),
    (r"\bkuraikka?\b",               "reduce"),
    (r"\bkoothu\b",                  "increase"),
    (r"\bethu?tthu\b",               "increase"),
    (r"\boff\s+pannu\b",             "turn off"),
    (r"\bon\s+pannu\b",              "turn on"),
    (r"\bon\s+panni\b",              "turn on"),
    # Search / open
    (r"\bsearch\s+pannu\b",          "search"),
    (r"\bopen\s+pannu\b",            "open"),
    # Time expressions
    (r"\beppadi\s+iruku\b",          "how is"),
    (r"\beppadi\s+iruka\b",          "how are you"),
    (r"\bkaalaila\b",                "in the morning"),
    (r"\bmaniku\b",                  "o'clock"),
    # Actions (multi-word only)
    (r"\bgym\s+poga\b",              "go to the gym"),
    (r"\bsubmit\s+panna\b",          "submit"),
    (r"\bsaapida\b",                 "take"),
    # Questions
    (r"\bmazhai\s+varuma\b",         "will it rain"),
    (r"\benna\s+panra\b",            "what are you doing"),
    (r"\byenna\s+panra\b",           "what are you doing"),
    # Time/status — NOTE: must appear before the single-token varen gloss is applied
    (r"\blate\s+(?:ah|a)\s+varen\b", "I will come late"),
    # Greeting
    (r"\beppadi\s+irukka\b",         "how are you"),
    (r"\beppadi\s+irukeenga\b",      "how are you"),
    # Negation forms
    (r"\bpanna\s+mateen\b",          "will not do"),
    (r"\bpanna\s+matten\b",          "will not do"),
    (r"\bpoga\s+mateen\b",           "will not go"),
    (r"\bpoga\s+matten\b",           "will not go"),
]

# ---------------------------------------------------------------------------
# Built-in word glosses
# Key: lowercase Tanglish token
# Value: English equivalent ("" = discourse filler, drop it)
# ---------------------------------------------------------------------------

_BUILTIN_GLOSSES: Dict[str, str] = {
    # Discourse fillers → drop
    "dei":      "",
    "da":       "",
    "la":       "",
    "pa":       "",
    "ppa":      "",
    "machi":    "",
    "bruh":     "",
    "ayya":     "",
    "thala":    "",
    "nu":       "",
    "anna":     "",
    "ku":       "",
    "kku":      "",
    "vendum":   "",
    "yov":      "",
    "ille":     "",
    "vaada":    "",
    "scene":    "",
    "ah":       "",
    "inga":     "",
    "anga":     "",
    "nga":      "",
    # Quantifiers / modifiers
    "oru":      "a",
    "onnu":     "a",
    "ondru":    "a",
    "konjam":   "a little",
    "romba":    "very",
    "bayangara":"very",
    "sikkiram": "quickly",
    "semma":    "great",
    "nalla":    "good",
    # Time
    "naalaikku":"tomorrow",
    "nalaiku":  "tomorrow",
    "naalaiku": "tomorrow",
    "inniku":   "today",
    "innaiku":  "today",
    "ipo":      "now",
    "ippo":     "now",
    "raathiri": "tonight",
    "kaalaila": "in the morning",
    # People
    "amma":     "mother",
    "appa":     "father",
    "thambi":   "my brother",
    "akka":     "my sister",
    # Content words
    "paatu":    "song",
    "padal":    "song",
    "podu":     "play",
    "vanakkam": "hello",
    "sollu":    "tell me",
    "sollunga": "tell me",
    "evlo":     "how much",
    "enna":     "what",
    "naan":     "I",
    "ava":      "AVA",
    "please":   "please",
    "aama":     "yes",
    "illai":    "no",
    "illya":    "no",
    "illa":     "no",
    "seri":     "okay",
    "okay":     "okay",
    "paarunga": "check",
    "paaru":    "check",
    "theriyum": "I know",
    "theriyathu":"I don't know",
    "puriyuthu": "I understand",
    "puriyala":  "I don't understand",
    # Single-word verb forms (kept here to avoid conflicting with multi-word phrases)
    "pogiren":  "going",
    "pokiren":  "going",
    "varen":    "coming",
    "vandhen":  "came",
    "vandhaan": "he came",
    "vandha":   "came",
}


# ---------------------------------------------------------------------------
# Compiled phrase table
# ---------------------------------------------------------------------------

@dataclass
class _CompiledPhrase:
    pattern:     re.Pattern
    replacement: str
    raw_pattern: str


def _compile_phrases(rules: List[Tuple[str, str]]) -> List[_CompiledPhrase]:
    compiled = []
    for raw, repl in rules:
        compiled.append(_CompiledPhrase(
            pattern=re.compile(raw, re.IGNORECASE),
            replacement=repl,
            raw_pattern=raw,
        ))
    return compiled


# ---------------------------------------------------------------------------
# SlangMapper
# ---------------------------------------------------------------------------

@dataclass
class PhraseMatch:
    """Record of a phrase-level substitution applied to the text."""
    original:    str
    replacement: str
    start:       int
    end:         int


@dataclass
class TokenGloss:
    """Record of a token-level substitution."""
    original: str
    gloss:    str    # empty string means it was a filler that was dropped


class SlangMapper:
    """
    Applies Tanglish → English phrase and token substitutions.

    Parameters
    ----------
    config : dict | None
        Parsed ``configs/ldm.yaml``.  If None, uses built-in defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        slang_cfg = (config or {}).get("slang", {})

        # Merge built-in glosses with config overrides
        self._glosses: Dict[str, str] = dict(_BUILTIN_GLOSSES)
        cfg_glosses = slang_cfg.get("glosses", {})
        self._glosses.update({k.lower(): v for k, v in cfg_glosses.items()})

        # Fillers from config (mark them as "" in the gloss map)
        for filler in slang_cfg.get("fillers", []):
            self._glosses.setdefault(filler.lower(), "")

        # Compile phrase patterns (built-in)
        self._phrases: List[_CompiledPhrase] = _compile_phrases(_BUILTIN_PHRASES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_phrases(self, text: str) -> Tuple[str, List[PhraseMatch]]:
        """
        Apply all phrase-level substitutions to *text*.

        Returns the transformed text and a list of applied matches.
        Applied left-to-right, longest pattern first.
        """
        matches: List[PhraseMatch] = []
        result = text.lower()  # phrases are case-insensitive

        for cp in self._phrases:
            # Find all matches *before* substitution to record positions
            for m in list(cp.pattern.finditer(result)):
                matches.append(PhraseMatch(
                    original=m.group(0),
                    replacement=cp.replacement,
                    start=m.start(),
                    end=m.end(),
                ))
            result = cp.pattern.sub(cp.replacement, result)

        return result, matches

    def apply_token_glosses(self, text: str) -> Tuple[str, List[TokenGloss]]:
        """
        Apply token-level gloss substitutions to *text*.

        Tokens not in the gloss dict are passed through unchanged.
        Empty-string glosses (fillers) are dropped.

        Returns the transformed text and a list of token changes.
        """
        applied: List[TokenGloss] = []
        tokens  = text.split()
        out_tokens: List[str] = []

        for tok in tokens:
            lower = tok.lower().strip(".,!?;:")
            if lower in self._glosses:
                gloss = self._glosses[lower]
                applied.append(TokenGloss(original=tok, gloss=gloss))
                if gloss:
                    out_tokens.append(gloss)
                # else: drop filler
            else:
                out_tokens.append(tok)

        return " ".join(out_tokens), applied

    def map(self, text: str) -> Tuple[str, List[PhraseMatch], List[TokenGloss]]:
        """
        Full two-pass mapping: phrases first, then token glosses.

        Returns (normalized_text, phrase_matches, token_glosses).
        """
        after_phrases, phrase_matches   = self.apply_phrases(text)
        after_tokens,  token_glosses    = self.apply_token_glosses(after_phrases)
        return after_tokens, phrase_matches, token_glosses

    def is_filler(self, token: str) -> bool:
        """Return True if *token* is a discourse filler (maps to "")."""
        return self._glosses.get(token.lower(), "UNKNOWN") == ""

    def gloss(self, token: str) -> Optional[str]:
        """Return the gloss for *token*, or None if not in the map."""
        return self._glosses.get(token.lower())

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SlangMapper":
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    @classmethod
    def default(cls) -> "SlangMapper":
        return cls(config=None)
