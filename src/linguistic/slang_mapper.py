"""
src/linguistic/slang_mapper.py
================================
Dictionary-backed Tamil/Tanglish slang mapper.

Responsibility
--------------
This module is the **reference dictionary layer** for Tamil slang and
informal vocabulary.  It loads CSV dictionaries from ``data/linguistic/``
and provides structured token-level lookups that return:

  - The normalized English equivalent
  - The linguistic category of the mapping
  - The source dictionary it came from
  - Whether the match was exact or case-insensitive

Design contract
---------------
1. **Word-boundary only** — the mapper NEVER performs substring matches.
   "pannu" maps to "do", but "appannu" is NOT matched.
2. **No semantic fabrication** — if a token is not in the dictionary, it
   is returned unchanged.  The mapper never invents content.
3. **Filler detection** — entries with normalized value "<FILLER>" are
   treated as zero-content tokens that can be dropped by callers.
4. **CSV-editable** — all mappings live in CSV files, not Python code.
   Adding a new word means editing the CSV, not the source.

Distinction from src/ldm/slang_mapper.py
-----------------------------------------
``src/ldm/slang_mapper.py`` is a **pipeline component** that applies
phrase-level and token-level substitutions in the LDM normalization
pipeline.  It uses hardcoded Python dicts for fast pipeline execution.

This module (``src/linguistic/slang_mapper.py``) is the **authoritative
reference dictionary** that:
  - Loads from editable CSV files
  - Returns structured metadata (category, dialect, notes)
  - Is independently queryable for NLP research and evaluation
  - Can be used to regenerate/validate the LDM pipeline dicts

The two modules serve different purposes and are intentionally separate.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default CSV paths (relative to project root)
# ---------------------------------------------------------------------------

_DEFAULT_SLANG_CSV    = Path("data/linguistic/slang_dictionary.csv")
_DEFAULT_INFORMAL_CSV = Path("data/linguistic/informal_dictionary.csv")

# ---------------------------------------------------------------------------
# Linguistic category constants
# ---------------------------------------------------------------------------

class LexCategory(str, Enum):
    VERB              = "VERB"
    INFORMAL_VERB     = "INFORMAL_VERB"
    FILLER            = "FILLER"
    INTENSIFIER       = "INTENSIFIER"
    ADJECTIVE         = "ADJECTIVE"
    QUANTIFIER        = "QUANTIFIER"
    TEMPORAL          = "TEMPORAL"
    PRONOUN           = "PRONOUN"
    QUESTION          = "QUESTION"
    GREETING          = "GREETING"
    AFFIRMATIVE       = "AFFIRMATIVE"
    NEGATION          = "NEGATION"
    CONTRACTION       = "CONTRACTION"
    EXPRESSION        = "EXPRESSION"
    PARTICLE          = "PARTICLE"
    MANNER            = "MANNER"
    UNKNOWN           = "UNKNOWN"


# Sentinel: tokens whose normalized value is this should be dropped
FILLER_SENTINEL = "<FILLER>"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DictEntry:
    """A single dictionary entry."""
    informal:    str        # original informal form (lowercase key)
    normalized:  str        # normalized English equivalent
    category:    str        # LexCategory string
    dialect:     str        # "Standard" | "Chennai" | "Madurai" | "Kongu" | "Nellai"
    example_in:  str        # example input
    example_out: str        # example output
    notes:       str        # free-text notes
    source_file: str        # which CSV this came from

    @property
    def is_filler(self) -> bool:
        return self.normalized == FILLER_SENTINEL

    @property
    def display_normalized(self) -> str:
        """Return the normalized form, or empty string for fillers."""
        return "" if self.is_filler else self.normalized


@dataclass
class MappingResult:
    """Result of a single token lookup."""
    token:       str           # the original input token
    normalized:  str           # normalized form (empty string if filler)
    category:    str           # LexCategory
    is_filler:   bool          # True if the token should be dropped
    matched:     bool          # True if a dictionary entry was found
    entry:       Optional[DictEntry]  # the full entry (None if not matched)

    @property
    def should_drop(self) -> bool:
        """True if this token should be dropped from the output."""
        return self.is_filler

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token":      self.token,
            "normalized": self.normalized,
            "category":   self.category,
            "is_filler":  self.is_filler,
            "matched":    self.matched,
        }


# ---------------------------------------------------------------------------
# SlangMapper
# ---------------------------------------------------------------------------

class SlangMapper:
    """
    Dictionary-backed Tamil/Tanglish slang mapper.

    Loads entries from CSV files and provides word-boundary token lookups.

    Parameters
    ----------
    slang_csv : Path | str | None
        Path to the slang dictionary CSV.
    informal_csv : Path | str | None
        Path to the informal dictionary CSV.
    project_root : Path | str | None
        Root of the project (used to resolve relative CSV paths).
    """

    def __init__(
        self,
        slang_csv:    Optional[Path | str] = None,
        informal_csv: Optional[Path | str] = None,
        project_root: Optional[Path | str] = None,
    ) -> None:
        self._root = Path(project_root) if project_root else self._find_root()
        self._entries: Dict[str, DictEntry] = {}

        # Load dictionaries
        slang_path    = Path(slang_csv)    if slang_csv    else self._root / _DEFAULT_SLANG_CSV
        informal_path = Path(informal_csv) if informal_csv else self._root / _DEFAULT_INFORMAL_CSV

        self._load_csv(slang_path,    source_name="slang")
        self._load_csv(informal_path, source_name="informal")

        logger.info(f"SlangMapper: loaded {len(self._entries)} entries from 2 dictionaries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, token: str) -> MappingResult:
        """
        Look up a single token.

        Uses case-insensitive, word-exact matching only.
        Never matches substrings.

        Parameters
        ----------
        token : str
            A single word/token (no spaces).

        Returns
        -------
        MappingResult with matched=False if not found.
        """
        key = token.lower().strip(".,!?;:\"'()")
        entry = self._entries.get(key)

        if entry is None:
            return MappingResult(
                token=token,
                normalized=token,  # pass-through unchanged
                category=LexCategory.UNKNOWN.value,
                is_filler=False,
                matched=False,
                entry=None,
            )

        return MappingResult(
            token=token,
            normalized=entry.display_normalized,
            category=entry.category,
            is_filler=entry.is_filler,
            matched=True,
            entry=entry,
        )

    def map_tokens(self, text: str) -> List[MappingResult]:
        """
        Split *text* on whitespace and look up each token.

        Returns one MappingResult per token.  Unknown tokens have
        matched=False and normalized=original_token (pass-through).

        This does NOT perform phrase-level matching — use
        ``ExpressionMapper`` for multi-word expressions.
        """
        tokens = text.split()
        return [self.lookup(tok) for tok in tokens]

    def apply(self, text: str, drop_fillers: bool = True) -> str:
        """
        Apply token-level mappings to *text* and return normalized string.

        Tokens not in the dictionary are kept unchanged.
        Fillers are dropped if *drop_fillers* is True.

        This is a convenience wrapper; for detailed results use
        ``map_tokens()``.
        """
        results = self.map_tokens(text)
        out_tokens = []
        for r in results:
            if r.is_filler and drop_fillers:
                continue
            out_tokens.append(r.normalized)
        return " ".join(out_tokens)

    def is_filler(self, token: str) -> bool:
        """Return True if *token* is a discourse filler."""
        result = self.lookup(token)
        return result.is_filler

    def get_category(self, token: str) -> str:
        """Return the linguistic category of *token*, or UNKNOWN."""
        return self.lookup(token).category

    def all_entries(self) -> List[DictEntry]:
        """Return all loaded dictionary entries."""
        return list(self._entries.values())

    def entries_by_category(self, category: str) -> List[DictEntry]:
        """Return all entries with a given category."""
        cat_upper = category.upper()
        return [e for e in self._entries.values() if e.category == cat_upper]

    def entries_by_dialect(self, dialect: str) -> List[DictEntry]:
        """Return all entries for a given dialect."""
        return [e for e in self._entries.values() if e.dialect.lower() == dialect.lower()]

    def vocabulary(self) -> List[str]:
        """Return all informal forms (dictionary keys)."""
        return list(self._entries.keys())

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path, source_name: str) -> None:
        """Load entries from a CSV file into the internal dict."""
        if not path.exists():
            logger.warning(f"Dictionary not found: {path}")
            return

        loaded = 0
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("informal", "").strip().lower()
                if not key or key.startswith("#"):
                    continue

                # Determine category field name (slang has 'category', informal has 'category')
                category = row.get("category", "UNKNOWN").strip().upper()
                dialect  = row.get("dialect", "Standard").strip()

                entry = DictEntry(
                    informal=key,
                    normalized=row.get("normalized", key).strip(),
                    category=category,
                    dialect=dialect,
                    example_in=row.get("example_input", "").strip(),
                    example_out=row.get("example_output", "").strip(),
                    notes=row.get("notes", "").strip(),
                    source_file=source_name,
                )

                if key in self._entries:
                    logger.debug(f"Duplicate key '{key}' in {source_name}; overwriting")
                self._entries[key] = entry
                loaded += 1

        logger.debug(f"Loaded {loaded} entries from {path.name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_root() -> Path:
        """Walk up from this file to find the project root (has configs/)."""
        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            if (parent / "configs").exists():
                return parent
        return Path.cwd()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        slang_csv: str | Path,
        informal_csv: str | Path,
    ) -> "SlangMapper":
        """Load from explicit CSV file paths."""
        return cls(slang_csv=slang_csv, informal_csv=informal_csv)

    @classmethod
    def default(cls, project_root: Optional[str | Path] = None) -> "SlangMapper":
        """Load with default CSV paths relative to project root."""
        return cls(project_root=project_root)

