"""
src/linguistic/expression_mapper.py
=====================================
Dictionary-backed multi-word Tamil/Tanglish expression mapper.

Responsibility
--------------
This module handles **multi-word** informal expressions — phrases that
must be matched as a unit because individual token lookup would break
their meaning.

Examples:
  "paathu sollu"       → "check and tell me"
  "remind pannu"       → "remind me"
  "late ah varen"      → "I will come late"

Design contract
---------------
1. **Phrase-boundary matching** — expressions are matched using word
   boundaries (``\\b``), left-to-right, longest-expression-first.
   This prevents a short expression from consuming tokens that belong
   to a longer one.
2. **No side-effect on non-matching text** — text that does not match
   any expression is returned exactly as received.
3. **No semantic fabrication** — only explicitly listed expressions
   produce output. Unknown phrases pass through unchanged.
4. **CSV-editable** — all expressions live in
   ``data/linguistic/personal_expressions.csv``.  No Python changes
   required to add or edit an expression.
5. **Audit trail** — every applied substitution is recorded in
   ``ExpressionMatch`` objects returned alongside the output.

Longest-match-first ordering
-----------------------------
Expressions are compiled in descending token-count order so that
"remind panni kudu" (3 tokens) is tried before "remind pannu" (2 tokens).
Within the same token count, alphabetical order is used for determinism.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default CSV path (relative to project root)
_DEFAULT_EXPR_CSV = Path("data/linguistic/personal_expressions.csv")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpressionEntry:
    """A single multi-word expression entry."""
    expression:  str    # original informal expression (lowercase key)
    normalized:  str    # normalized English equivalent
    category:    str    # e.g. ASSISTANT_CMD, QUERY_CMD, STATUS_STMT
    dialect:     str    # Standard | Chennai | Madurai | Kongu | Nellai
    example_in:  str
    example_out: str
    notes:       str
    token_count: int    # number of whitespace-separated tokens (for ordering)


@dataclass
class ExpressionMatch:
    """Record of a single expression substitution applied to text."""
    expression:  str    # the matched expression (original form)
    normalized:  str    # the substituted normalized form
    category:    str
    start:       int    # character position in the *input* text
    end:         int    # character position in the *input* text


@dataclass
class ExpressionMapResult:
    """Full result of applying expression mapping to a text."""
    input_text:  str
    output_text: str
    matches:     List[ExpressionMatch] = field(default_factory=list)

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def was_modified(self) -> bool:
        return self.input_text.lower().strip() != self.output_text.lower().strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input":    self.input_text,
            "output":   self.output_text,
            "n_matches": self.n_matches,
            "matches": [
                {"expression": m.expression, "normalized": m.normalized, "category": m.category}
                for m in self.matches
            ],
        }


# ---------------------------------------------------------------------------
# Compiled expression (regex + metadata)
# ---------------------------------------------------------------------------

@dataclass
class _CompiledExpression:
    entry:   ExpressionEntry
    pattern: re.Pattern


# ---------------------------------------------------------------------------
# ExpressionMapper
# ---------------------------------------------------------------------------

class ExpressionMapper:
    """
    Multi-word Tamil/Tanglish expression mapper backed by CSV.

    Parameters
    ----------
    expr_csv : Path | str | None
        Path to the personal expressions CSV.
    project_root : Path | str | None
        Root of the project (used to resolve relative paths).
    """

    def __init__(
        self,
        expr_csv:     Optional[Path | str] = None,
        project_root: Optional[Path | str] = None,
    ) -> None:
        self._root = Path(project_root) if project_root else self._find_root()
        self._entries:  List[ExpressionEntry]    = []
        self._compiled: List[_CompiledExpression] = []

        csv_path = Path(expr_csv) if expr_csv else self._root / _DEFAULT_EXPR_CSV
        self._load_csv(csv_path)
        self._compile()

        logger.info(f"ExpressionMapper: loaded {len(self._entries)} expressions")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self, text: str) -> ExpressionMapResult:
        """
        Apply all expression mappings to *text*, longest-first.

        Each expression is matched using word boundaries (``\\b``).
        Overlapping matches are resolved by left-to-right priority after
        longest-first ordering.

        Parameters
        ----------
        text : str
            Input text (may be raw Tanglish or partially glossed).

        Returns
        -------
        ExpressionMapResult with the transformed text and applied matches.
        """
        result_text = text.lower()
        applied: List[ExpressionMatch] = []

        for ce in self._compiled:
            for m in list(ce.pattern.finditer(result_text)):
                applied.append(ExpressionMatch(
                    expression=m.group(0),
                    normalized=ce.entry.normalized,
                    category=ce.entry.category,
                    start=m.start(),
                    end=m.end(),
                ))
            result_text = ce.pattern.sub(ce.entry.normalized, result_text)

        return ExpressionMapResult(
            input_text=text,
            output_text=result_text.strip(),
            matches=applied,
        )

    def lookup(self, expression: str) -> Optional[ExpressionEntry]:
        """
        Look up a specific expression (exact match, case-insensitive).

        Returns the ExpressionEntry if found, else None.
        """
        key = expression.lower().strip()
        for entry in self._entries:
            if entry.expression == key:
                return entry
        return None

    def all_entries(self) -> List[ExpressionEntry]:
        """Return all loaded expression entries."""
        return list(self._entries)

    def entries_by_category(self, category: str) -> List[ExpressionEntry]:
        """Return expressions with the given category."""
        cat_upper = category.upper()
        return [e for e in self._entries if e.category == cat_upper]

    def vocabulary(self) -> List[str]:
        """Return all expression strings (keys)."""
        return [e.expression for e in self._entries]

    # ------------------------------------------------------------------
    # Loading and compilation
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"Expressions CSV not found: {path}")
            return

        loaded = 0
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                expr = row.get("expression", "").strip().lower()
                if not expr or expr.startswith("#"):
                    continue
                normalized = row.get("normalized", expr).strip()
                entry = ExpressionEntry(
                    expression=expr,
                    normalized=normalized,
                    category=row.get("category", "EXPRESSION").strip().upper(),
                    dialect=row.get("dialect", "Standard").strip(),
                    example_in=row.get("example_input", "").strip(),
                    example_out=row.get("example_output", "").strip(),
                    notes=row.get("notes", "").strip(),
                    token_count=len(expr.split()),
                )
                self._entries.append(entry)
                loaded += 1

        logger.debug(f"Loaded {loaded} expressions from {path.name}")

    def _compile(self) -> None:
        """
        Sort entries longest-first, then compile each into a regex pattern.

        Longest-first ensures that "remind panni kudu" (3 tokens) is
        tried before "remind pannu" (2 tokens), preventing the shorter
        match from consuming tokens that belong to the longer phrase.
        """
        # Sort: descending token count, then alphabetical for determinism
        sorted_entries = sorted(
            self._entries,
            key=lambda e: (-e.token_count, e.expression),
        )

        self._compiled = []
        for entry in sorted_entries:
            # Escape the expression and wrap in word boundaries
            escaped = re.escape(entry.expression)
            # Replace escaped spaces with flexible whitespace
            escaped = escaped.replace(r"\ ", r"\s+")
            pattern = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
            self._compiled.append(_CompiledExpression(entry=entry, pattern=pattern))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_root() -> Path:
        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            if (parent / "configs").exists():
                return parent
        return Path.cwd()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, expr_csv: str | Path) -> "ExpressionMapper":
        """Load from an explicit CSV path."""
        return cls(expr_csv=expr_csv)

    @classmethod
    def default(cls, project_root: Optional[str | Path] = None) -> "ExpressionMapper":
        """Load with default CSV paths relative to project root."""
        return cls(project_root=project_root)
