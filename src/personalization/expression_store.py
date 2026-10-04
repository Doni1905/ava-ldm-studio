"""
src/personalization/expression_store.py
======================================
User-specific multi-word common expressions store.

Allows users to define custom conversational shortcuts, idiomatic phrases,
or personal shorthand expressions that map to standard English actions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .profile import sanitize_input

logger = logging.getLogger(__name__)


@dataclass
class PersonalExpressionEntry:
    """A personalized multi-word expression mapping."""
    expression: str
    normalized: str
    category: str = "EXPRESSION"
    usage_count: int = 0
    token_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalExpressionEntry":
        return cls(
            expression=data["expression"],
            normalized=data["normalized"],
            category=data.get("category", "EXPRESSION"),
            usage_count=data.get("usage_count", 0),
            token_count=data.get("token_count", len(data["expression"].split())),
        )


class CommonExpressionStore:
    """Manages persistent personalized multi-word expressions."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._entries: Dict[str, PersonalExpressionEntry] = {}
        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_expression(
        self,
        expression: str,
        normalized: str,
        category: str = "EXPRESSION",
    ) -> PersonalExpressionEntry:
        """Register a personal multi-word expression."""
        expr_clean = sanitize_input(expression).lower()
        norm_clean = sanitize_input(normalized)
        tokens = expr_clean.split()

        entry = PersonalExpressionEntry(
            expression=expr_clean,
            normalized=norm_clean,
            category=category.upper(),
            token_count=len(tokens),
        )
        self._entries[expr_clean] = entry
        if self.storage_path:
            self.save()
        return entry

    def get_expression(self, expression: str) -> Optional[PersonalExpressionEntry]:
        """Look up a registered personal expression."""
        return self._entries.get(expression.strip().lower())

    def remove_expression(self, expression: str) -> bool:
        """Remove a registered personal expression."""
        key = expression.strip().lower()
        if key in self._entries:
            del self._entries[key]
            if self.storage_path:
                self.save()
            return True
        return False

    def list_expressions(self) -> List[PersonalExpressionEntry]:
        """Return all user expressions sorted longest-first."""
        return sorted(self._entries.values(), key=lambda e: (-e.token_count, e.expression))

    def clear(self) -> None:
        """Clear all custom expressions."""
        self._entries.clear()
        if self.storage_path and self.storage_path.exists():
            self.save()

    def apply_to_text(self, text: str) -> str:
        """
        Apply personalized expressions to text using longest-match-first priority.
        """
        if not text or not self._entries:
            return text

        result = text
        for entry in self.list_expressions():
            pattern = re.compile(
                r"\b" + re.escape(entry.expression).replace(r"\ ", r"\s+") + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(result):
                entry.usage_count += 1
                result = pattern.sub(entry.normalized, result)

        return result

    def save(self) -> None:
        """Persist expressions to JSON."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._entries.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load expressions from JSON."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = {k: PersonalExpressionEntry.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load user expressions from {self.storage_path}: {e}")
