"""
src/personalization/vocabulary.py
=================================
User-specific vocabulary and custom terminology store.

Enables users to register personalized terminology (e.g., project codenames,
local acronyms, domain jargon) that the LDM normalizer should preserve or map.
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
class VocabularyEntry:
    """A personalized vocabulary mapping."""
    term: str
    normalized: str
    category: str = "CUSTOM"
    usage_count: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyEntry":
        return cls(
            term=data["term"],
            normalized=data["normalized"],
            category=data.get("category", "CUSTOM"),
            usage_count=data.get("usage_count", 0),
            notes=data.get("notes", ""),
        )


class UserVocabularyStore:
    """Manages persistent personalized vocabulary for a user."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._entries: Dict[str, VocabularyEntry] = {}
        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_term(
        self,
        term: str,
        normalized: str,
        category: str = "CUSTOM",
        notes: str = "",
    ) -> VocabularyEntry:
        """Register or update a user-specific vocabulary term."""
        term_clean = sanitize_input(term).lower()
        norm_clean = sanitize_input(normalized)

        entry = VocabularyEntry(
            term=term_clean,
            normalized=norm_clean,
            category=category.upper(),
            notes=notes.strip(),
        )
        self._entries[term_clean] = entry
        if self.storage_path:
            self.save()
        return entry

    def get_term(self, term: str) -> Optional[VocabularyEntry]:
        """Lookup a user vocabulary term by exact case-insensitive match."""
        return self._entries.get(term.strip().lower())

    def remove_term(self, term: str) -> bool:
        """Remove a term from user vocabulary."""
        key = term.strip().lower()
        if key in self._entries:
            del self._entries[key]
            if self.storage_path:
                self.save()
            return True
        return False

    def list_terms(self) -> List[VocabularyEntry]:
        """Return all user-defined vocabulary entries."""
        return list(self._entries.values())

    def clear(self) -> None:
        """Clear all custom vocabulary."""
        self._entries.clear()
        if self.storage_path and self.storage_path.exists():
            self.save()

    def apply_to_text(self, text: str) -> str:
        """
        Substitute registered user vocabulary terms in text using exact word boundaries.
        Increments usage counts for matched terms.
        """
        if not text or not self._entries:
            return text

        result = text
        for term, entry in sorted(self._entries.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            if pattern.search(result):
                entry.usage_count += 1
                result = pattern.sub(entry.normalized, result)

        return result

    def save(self) -> None:
        """Persist user vocabulary to JSON."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._entries.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load user vocabulary from JSON."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = {k: VocabularyEntry.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load user vocabulary from {self.storage_path}: {e}")
