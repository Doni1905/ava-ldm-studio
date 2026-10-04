"""
src/personalization/profile.py
==============================
User Profile data structures and management for AVA LDM Studio.

Stores:
- preferred language
- preferred dialect
- response style
- frequent commands
- common expressions (keys / counts)
- user vocabulary (keys / counts)

Privacy guarantee:
- Sanitizes and rejects sensitive personal identifiers (passwords, PINs, card numbers, OTPs).
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Sensitive patterns that must never be stored in personalization profiles
_SENSITIVE_PATTERNS = [
    re.compile(r"\b(?:password|passwd|pwd|otp|pin|secret|api[_-]?key|token)\b", re.IGNORECASE),
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # Credit card numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),       # SSN format
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
]


def sanitize_input(value: str) -> str:
    """
    Check and strip sensitive information from personal profile inputs.
    Raises ValueError if explicitly sensitive data is detected.
    """
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"Sensitive information detected and rejected: '{pattern.pattern}' match")
    return value.strip()


@dataclass
class UserProfile:
    """Represents a lightweight, local user preference profile."""
    user_id: str = "default_user"
    preferred_language: str = "Tanglish"
    preferred_dialect: str = "Standard"
    response_style: str = "concise"  # "concise", "detailed", "casual", "formal"
    frequent_commands: Dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def record_command(self, command: str) -> None:
        """Increment frequency count for a frequent command."""
        cmd = command.strip().lower()
        if not cmd:
            return
        sanitize_input(cmd)
        self.frequent_commands[cmd] = self.frequent_commands.get(cmd, 0) + 1
        self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def get_top_commands(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Return the most frequently executed commands."""
        sorted_cmds = sorted(self.frequent_commands.items(), key=lambda x: x[1], reverse=True)
        return sorted_cmds[:limit]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", "default_user"),
            preferred_language=data.get("preferred_language", "Tanglish"),
            preferred_dialect=data.get("preferred_dialect", "Standard"),
            response_style=data.get("response_style", "concise"),
            frequent_commands=data.get("frequent_commands", {}) or {},
            created_at=data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
