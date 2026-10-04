"""
src/personalization/preference_store.py
======================================
Preference Store and Personalization Manager for AVA LDM Studio.

Coordinates:
- UserProfile persistence (create, retrieve, update, reset)
- UserVocabularyStore integration
- CommonExpressionStore integration
- Linguistic influence rules (preserving explicit user instructions over preferences)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .expression_store import CommonExpressionStore
from .profile import UserProfile, sanitize_input
from .vocabulary import UserVocabularyStore

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_DIR = Path("data/personalization")


class PreferenceStore:
    """
    Central local persistent storage coordinator for AVA personalization.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_STORAGE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._profiles_dir = self.base_dir / "profiles"
        self._vocab_dir = self.base_dir / "vocabulary"
        self._expr_dir = self.base_dir / "expressions"

        for d in [self._profiles_dir, self._vocab_dir, self._expr_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._active_profiles: Dict[str, UserProfile] = {}
        self._vocab_stores: Dict[str, UserVocabularyStore] = {}
        self._expr_stores: Dict[str, CommonExpressionStore] = {}

    # ------------------------------------------------------------------
    # Profile Lifecycle (Create, Retrieve, Update, Reset)
    # ------------------------------------------------------------------

    def create_profile(
        self,
        user_id: str = "default_user",
        preferred_language: str = "Tanglish",
        preferred_dialect: str = "Standard",
        response_style: str = "concise",
    ) -> UserProfile:
        """Create and persist a new user profile."""
        u_id = sanitize_input(user_id)
        profile = UserProfile(
            user_id=u_id,
            preferred_language=preferred_language,
            preferred_dialect=preferred_dialect,
            response_style=response_style,
        )
        self._active_profiles[u_id] = profile
        self._save_profile_to_disk(profile)
        logger.info(f"Created personalization profile for user '{u_id}'")
        return profile

    def retrieve_profile(self, user_id: str = "default_user") -> UserProfile:
        """Retrieve user profile from cache or disk; auto-creates if missing."""
        u_id = sanitize_input(user_id)
        if u_id in self._active_profiles:
            return self._active_profiles[u_id]

        file_path = self._profiles_dir / f"{u_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                self._active_profiles[u_id] = profile
                return profile
            except Exception as e:
                logger.warning(f"Failed to read profile {file_path}: {e}")

        # If not on disk, create default
        return self.create_profile(user_id=u_id)

    def update_profile(
        self,
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> UserProfile:
        """Update fields in an existing user profile."""
        profile = self.retrieve_profile(user_id)

        for key, val in kwargs.items():
            if hasattr(profile, key) and key not in ("user_id", "created_at"):
                if isinstance(val, str):
                    val = sanitize_input(val)
                setattr(profile, key, val)

        self._save_profile_to_disk(profile)
        logger.info(f"Updated personalization profile for user '{user_id}'")
        return profile

    def reset_profile(self, user_id: str = "default_user") -> UserProfile:
        """Reset profile to factory defaults and clear custom vocabulary/expressions."""
        u_id = sanitize_input(user_id)

        # 1. Reset profile values
        profile = UserProfile(user_id=u_id)
        self._active_profiles[u_id] = profile
        self._save_profile_to_disk(profile)

        # 2. Reset vocabulary
        vocab = self.get_vocabulary_store(u_id)
        vocab.clear()

        # 3. Reset expressions
        expr = self.get_expression_store(u_id)
        expr.clear()

        logger.info(f"Reset personalization profile and data for user '{u_id}'")
        return profile

    # ------------------------------------------------------------------
    # Sub-Store Accessors
    # ------------------------------------------------------------------

    def get_vocabulary_store(self, user_id: str = "default_user") -> UserVocabularyStore:
        """Get or initialize the persistent vocabulary store for a user."""
        u_id = sanitize_input(user_id)
        if u_id not in self._vocab_stores:
            path = self._vocab_dir / f"{u_id}.json"
            self._vocab_stores[u_id] = UserVocabularyStore(storage_path=path)
        return self._vocab_stores[u_id]

    def get_expression_store(self, user_id: str = "default_user") -> CommonExpressionStore:
        """Get or initialize the persistent expression store for a user."""
        u_id = sanitize_input(user_id)
        if u_id not in self._expr_stores:
            path = self._expr_dir / f"{u_id}.json"
            self._expr_stores[u_id] = CommonExpressionStore(storage_path=path)
        return self._expr_stores[u_id]

    # ------------------------------------------------------------------
    # Linguistic Influence & Conflict Resolution Rules
    # ------------------------------------------------------------------

    def resolve_effective_dialect(
        self,
        user_id: str,
        detected_dialect: Optional[str] = None,
        explicit_dialect: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Determine the effective dialect label to apply.

        Contract Rules:
        1. Explicit user instructions MUST NEVER be overridden.
           If explicit_dialect is supplied, it always takes precedence.
        2. Strong speech detection evidence takes precedence over static preferences.
           If detected_dialect is confident and regional (not neutral "Standard"), use it.
        3. If detected speech is neutral/Standard and user has a configured preferred_dialect,
           the preference is applied as a soft prior.

        Returns:
            (effective_dialect, decision_source)
        """
        # Rule 1: Explicit user instruction
        if explicit_dialect and explicit_dialect.strip().lower() != "auto":
            return explicit_dialect.strip(), "explicit_instruction"

        profile = self.retrieve_profile(user_id)

        # Rule 2: Strong regional dialect detected from speech markers
        if detected_dialect and detected_dialect.strip().lower() not in ("standard", "unknown", ""):
            return detected_dialect.strip(), "detected_speech"

        # Rule 3: Fallback to personalized preferred dialect
        if profile.preferred_dialect and profile.preferred_dialect.strip().lower() != "standard":
            return profile.preferred_dialect.strip(), "personalized_preference"

        return "Standard", "default_standard"

    def apply_personalization(
        self,
        user_id: str,
        text: str,
    ) -> str:
        """
        Apply personalized expressions and custom vocabulary to text.
        Executes without altering unrelated semantic content.
        """
        expr_store = self.get_expression_store(user_id)
        vocab_store = self.get_vocabulary_store(user_id)

        # 1. Multi-word expressions first (longest match)
        step1 = expr_store.apply_to_text(text)
        # 2. Single token vocabulary second
        step2 = vocab_store.apply_to_text(step1)
        return step2

    # ------------------------------------------------------------------
    # Internal Disk Helpers
    # ------------------------------------------------------------------

    def _save_profile_to_disk(self, profile: UserProfile) -> None:
        target = self._profiles_dir / f"{profile.user_id}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
