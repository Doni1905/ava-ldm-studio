"""
src/personalization/__init__.py
===============================
AVA Local Personalization Package.

Provides local persistent storage and personalization management for:
- UserProfile (preferred language, dialect, response style, frequent commands)
- UserVocabularyStore (custom terminology)
- CommonExpressionStore (personal multi-word expressions)
- PreferenceStore (persistence coordinator and dialect influence resolution)
"""

from .expression_store import CommonExpressionStore, PersonalExpressionEntry
from .preference_store import PreferenceStore
from .profile import UserProfile, sanitize_input
from .vocabulary import UserVocabularyStore, VocabularyEntry

__all__ = [
    "UserProfile",
    "sanitize_input",
    "UserVocabularyStore",
    "VocabularyEntry",
    "CommonExpressionStore",
    "PersonalExpressionEntry",
    "PreferenceStore",
]
