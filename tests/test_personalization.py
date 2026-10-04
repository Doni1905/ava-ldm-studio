"""
tests/test_personalization.py
=============================
Unit tests for the M13 Personalization module.

Verifies:
1. Profile Lifecycle (create, retrieve, update, reset)
2. User Vocabulary Store (add, lookup, apply, remove, persistence)
3. Common Expression Store (multi-word mapping, usage count, persistence)
4. Conflict Resolution (Explicit user instructions override preferences)
5. Sensitive Information Filtering (Rejecting passwords, OTPs, card numbers)
6. Disk Persistence & Reloading across fresh instances
"""

import io
import shutil
import sys
import unittest
from pathlib import Path

# Safe UTF-8 reconfiguration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.personalization.expression_store import CommonExpressionStore
from src.personalization.preference_store import PreferenceStore
from src.personalization.profile import UserProfile, sanitize_input
from src.personalization.vocabulary import UserVocabularyStore


class TestProfileLifecycle(unittest.TestCase):
    def setUp(self):
        self.test_dir = _ROOT / "data" / "test_personalization_lifecycle"
        self.store = PreferenceStore(base_dir=self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_and_retrieve_profile(self):
        profile = self.store.create_profile(
            user_id="user_test_1",
            preferred_language="Tamil",
            preferred_dialect="Madurai",
            response_style="casual",
        )
        self.assertEqual(profile.user_id, "user_test_1")
        self.assertEqual(profile.preferred_dialect, "Madurai")
        self.assertEqual(profile.response_style, "casual")

        # Retrieve profile
        retrieved = self.store.retrieve_profile("user_test_1")
        self.assertEqual(retrieved.preferred_language, "Tamil")
        self.assertEqual(retrieved.preferred_dialect, "Madurai")

    def test_update_profile(self):
        self.store.create_profile(user_id="user_update")
        updated = self.store.update_profile(
            user_id="user_update",
            preferred_dialect="Chennai",
            response_style="detailed",
        )
        self.assertEqual(updated.preferred_dialect, "Chennai")
        self.assertEqual(updated.response_style, "detailed")

        # Verify on-disk persistence
        fresh_store = PreferenceStore(base_dir=self.test_dir)
        reloaded = fresh_store.retrieve_profile("user_update")
        self.assertEqual(reloaded.preferred_dialect, "Chennai")

    def test_reset_profile(self):
        self.store.create_profile(
            user_id="user_reset",
            preferred_dialect="Kongu",
            response_style="formal",
        )
        # Add custom vocab and expression
        vocab = self.store.get_vocabulary_store("user_reset")
        vocab.add_term("projx", "Project X")
        expr = self.store.get_expression_store("user_reset")
        expr.add_expression("log off now", "sign out")

        # Reset profile
        reset_prof = self.store.reset_profile("user_reset")
        self.assertEqual(reset_prof.preferred_dialect, "Standard")
        self.assertEqual(reset_prof.response_style, "concise")

        # Verify vocab and expressions are cleared
        self.assertEqual(len(vocab.list_terms()), 0)
        self.assertEqual(len(expr.list_expressions()), 0)

    def test_frequent_command_tracking(self):
        profile = self.store.create_profile("user_cmd")
        profile.record_command("remind me tomorrow")
        profile.record_command("remind me tomorrow")
        profile.record_command("call mother")

        top = profile.get_top_commands(limit=2)
        self.assertEqual(top[0][0], "remind me tomorrow")
        self.assertEqual(top[0][1], 2)
        self.assertEqual(top[1][0], "call mother")
        self.assertEqual(top[1][1], 1)


class TestUserVocabularyStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = _ROOT / "data" / "test_personalization_vocab"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = UserVocabularyStore(storage_path=self.test_dir / "vocab.json")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_get_term(self):
        self.vocab.add_term("ldm", "Linguistic Dialect Model", category="ACRONYM")
        entry = self.vocab.get_term("ldm")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.normalized, "Linguistic Dialect Model")
        self.assertEqual(entry.category, "ACRONYM")

    def test_apply_word_boundary_safety(self):
        self.vocab.add_term("ava", "Accentric Virtual Assistant")
        # Exact word match
        text = "Hello ava please help me."
        res = self.vocab.apply_to_text(text)
        self.assertIn("Accentric Virtual Assistant", res)

        # Word boundary: should not match inside longer words like 'available'
        safe_text = "The system is available now."
        res_safe = self.vocab.apply_to_text(safe_text)
        self.assertEqual(res_safe, safe_text)

    def test_remove_term(self):
        self.vocab.add_term("temp", "temporary")
        self.assertTrue(self.vocab.remove_term("temp"))
        self.assertIsNone(self.vocab.get_term("temp"))
        self.assertFalse(self.vocab.remove_term("nonexistent"))


class TestCommonExpressionStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = _ROOT / "data" / "test_personalization_expr"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.expr_store = CommonExpressionStore(storage_path=self.test_dir / "expr.json")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_apply_expression(self):
        self.expr_store.add_expression("innaiku work over", "I have finished work today")
        text = "Dei innaiku work over remind me to relax."
        normalized = self.expr_store.apply_to_text(text)
        self.assertIn("I have finished work today", normalized)

    def test_longest_match_priority(self):
        self.expr_store.add_expression("meeting done", "meeting finished")
        self.expr_store.add_expression("meeting done for the day", "all meetings completed")

        text = "The meeting done for the day."
        out = self.expr_store.apply_to_text(text)
        self.assertEqual(out, "The all meetings completed.")


class TestConflictResolutionAndLinguisticInfluence(unittest.TestCase):
    """
    Verifies that personalization influences interpretation only where appropriate,
    and NEVER overrides explicit user instructions.
    """
    def setUp(self):
        self.test_dir = _ROOT / "data" / "test_personalization_conflict"
        self.store = PreferenceStore(base_dir=self.test_dir)
        self.store.create_profile("user_conflict", preferred_dialect="Madurai")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_explicit_instruction_overrides_preference(self):
        # User preference is 'Madurai', but user explicitly requests 'Chennai'
        dialect, reason = self.store.resolve_effective_dialect(
            user_id="user_conflict",
            detected_dialect="Standard",
            explicit_dialect="Chennai",
        )
        self.assertEqual(dialect, "Chennai")
        self.assertEqual(reason, "explicit_instruction")

    def test_detected_speech_overrides_preference(self):
        # User preference is 'Madurai', but speech detection found strong 'Kongu' markers ('ayya', 'la')
        dialect, reason = self.store.resolve_effective_dialect(
            user_id="user_conflict",
            detected_dialect="Kongu",
            explicit_dialect=None,
        )
        self.assertEqual(dialect, "Kongu")
        self.assertEqual(reason, "detected_speech")

    def test_neutral_speech_uses_preferred_dialect_fallback(self):
        # Speech is neutral 'Standard', no explicit instruction -> fallback to user preferred 'Madurai'
        dialect, reason = self.store.resolve_effective_dialect(
            user_id="user_conflict",
            detected_dialect="Standard",
            explicit_dialect=None,
        )
        self.assertEqual(dialect, "Madurai")
        self.assertEqual(reason, "personalized_preference")


class TestSensitiveInformationFiltering(unittest.TestCase):
    def test_reject_passwords_and_pins(self):
        with self.assertRaises(ValueError):
            sanitize_input("my password is secret123")

        with self.assertRaises(ValueError):
            sanitize_input("OTP code 123456")

        with self.assertRaises(ValueError):
            sanitize_input("bank PIN 9988")

    def test_reject_card_numbers(self):
        with self.assertRaises(ValueError):
            sanitize_input("Card 4111 2222 3333 4444")

    def test_allow_safe_terms(self):
        clean = sanitize_input("project_phoenix")
        self.assertEqual(clean, "project_phoenix")

        clean_expr = sanitize_input("remind me to pay bills")
        self.assertEqual(clean_expr, "remind me to pay bills")


if __name__ == "__main__":
    unittest.main()
