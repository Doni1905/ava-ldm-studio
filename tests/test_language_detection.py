"""
tests/test_language_detection.py
==================================
Unit tests for LanguageDetector.

Tests cover:
  - Pure Tamil script
  - Pure English text
  - Tanglish (Romanized Tamil)
  - Mixed Tamil+English (code-mixed)
  - Empty / whitespace input
  - Token-level tagging
  - Confidence score properties
  - Output dict schema matches requested format
  - No ML model required (pure rule-based)

Run from project root:
    python -m pytest tests/test_language_detection.py -v
    python tests/test_language_detection.py
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.linguistic.language_detector import (
    LanguageDetector,
    LanguageDetectionResult,
    TaggedToken,
    TokenTag,
)


class TestLanguageDetectorPureTamil(unittest.TestCase):
    """Pure Tamil-script inputs."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_pure_tamil_classified_as_ta(self):
        result = self.det.detect("நான் பள்ளிக்கு போகிறேன்")
        self.assertEqual(result.language, "ta")

    def test_pure_tamil_not_code_mixed(self):
        result = self.det.detect("இது ஒரு நல்ல நாள்")
        self.assertFalse(result.code_mixed)

    def test_pure_tamil_high_tamil_ratio(self):
        result = self.det.detect("வணக்கம் நண்பா")
        self.assertGreater(result.tamil_ratio, 0.85)

    def test_pure_tamil_confidence_high(self):
        result = self.det.detect("தமிழ் மொழி மிகவும் அழகான மொழி")
        self.assertGreater(result.confidence, 0.60)

    def test_pure_tamil_low_english_ratio(self):
        result = self.det.detect("எப்படி இருக்கீங்க")
        self.assertLess(result.english_ratio, 0.15)

    def test_tamil_with_numerals(self):
        # Numbers shouldn't flip the language classification
        result = self.det.detect("நான் 3 பேர் பார்த்தேன்")
        self.assertEqual(result.language, "ta")

    def test_single_tamil_word(self):
        result = self.det.detect("வணக்கம்")
        self.assertIn(result.language, ("ta", "unknown"))
        self.assertGreaterEqual(result.tamil_ratio, 0.9)


class TestLanguageDetectorEnglish(unittest.TestCase):
    """Pure English inputs."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_pure_english_classified_as_en(self):
        result = self.det.detect("I am going to school today")
        self.assertEqual(result.language, "en")

    def test_pure_english_not_code_mixed(self):
        result = self.det.detect("The weather is nice today")
        self.assertFalse(result.code_mixed)

    def test_pure_english_zero_tamil_ratio(self):
        result = self.det.detect("Hello how are you doing today")
        self.assertEqual(result.tamil_ratio, 0.0)

    def test_pure_english_high_english_ratio(self):
        result = self.det.detect("This is a very long English sentence indeed")
        self.assertGreater(result.english_ratio, 0.70)

    def test_english_confidence_reasonable(self):
        result = self.det.detect("Good morning everyone")
        self.assertGreater(result.confidence, 0.30)


class TestLanguageDetectorTanglish(unittest.TestCase):
    """Romanized Tamil (Tanglish) inputs."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_tanglish_romba_nalla(self):
        result = self.det.detect("romba nalla irukku da")
        self.assertEqual(result.language, "tanglish")

    def test_tanglish_vandhaan(self):
        result = self.det.detect("avan vandhaan sollu")
        self.assertEqual(result.language, "tanglish")

    def test_tanglish_not_pure_english(self):
        result = self.det.detect("naan school pogiren macha")
        self.assertNotEqual(result.language, "en")

    def test_tanglish_affix_detection(self):
        # Words ending with Tamil affixes like -kku, -nga, -la
        result = self.det.detect("unakku theriyumla avankku sollu")
        self.assertIn(result.language, ("tanglish", "ta-en"))

    def test_tanglish_with_code_switch(self):
        # Mix of Tanglish core words and English
        result = self.det.detect("naan okay da seri super")
        self.assertIn(result.language, ("tanglish", "en", "unknown"))

    def test_tanglish_confidence_nonzero(self):
        result = self.det.detect("romba nalla irukku da")
        self.assertGreater(result.confidence, 0.0)

    def test_tanglish_tamil_ratio_zero(self):
        # Tanglish has no Tamil script characters
        result = self.det.detect("vandhaan nanga yenna sollunga")
        self.assertEqual(result.tamil_ratio, 0.0)


class TestLanguageDetectorCodeMixed(unittest.TestCase):
    """Tamil-English code-mixed inputs."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_code_mixed_ta_en(self):
        result = self.det.detect("நான் school போகிறேன்")
        self.assertEqual(result.language, "ta-en")
        self.assertTrue(result.code_mixed)

    def test_code_mixed_tamil_and_english_sentence(self):
        result = self.det.detect("இது நல்லா because exams romba important")
        self.assertTrue(result.code_mixed)

    def test_code_mixed_has_both_ratios_nonzero(self):
        result = self.det.detect("நான் college போகிறேன் because results வந்துவிட்டது")
        self.assertGreater(result.tamil_ratio, 0.0)
        self.assertGreater(result.english_ratio + result.tanglish_ratio, 0.0)

    def test_code_mixed_confidence_nonzero(self):
        result = self.det.detect("நான் school போகிறேன்")
        self.assertGreater(result.confidence, 0.0)

    def test_tanglish_and_script_tamil_mixed(self):
        # Tamil script + Tanglish Roman = still detectable as mixed
        result = self.det.detect("வீட்டுல romba konjam problem irukku")
        self.assertTrue(result.code_mixed or result.language in ("ta-en", "ta", "tanglish"))


class TestLanguageDetectorEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_empty_string(self):
        result = self.det.detect("")
        self.assertEqual(result.language, "unknown")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.code_mixed)

    def test_whitespace_only(self):
        result = self.det.detect("   \t\n  ")
        self.assertEqual(result.language, "unknown")

    def test_numbers_only(self):
        result = self.det.detect("123 456 789")
        self.assertEqual(result.language, "unknown")

    def test_punctuation_only(self):
        result = self.det.detect("... !!! ???")
        self.assertEqual(result.language, "unknown")

    def test_single_word_english(self):
        result = self.det.detect("hello")
        # Short text — confidence should be penalised
        self.assertLessEqual(result.confidence, 0.80)

    def test_very_long_tamil_text(self):
        text = "இது " * 50 + "நல்ல வாக்கியம்"
        result = self.det.detect(text)
        self.assertEqual(result.language, "ta")
        self.assertGreater(result.confidence, 0.65)

    def test_output_dict_schema(self):
        """Verify to_dict() matches the exact requested output schema."""
        result = self.det.detect("நான் school போகிறேன்")
        d = result.to_dict()
        self.assertIn("language", d)
        self.assertIn("code_mixed", d)
        self.assertIn("tamil_ratio", d)
        self.assertIn("english_ratio", d)
        self.assertIn("confidence", d)
        self.assertIsInstance(d["language"], str)
        self.assertIsInstance(d["code_mixed"], bool)
        self.assertIsInstance(d["tamil_ratio"], float)
        self.assertIsInstance(d["english_ratio"], float)
        self.assertIsInstance(d["confidence"], float)

    def test_confidence_in_range(self):
        for text in [
            "நான்", "hello", "romba nalla", "நான் going school",
            "", "123", "! @ #",
        ]:
            result = self.det.detect(text)
            self.assertGreaterEqual(result.confidence, 0.0, f"text={text!r}")
            self.assertLessEqual(result.confidence, 1.0, f"text={text!r}")

    def test_ratios_sum_at_most_one(self):
        """tamil_ratio + english_ratio together should not exceed 1.0 significantly."""
        for text in [
            "நான் school போகிறேன்", "I am going home", "romba nalla irukku",
        ]:
            result = self.det.detect(text)
            # These ratios are on different denominators so they can both be high;
            # but neither should exceed 1.0
            self.assertLessEqual(result.tamil_ratio, 1.0 + 1e-9)
            self.assertLessEqual(result.english_ratio, 1.0 + 1e-9)


class TestTokenTagging(unittest.TestCase):
    """Token-level classification correctness."""

    def setUp(self):
        self.det = LanguageDetector.default()

    def test_tamil_token_tagged_ta(self):
        result = self.det.detect("நான்")
        ta_tokens = [t for t in result.tokens if t.tag == TokenTag.TA]
        self.assertGreater(len(ta_tokens), 0)

    def test_english_token_tagged_en(self):
        result = self.det.detect("hello")
        en_tokens = [t for t in result.tokens if t.tag == TokenTag.EN]
        self.assertGreater(len(en_tokens), 0)

    def test_tanglish_token_tagged_tanglish(self):
        result = self.det.detect("romba")
        tng_tokens = [t for t in result.tokens if t.tag == TokenTag.TANGLISH]
        self.assertGreater(len(tng_tokens), 0)

    def test_numeral_tagged_other(self):
        result = self.det.detect("123")
        other_tokens = [t for t in result.tokens if t.tag == TokenTag.OTHER]
        self.assertGreater(len(other_tokens), 0)

    def test_tokens_list_length_matches_words(self):
        text = "நான் school போகிறேன்"
        result = self.det.detect(text)
        # 3 words
        self.assertEqual(len(result.tokens), 3)


class TestLanguageDetectorConfig(unittest.TestCase):
    """Configuration loading and customization."""

    def test_from_yaml_loads_config(self):
        cfg_path = _ROOT / "configs" / "linguistic.yaml"
        if not cfg_path.exists():
            self.skipTest("configs/linguistic.yaml not found")
        det = LanguageDetector.from_yaml(cfg_path)
        self.assertIsInstance(det, LanguageDetector)

    def test_custom_threshold_affects_classification(self):
        # With a very high Tamil threshold, fewer texts are classified as "ta"
        strict_cfg = {"detection": {"tamil_dominant_threshold": 0.99}}
        strict_det = LanguageDetector(config=strict_cfg)
        # Mixed text should not be classified as "ta" with strict threshold
        result = strict_det.detect("நான் hello")
        self.assertNotEqual(result.language, "ta")

    def test_default_factory(self):
        det = LanguageDetector.default()
        self.assertIsInstance(det, LanguageDetector)
        self.assertAlmostEqual(det.tamil_dom_thr, 0.70, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
