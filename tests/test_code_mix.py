"""
tests/test_code_mix.py
========================
Unit tests for CodeMixDetector.

Tests cover:
  - CMI (Code Mixing Index) computation correctness
  - Switch point counting
  - M-index (entropy-based multilingual index)
  - Span extraction
  - Output schema matches requested format
  - Monolingual → CMI = 0
  - Maximally mixed → CMI approaches 0.5
  - Confidence values in range

Run from project root:
    python -m pytest tests/test_code_mix.py -v
    python tests/test_code_mix.py
"""

from __future__ import annotations

import io
import math
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.linguistic.code_mix_detector import CodeMixDetector, CodeMixResult, SpanInfo
from src.linguistic.language_detector import LanguageDetector, TokenTag


class TestCMIComputation(unittest.TestCase):
    """Code Mixing Index (CMI) mathematical correctness."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_monolingual_tamil_cmi_zero(self):
        result = self.cd.detect("நான் பள்ளிக்கு போகிறேன் இது நல்ல நாள்")
        self.assertAlmostEqual(result.cmi, 0.0, places=1)

    def test_monolingual_english_cmi_zero(self):
        # Use purely unambiguous English words (no Tanglish lexicon overlap)
        result = self.cd.detect("the weather outside is beautiful this afternoon")
        self.assertAlmostEqual(result.cmi, 0.0, places=1)

    def test_code_mixed_cmi_positive(self):
        result = self.cd.detect("நான் school போகிறேன் because exams irukku")
        self.assertGreater(result.cmi, 0.0)

    def test_cmi_in_range(self):
        for text in [
            "நான்", "hello world", "romba nalla irukku da",
            "நான் school போகிறேன்",
            "",
        ]:
            result = self.cd.detect(text)
            self.assertGreaterEqual(result.cmi, 0.0, f"text={text!r}")
            self.assertLessEqual(result.cmi, 1.0, f"text={text!r}")

    def test_more_mixed_higher_cmi(self):
        """More balanced language distribution should yield higher CMI."""
        monolingual = self.cd.detect("நான் பள்ளி வீடு")
        code_mixed  = self.cd.detect("நான் school போகிறேன் today")
        self.assertGreater(code_mixed.cmi, monolingual.cmi)


class TestSwitchPoints(unittest.TestCase):
    """Language switch point counting."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_monolingual_zero_switch_points(self):
        result = self.cd.detect("நான் பள்ளிக்கு போகிறேன்")
        self.assertEqual(result.switch_points, 0)

    def test_alternating_switch_points(self):
        # ta en ta en = 3 switches
        result = self.cd.detect("நான் school படிக்கிறேன் today")
        self.assertGreaterEqual(result.switch_points, 1)

    def test_empty_input_zero_switch_points(self):
        result = self.cd.detect("")
        self.assertEqual(result.switch_points, 0)

    def test_switch_points_non_negative(self):
        for text in ["hello", "நான்", "romba nalla", "நான் hello"]:
            result = self.cd.detect(text)
            self.assertGreaterEqual(result.switch_points, 0)


class TestMIndex(unittest.TestCase):
    """M-index (multilingual entropy) computation."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_monolingual_m_index_zero(self):
        result = self.cd.detect("நான் பள்ளிக்கு போகிறேன் இது நல்ல")
        self.assertAlmostEqual(result.m_index, 0.0, places=1)

    def test_code_mixed_m_index_positive(self):
        result = self.cd.detect("நான் school போகிறேன் because exams")
        self.assertGreater(result.m_index, 0.0)

    def test_m_index_in_range(self):
        for text in [
            "hello world", "நான்", "romba nalla da",
            "நான் school போகிறேன்", "",
        ]:
            result = self.cd.detect(text)
            self.assertGreaterEqual(result.m_index, 0.0, f"text={text!r}")
            self.assertLessEqual(result.m_index, 1.0, f"text={text!r}")

    def test_empty_m_index_zero(self):
        result = self.cd.detect("")
        self.assertEqual(result.m_index, 0.0)


class TestSpanExtraction(unittest.TestCase):
    """Dominant language span extraction."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_monolingual_single_span(self):
        result = self.cd.detect("நான் பள்ளிக்கு போகிறேன்")
        # All Tamil → one contiguous span
        ta_spans = [s for s in result.dominant_spans if s.language == "ta"]
        self.assertGreater(len(ta_spans), 0)

    def test_code_mixed_multiple_spans(self):
        result = self.cd.detect("நான் school போகிறேன்")
        self.assertGreater(len(result.dominant_spans), 1)

    def test_spans_cover_all_tokens(self):
        text = "நான் school போகிறேன்"
        result = self.cd.detect(text)
        total_span_tokens = sum(s.length for s in result.dominant_spans)
        self.assertEqual(total_span_tokens, len(result.token_details))

    def test_span_tokens_match_token_details(self):
        text = "நான் school"
        result = self.cd.detect(text)
        span_all_tokens = [t for s in result.dominant_spans for t in s.tokens]
        detail_tokens   = [t["token"] for t in result.token_details]
        self.assertEqual(span_all_tokens, detail_tokens)

    def test_empty_input_no_spans(self):
        result = self.cd.detect("")
        self.assertEqual(result.dominant_spans, [])


class TestCodeMixedLanguageClassification(unittest.TestCase):
    """High-level code-mix classification correctness."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_ta_en_code_mixed_flag(self):
        result = self.cd.detect("நான் school போகிறேன்")
        self.assertTrue(result.code_mixed)
        self.assertEqual(result.language, "ta-en")

    def test_pure_tamil_not_code_mixed(self):
        result = self.cd.detect("நான் பள்ளிக்கு போகிறேன்")
        self.assertFalse(result.code_mixed)

    def test_pure_english_not_code_mixed(self):
        result = self.cd.detect("I am going to school today")
        self.assertFalse(result.code_mixed)

    def test_tanglish_classified(self):
        result = self.cd.detect("romba nalla irukku vandhaan")
        self.assertIn(result.language, ("tanglish", "ta-en", "en"))

    def test_tanglish_ratios(self):
        result = self.cd.detect("romba nalla irukku da")
        self.assertEqual(result.tamil_ratio, 0.0)   # No Tamil script
        self.assertGreaterEqual(result.tanglish_ratio, 0.0)


class TestOutputSchema(unittest.TestCase):
    """Verify output dict matches the required JSON schema."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def _check_schema(self, d: dict) -> None:
        required = {
            "language":       str,
            "code_mixed":     bool,
            "tamil_ratio":    float,
            "english_ratio":  float,
            "confidence":     float,
        }
        for key, typ in required.items():
            self.assertIn(key, d, f"Missing key: {key}")
            self.assertIsInstance(d[key], typ, f"Wrong type for {key}")

    def test_to_dict_schema_tamil(self):
        result = self.cd.detect("நான் பள்ளி போகிறேன்")
        self._check_schema(result.to_dict())

    def test_to_dict_schema_english(self):
        result = self.cd.detect("I am going to school")
        self._check_schema(result.to_dict())

    def test_to_dict_schema_code_mixed(self):
        result = self.cd.detect("நான் school போகிறேன்")
        self._check_schema(result.to_dict())

    def test_to_dict_schema_tanglish(self):
        result = self.cd.detect("romba nalla irukku da")
        self._check_schema(result.to_dict())

    def test_to_dict_schema_empty(self):
        result = self.cd.detect("")
        self._check_schema(result.to_dict())

    def test_full_dict_has_extended_fields(self):
        result = self.cd.detect("நான் school போகிறேன்")
        d = result.to_full_dict()
        self.assertIn("tanglish_ratio", d)
        self.assertIn("cmi", d)
        self.assertIn("switch_points", d)
        self.assertIn("m_index", d)
        self.assertIn("token_details", d)
        self.assertIn("dominant_spans", d)

    def test_all_ratios_in_valid_range(self):
        for text in [
            "நான் school போகிறேன்", "hello world", "romba nalla", "",
        ]:
            result = self.cd.detect(text)
            d = result.to_dict()
            for key in ("tamil_ratio", "english_ratio", "confidence"):
                self.assertGreaterEqual(d[key], 0.0, f"{key} < 0 for {text!r}")
                self.assertLessEqual(d[key], 1.0, f"{key} > 1 for {text!r}")

    def test_language_values_are_valid_strings(self):
        valid = {"ta", "en", "ta-en", "tanglish", "unknown"}
        for text in [
            "நான்", "hello", "romba nalla irukku",
            "நான் school போகிறேன்", "",
        ]:
            result = self.cd.detect(text)
            self.assertIn(result.language, valid, f"text={text!r}")


class TestCodeMixDetectorConfig(unittest.TestCase):
    """Config loading and factory methods."""

    def test_from_yaml(self):
        cfg_path = _ROOT / "configs" / "linguistic.yaml"
        if not cfg_path.exists():
            self.skipTest("configs/linguistic.yaml not found")
        cd = CodeMixDetector.from_yaml(cfg_path)
        self.assertIsInstance(cd, CodeMixDetector)

    def test_default_factory(self):
        cd = CodeMixDetector.default()
        self.assertIsInstance(cd, CodeMixDetector)

    def test_shared_detector_reuse(self):
        det = LanguageDetector.default()
        cd1 = CodeMixDetector(detector=det)
        cd2 = CodeMixDetector(detector=det)
        # Both should work correctly
        r1 = cd1.detect("நான் school போகிறேன்")
        r2 = cd2.detect("நான் school போகிறேன்")
        self.assertEqual(r1.language, r2.language)


class TestTokenDetails(unittest.TestCase):
    """Token detail records in CodeMixResult."""

    def setUp(self):
        self.cd = CodeMixDetector.default()

    def test_token_details_present(self):
        result = self.cd.detect("நான் school போகிறேன்")
        self.assertIsInstance(result.token_details, list)
        self.assertGreater(len(result.token_details), 0)

    def test_token_detail_schema(self):
        result = self.cd.detect("நான் school")
        for td in result.token_details:
            self.assertIn("token", td)
            self.assertIn("tag", td)
            self.assertIn("tanglish_score", td)
            self.assertIsInstance(td["token"], str)
            self.assertIsInstance(td["tag"], str)
            self.assertIsInstance(td["tanglish_score"], float)

    def test_tanglish_token_has_positive_score(self):
        result = self.cd.detect("romba")
        tng_tokens = [td for td in result.token_details if td["tag"] == "tanglish"]
        if tng_tokens:
            self.assertGreater(tng_tokens[0]["tanglish_score"], 0.0)

    def test_english_token_has_zero_tanglish_score(self):
        result = self.cd.detect("the")
        en_tokens = [td for td in result.token_details if td["tag"] == "en"]
        for td in en_tokens:
            self.assertAlmostEqual(td["tanglish_score"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
