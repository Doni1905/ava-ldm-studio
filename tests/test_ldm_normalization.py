"""
tests/test_ldm_normalization.py
================================
Comprehensive tests for the Linguistic Normalization module (M7).

Test coverage
-------------
1. SlangMapper — phrase substitutions (22 rules)
2. SlangMapper — token glosses (fillers dropped, content glossed)
3. DialectAdapter — per-dialect filler removal
4. SemanticPreserver — entity extraction
5. SemanticPreserver — entity verification
6. LinguisticNormalizer — end-to-end examples derived from dataset.ts
7. Constraint tests — module does not hallucinate content
8. Schema tests — output dict matches expected keys

Run:
    python -m pytest tests/test_ldm_normalization.py -v
    python tests/test_ldm_normalization.py
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

from src.ldm.slang_mapper import SlangMapper, PhraseMatch, TokenGloss
from src.ldm.semantic_preserver import (
    SemanticPreserver, SemanticAnchor, ExtractionResult, PreservationReport,
    ET_TIME, ET_DATE, ET_PERSON, ET_PLACE, ET_APP, ET_DEVICE, ET_TASK, ET_NEG, ET_QTY,
)
from src.ldm.dialect_adapter import DialectAdapter, DIALECT_CHENNAI, DIALECT_NELLAI
from src.ldm.normalizer import LinguisticNormalizer, NormalizationResult


# ============================================================================
# Helper
# ============================================================================

def _normalizer(dialect: str = "Standard") -> LinguisticNormalizer:
    return LinguisticNormalizer.default()


# ============================================================================
# 1. SlangMapper — Phrase substitutions
# ============================================================================

class TestSlangMapperPhrases(unittest.TestCase):

    def setUp(self):
        self.sm = SlangMapper.default()

    def _apply_phrases(self, text: str) -> str:
        result, _ = self.sm.apply_phrases(text)
        return result

    def test_remind_pannu(self):
        self.assertIn("remind me", self._apply_phrases("remind pannu"))

    def test_remind_pannunga(self):
        self.assertIn("remind me", self._apply_phrases("remind pannunga"))

    def test_call_pottu_kudu(self):
        self.assertIn("call", self._apply_phrases("call pottu kudu"))

    def test_message_anuppu(self):
        self.assertIn("send message to", self._apply_phrases("message anuppu"))

    def test_alarm_vai(self):
        self.assertIn("set alarm", self._apply_phrases("alarm vai"))

    def test_alarm_vaikku(self):
        self.assertIn("set alarm", self._apply_phrases("alarm vaikku"))

    def test_vazhi_kaatu(self):
        self.assertIn("directions to", self._apply_phrases("vazhi kaatu"))

    def test_route_sollu(self):
        self.assertIn("route to", self._apply_phrases("route sollu"))

    def test_open_pannu(self):
        self.assertIn("open", self._apply_phrases("open pannu"))

    def test_off_pannu(self):
        self.assertIn("turn off", self._apply_phrases("off pannu"))

    def test_on_pannu(self):
        self.assertIn("turn on", self._apply_phrases("on pannu"))

    def test_kammi_pannu(self):
        self.assertIn("reduce", self._apply_phrases("volume kammi pannu"))

    def test_koothu(self):
        self.assertIn("increase", self._apply_phrases("brightness koothu"))

    def test_search_pannu(self):
        self.assertIn("search", self._apply_phrases("search pannu"))

    def test_eppadi_iruku(self):
        self.assertIn("how is", self._apply_phrases("eppadi iruku"))

    def test_kaalaila(self):
        self.assertIn("in the morning", self._apply_phrases("kaalaila"))

    def test_submit_panna(self):
        self.assertIn("submit", self._apply_phrases("submit panna"))

    def test_gym_poga(self):
        self.assertIn("go to the gym", self._apply_phrases("gym poga"))

    def test_mazhai_varuma(self):
        self.assertIn("will it rain", self._apply_phrases("mazhai varuma"))

    def test_late_ah_varen(self):
        self.assertIn("I will come late", self._apply_phrases("late ah varen"))

    def test_phrase_records_returned(self):
        _, matches = self.sm.apply_phrases("remind pannu nalaiku")
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0)

    def test_phrase_match_type(self):
        _, matches = self.sm.apply_phrases("remind pannu")
        self.assertIsInstance(matches[0], PhraseMatch)


# ============================================================================
# 2. SlangMapper — Token glosses
# ============================================================================

class TestSlangMapperTokens(unittest.TestCase):

    def setUp(self):
        self.sm = SlangMapper.default()

    def _apply_tokens(self, text: str) -> str:
        result, _ = self.sm.apply_token_glosses(text)
        return result

    def test_filler_dei_dropped(self):
        self.assertNotIn("dei", self._apply_tokens("dei hello"))
        self.assertIn("hello", self._apply_tokens("dei hello"))

    def test_filler_machi_dropped(self):
        out = self._apply_tokens("machi come here")
        self.assertNotIn("machi", out)
        self.assertIn("come", out)

    def test_filler_da_dropped(self):
        out = self._apply_tokens("go da")
        self.assertNotIn("da", out)

    def test_nalaiku_glossed(self):
        out = self._apply_tokens("nalaiku remind me")
        self.assertIn("tomorrow", out)
        self.assertNotIn("nalaiku", out)

    def test_inniku_glossed(self):
        out = self._apply_tokens("inniku gym")
        self.assertIn("today", out)

    def test_amma_glossed(self):
        out = self._apply_tokens("amma call")
        self.assertIn("mother", out)

    def test_konjam_glossed(self):
        out = self._apply_tokens("konjam music")
        self.assertIn("a little", out)

    def test_romba_glossed(self):
        out = self._apply_tokens("romba nalla")
        self.assertIn("very", out)

    def test_token_gloss_records(self):
        _, glosses = self.sm.apply_token_glosses("dei nalaiku")
        self.assertGreater(len(glosses), 0)
        self.assertIsInstance(glosses[0], TokenGloss)

    def test_unknown_token_passthrough(self):
        out = self._apply_tokens("xyzunknownword")
        self.assertIn("xyzunknownword", out)

    def test_is_filler(self):
        self.assertTrue(self.sm.is_filler("dei"))
        self.assertTrue(self.sm.is_filler("machi"))
        self.assertFalse(self.sm.is_filler("hello"))

    def test_gloss_lookup(self):
        self.assertEqual(self.sm.gloss("nalaiku"), "tomorrow")
        self.assertEqual(self.sm.gloss("dei"), "")
        self.assertIsNone(self.sm.gloss("unknown_word"))


# ============================================================================
# 3. DialectAdapter
# ============================================================================

class TestDialectAdapter(unittest.TestCase):

    def setUp(self):
        self.da = DialectAdapter.default()

    def test_chennai_dei_removed(self):
        out, _ = self.da.adapt("dei come here", "Chennai")
        self.assertNotIn("dei", out.lower())

    def test_chennai_vaada_removed(self):
        out, _ = self.da.adapt("vaada let us go", "Chennai")
        self.assertNotIn("vaada", out.lower())

    def test_chennai_scene_removed(self):
        out, _ = self.da.adapt("scene iruku", "Chennai")
        self.assertNotIn("scene", out.lower())

    def test_nellai_pa_removed_sentence_final(self):
        out, _ = self.da.adapt("come here pa.", "Nellai")
        self.assertNotIn(" pa", out.lower())

    def test_nellai_ille_replaced(self):
        out, _ = self.da.adapt("ille pa", "Nellai")
        self.assertIn("no", out.lower())

    def test_madurai_aama_replaced(self):
        out, _ = self.da.adapt("aama naan varuveen", "Madurai")
        self.assertIn("yes", out.lower())

    def test_kongu_ayya_removed(self):
        out, _ = self.da.adapt("ayya neenga sollunga", "Kongu")
        self.assertNotIn("ayya", out.lower())

    def test_standard_no_changes(self):
        text = "I am going to college"
        out, changes = self.da.adapt(text, "Standard")
        self.assertEqual(len(changes), 0)

    def test_unknown_dialect_no_crash(self):
        out, _ = self.da.adapt("hello world", "SomeUnknownDialect")
        self.assertEqual(out, "hello world")

    def test_adaptation_records_returned(self):
        _, changes = self.da.adapt("dei vaada come", "Chennai")
        self.assertGreater(len(changes), 0)

    def test_list_dialects(self):
        dialects = self.da.list_dialects()
        self.assertIn("Chennai", dialects)
        self.assertIn("Nellai", dialects)


# ============================================================================
# 4. SemanticPreserver — Extraction
# ============================================================================

class TestSemanticExtraction(unittest.TestCase):

    def setUp(self):
        self.sp = SemanticPreserver.default()

    def test_time_tomorrow_extracted(self):
        result = self.sp.extract("remind me tomorrow")
        types = result.entity_types_present
        self.assertIn(ET_DATE, types)

    def test_time_today_extracted(self):
        result = self.sp.extract("go today morning")
        self.assertIn(ET_DATE, result.entity_types_present)

    def test_tanglish_time_nalaiku_extracted(self):
        result = self.sp.extract("nalaiku remind me")
        self.assertTrue(result.time_refs)

    def test_person_mother_extracted(self):
        result = self.sp.extract("call amma")
        self.assertIn(ET_PERSON, result.entity_types_present)

    def test_person_english_extracted(self):
        result = self.sp.extract("call mother")
        self.assertIn(ET_PERSON, result.entity_types_present)

    def test_app_whatsapp_extracted(self):
        result = self.sp.extract("open whatsapp")
        self.assertIn(ET_APP, result.entity_types_present)

    def test_place_chennai_extracted(self):
        result = self.sp.extract("traffic in chennai")
        self.assertIn(ET_PLACE, result.entity_types_present)

    def test_device_volume_extracted(self):
        result = self.sp.extract("reduce volume")
        self.assertIn(ET_DEVICE, result.entity_types_present)

    def test_negation_detected(self):
        result = self.sp.extract("I will not go")
        self.assertTrue(result.has_negation)
        self.assertIn(ET_NEG, result.entity_types_present)

    def test_tanglish_negation_detected(self):
        result = self.sp.extract("poga mateen")
        # "mateen" matches negation pattern after phrase mapping
        # Direct extraction: "mateen" in negation markers
        self.assertTrue(result.has_negation or ET_NEG in result.entity_types_present)

    def test_clock_time_extracted(self):
        result = self.sp.extract("set alarm for 6:30 am")
        self.assertIn(ET_TIME, result.entity_types_present)

    def test_digit_quantity_extracted(self):
        result = self.sp.extract("buy 5 books")
        self.assertIn("5", result.quantities)

    def test_task_assignment_extracted(self):
        result = self.sp.extract("submit my assignment")
        self.assertIn(ET_TASK, result.entity_types_present)

    def test_task_medicine_extracted(self):
        result = self.sp.extract("take medicine tomorrow")
        self.assertIn(ET_TASK, result.entity_types_present)

    def test_empty_extraction(self):
        result = self.sp.extract("")
        self.assertEqual(result.anchors, [])

    def test_extraction_result_type(self):
        result = self.sp.extract("call mother")
        self.assertIsInstance(result, ExtractionResult)


# ============================================================================
# 5. SemanticPreserver — Verification
# ============================================================================

class TestSemanticVerification(unittest.TestCase):

    def setUp(self):
        self.sp = SemanticPreserver.default()

    def test_preserved_entity_passes(self):
        extraction = self.sp.extract("call mother tomorrow")
        report = self.sp.verify(extraction, "Call mother tomorrow.")
        self.assertTrue(report.all_preserved)

    def test_missing_entity_fails(self):
        extraction = self.sp.extract("call mother")
        # Deliberately drop 'mother' from output
        report = self.sp.verify(extraction, "make a phone call")
        self.assertFalse(report.all_preserved)

    def test_missing_app_detected(self):
        extraction = self.sp.extract("open whatsapp")
        report = self.sp.verify(extraction, "open the app")
        self.assertFalse(report.all_preserved)

    def test_negation_preserved(self):
        extraction = self.sp.extract("I will not go to school")
        report = self.sp.verify(extraction, "I will not go to school.")
        self.assertTrue(report.all_preserved)

    def test_negation_missing_warning(self):
        extraction = self.sp.extract("do not call")
        report = self.sp.verify(extraction, "call the person")
        self.assertFalse(report.all_preserved)
        self.assertGreater(len(report.warnings), 0)

    def test_report_type(self):
        extraction = self.sp.extract("hello")
        report = self.sp.verify(extraction, "hello")
        self.assertIsInstance(report, PreservationReport)


# ============================================================================
# 6. LinguisticNormalizer — End-to-end with dataset.ts examples
# ============================================================================

class TestNormalizerEndToEnd(unittest.TestCase):
    """
    Ground-truth examples derived from src/lib/ldm/dataset.ts.
    We check semantic intent is preserved, not exact string match,
    since the Python normalizer and TypeScript processor may phrase
    things slightly differently.
    """

    def setUp(self):
        self.norm = LinguisticNormalizer.default()

    def _normalize(self, text: str, dialect: str = "Chennai") -> str:
        return self.norm.normalize(text, dialect=dialect).normalized_text

    def test_example_1_reminder(self):
        out = self._normalize("Dei nalaiku assignment submit panna remind pannu")
        self.assertIn("tomorrow", out.lower())
        self.assertIn("assignment", out.lower())
        self.assertIn("submit", out.lower())
        self.assertIn("remind", out.lower())

    def test_example_2_gym_reminder(self):
        out = self._normalize("Machi inniku evening gym poga remind pannu")
        self.assertIn("today", out.lower())
        self.assertIn("gym", out.lower())
        self.assertIn("remind", out.lower())

    def test_example_4_call_amma(self):
        out = self._normalize("Amma ku call pannu", dialect="Standard")
        self.assertIn("call", out.lower())
        self.assertIn("mother", out.lower())

    def test_example_5_message_thambi(self):
        out = self._normalize("Thambi ku oru message anuppu naan late ah varen nu", dialect="Madurai")
        self.assertIn("message", out.lower())
        self.assertIn("late", out.lower())

    def test_example_6_play_song(self):
        out = self._normalize("Semma song ondru podu", dialect="Chennai")
        self.assertIn("play", out.lower())
        self.assertIn("song", out.lower())

    def test_example_7_weather(self):
        out = self._normalize("Inniku weather eppadi iruku", dialect="Standard")
        self.assertIn("today", out.lower())
        self.assertIn("weather", out.lower())

    def test_example_8_whatsapp(self):
        out = self._normalize("WhatsApp open pannu da", dialect="Chennai")
        self.assertIn("open", out.lower())
        self.assertIn("whatsapp", out.lower())

    def test_example_9_coimbatore_navigate(self):
        out = self._normalize("Coimbatore ku vazhi kaatu", dialect="Kongu")
        self.assertIn("directions", out.lower())
        self.assertIn("coimbatore", out.lower())

    def test_example_10_volume_reduce(self):
        out = self._normalize("Volume konjam kammi pannu", dialect="Standard")
        self.assertIn("volume", out.lower())
        self.assertIn("reduce", out.lower())

    def test_example_14_call_appa(self):
        out = self._normalize("Appa ku phone pottu kudu", dialect="Nellai")
        self.assertIn("call", out.lower())
        self.assertIn("father", out.lower())

    def test_example_16_weather_question(self):
        out = self._normalize("Naalaikku mazhai varuma", dialect="Standard")
        self.assertIn("tomorrow", out.lower())
        self.assertIn("rain", out.lower())

    def test_example_19_brightness(self):
        out = self._normalize("Brightness konjam koothu", dialect="Standard")
        self.assertIn("brightness", out.lower())
        self.assertIn("increase", out.lower())

    def test_example_22_search(self):
        out = self._normalize("Bangalore population evlo nu search pannu", dialect="Standard")
        self.assertIn("bangalore", out.lower())
        self.assertIn("search", out.lower())

    def test_example_27_traffic_chennai(self):
        out = self._normalize("Chennai la traffic eppadi iruku nu sollu", dialect="Chennai")
        self.assertIn("chennai", out.lower())
        self.assertIn("traffic", out.lower())

    def test_example_28_wifi_off(self):
        out = self._normalize("Wifi off pannu", dialect="Standard")
        self.assertIn("wifi", out.lower())
        self.assertIn("turn off", out.lower())

    def test_english_passthrough(self):
        out = self._normalize("Remind me to pay the electricity bill tomorrow", dialect="Standard")
        self.assertIn("remind", out.lower())
        self.assertIn("electricity", out.lower())
        self.assertIn("bill", out.lower())
        self.assertIn("tomorrow", out.lower())


# ============================================================================
# 7. Constraint tests — no hallucination
# ============================================================================

class TestNormalizerConstraints(unittest.TestCase):
    """Verify that the normalizer does NOT add content absent from input."""

    def setUp(self):
        self.norm = LinguisticNormalizer.default()

    def test_no_new_person_added(self):
        out = self.norm.normalize("play music").normalized_text
        # No person should appear out of nowhere
        for person_word in ["mother", "father", "brother", "sister", "friend"]:
            self.assertNotIn(person_word, out.lower())

    def test_no_new_place_added(self):
        out = self.norm.normalize("set alarm 7 am").normalized_text
        for place in ["chennai", "madurai", "bangalore"]:
            self.assertNotIn(place, out.lower())

    def test_no_new_time_added(self):
        out = self.norm.normalize("call whatsapp").normalized_text
        # No time should be fabricated
        self.assertNotIn("tomorrow", out.lower())
        self.assertNotIn("tonight", out.lower())

    def test_negation_not_added(self):
        out = self.norm.normalize("call mother").normalized_text
        # No negation in input, so output should not contain negation
        for neg in ["not", "don't", "won't", "never"]:
            self.assertNotIn(neg, out.lower())

    def test_empty_input_empty_output(self):
        result = self.norm.normalize("")
        self.assertEqual(result.normalized_text, "")

    def test_whitespace_input_empty_output(self):
        result = self.norm.normalize("   \t  ")
        self.assertEqual(result.normalized_text, "")

    def test_english_unchanged_entity_values(self):
        # English-only input: no Tanglish glossing should alter key words
        result = self.norm.normalize("Remind me to submit my assignment tomorrow.")
        out = result.normalized_text.lower()
        self.assertIn("submit", out)
        self.assertIn("assignment", out)
        self.assertIn("tomorrow", out)


# ============================================================================
# 8. Schema tests — output dict matches expected keys
# ============================================================================

class TestNormalizerOutputSchema(unittest.TestCase):

    def setUp(self):
        self.norm = LinguisticNormalizer.default()

    def test_to_dict_keys(self):
        result = self.norm.normalize("call mother")
        d = result.to_dict()
        for key in ("input", "normalized", "dialect", "language",
                    "all_entities_preserved", "warnings"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_normalized_is_string(self):
        result = self.norm.normalize("play music")
        self.assertIsInstance(result.normalized_text, str)

    def test_dialect_is_string(self):
        result = self.norm.normalize("play music", dialect="Chennai")
        self.assertEqual(result.dialect, "Chennai")

    def test_preservation_is_bool(self):
        result = self.norm.normalize("call mother tomorrow")
        self.assertIsInstance(result.preservation.all_preserved, bool)

    def test_warnings_is_list(self):
        result = self.norm.normalize("call mother")
        self.assertIsInstance(result.preservation.warnings, list)

    def test_trace_available(self):
        result = self.norm.normalize("remind pannu nalaiku")
        self.assertIsNotNone(result.trace)
        self.assertEqual(result.trace.input_text, "remind pannu nalaiku")

    def test_output_capitalised(self):
        result = self.norm.normalize("call mother tomorrow")
        if result.normalized_text:
            self.assertTrue(result.normalized_text[0].isupper())

    def test_output_terminated(self):
        result = self.norm.normalize("call mother")
        if result.normalized_text:
            self.assertIn(result.normalized_text[-1], ".!?")

    def test_batch_returns_list(self):
        results = self.norm.normalize_batch(["call mother", "play music"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, NormalizationResult)

    def test_from_yaml_loads(self):
        cfg_path = _ROOT / "configs" / "ldm.yaml"
        if not cfg_path.exists():
            self.skipTest("configs/ldm.yaml not found")
        norm = LinguisticNormalizer.from_yaml(cfg_path)
        result = norm.normalize("call amma")
        self.assertIsInstance(result.normalized_text, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
