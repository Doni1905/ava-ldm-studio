"""
tests/test_intent_extraction.py
===============================
Unit tests for the Intent & Entity Extraction module.
"""

import unittest
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.intent.schema import IntentExtractionResult
from src.intent.classifier import IntentClassifier, UNKNOWN_INTENT
from src.intent.entity_extractor import EntityExtractor
from src.intent.validator import IntentValidator


class TestIntentSchema(unittest.TestCase):
    def test_to_dict_and_json(self):
        result = IntentExtractionResult(
            intent="CREATE_REMINDER",
            entities={"task": "submit assignment", "date": "tomorrow"},
            confidence=0.95
        )
        
        d = result.to_dict()
        self.assertEqual(d["intent"], "CREATE_REMINDER")
        self.assertEqual(d["entities"]["task"], "submit assignment")
        self.assertEqual(d["confidence"], 0.95)
        
        j = result.to_json()
        self.assertIn("CREATE_REMINDER", j)
        self.assertIn("submit assignment", j)

    def test_from_dict_valid(self):
        data = {
            "intent": "MAKE_CALL",
            "entities": {"person": "mother"},
            "confidence": 0.95
        }
        result = IntentExtractionResult.from_dict(data)
        self.assertEqual(result.intent, "MAKE_CALL")
        self.assertEqual(result.entities["person"], "mother")

    def test_from_dict_invalid(self):
        with self.assertRaises(ValueError):
            IntentExtractionResult.from_dict({"intent": "MAKE_CALL"}) # missing entities
            

class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier.default()
        
    def test_create_reminder(self):
        intent, conf, ambig, _ = self.classifier.classify("remind me tomorrow")
        self.assertEqual(intent, "CREATE_REMINDER")
        
    def test_make_call(self):
        intent, conf, ambig, _ = self.classifier.classify("call mother")
        self.assertEqual(intent, "MAKE_CALL")
        
    def test_send_message(self):
        intent, conf, ambig, _ = self.classifier.classify("send message to brother")
        self.assertEqual(intent, "SEND_MESSAGE")
        
    def test_unknown_intent(self):
        intent, conf, ambig, _ = self.classifier.classify("just random words")
        self.assertEqual(intent, UNKNOWN_INTENT)
        self.assertEqual(conf, 0.0)


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor.default()
        
    def test_extract_date(self):
        entities = self.extractor.extract("remind me tomorrow")
        self.assertEqual(entities.get("date"), "tomorrow")
        
    def test_extract_person(self):
        entities = self.extractor.extract("call mother")
        self.assertEqual(entities.get("person"), "mother")
        
    def test_extract_app(self):
        entities = self.extractor.extract("open whatsapp")
        self.assertEqual(entities.get("app"), "whatsapp")
        
    def test_extract_task(self):
        entities = self.extractor.extract("submit my assignment tomorrow")
        self.assertEqual(entities.get("task"), "assignment")


class TestIntentValidator(unittest.TestCase):
    def setUp(self):
        self.validator = IntentValidator.default()
        
    def test_end_to_end_reminder(self):
        result = self.validator.process("remind me to submit my assignment tomorrow")
        self.assertEqual(result.intent, "CREATE_REMINDER")
        self.assertEqual(result.entities.get("date"), "tomorrow")
        self.assertEqual(result.entities.get("task"), "assignment")
        self.assertGreater(result.confidence, 0.9)
        
    def test_end_to_end_call(self):
        result = self.validator.process("call mother")
        self.assertEqual(result.intent, "MAKE_CALL")
        self.assertEqual(result.entities.get("person"), "mother")
        
    def test_entity_filtering(self):
        # MAKE_CALL does not allow 'app' in its optional entities list in the schema
        # (Assuming the schema matches configs/intents.yaml)
        result = self.validator.process("call mother on whatsapp")
        self.assertEqual(result.intent, "MAKE_CALL")
        self.assertEqual(result.entities.get("person"), "mother")
        self.assertNotIn("app", result.entities) # Filtered out


if __name__ == "__main__":
    unittest.main()
