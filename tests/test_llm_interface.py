"""
tests/test_llm_interface.py
===========================
Unit tests for the AVA Local LLM interface.

Verifies:
1. LDM Handoff data contract and conversion
2. PromptBuilder formatting & dialect filtering
3. Local engine generation, latency, and memory tracking
4. Streaming response generation
5. Configurable model loader (YAML & dictionary overrides)
6. Adapter architecture connecting LDM output to LLM
"""

import io
import sys
import unittest
from pathlib import Path

# Force UTF-8 stdout safely
if hasattr(sys.stdout, "buffer") and not getattr(sys.stdout, "closed", False):
    try:
        if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.llm.base import BaseLLM, LDMHandoff, LLMResponse
from src.llm.local_engine import MemoryTracker, MockLLMEngine
from src.llm.model_loader import ModelLoader, load_llm
from src.llm.prompt_builder import PromptBuilder


class TestLDMHandoff(unittest.TestCase):
    def test_construction_and_dict(self):
        handoff = LDMHandoff(
            normalized_text="Remind me to submit my assignment tomorrow.",
            intent="CREATE_REMINDER",
            entities={"task": "assignment", "date": "tomorrow"},
            language="tanglish",
            dialect="Chennai",
            confidence={"intent": 0.95},
        )
        d = handoff.to_dict()
        self.assertEqual(d["normalized_text"], "Remind me to submit my assignment tomorrow.")
        self.assertEqual(d["intent"], "CREATE_REMINDER")
        self.assertEqual(d["entities"]["task"], "assignment")

    def test_from_pipeline_dict(self):
        # Simulates output dict from LdmPipeline.process()
        pipeline_output = {
            "transcript": "dei nalaiku assignment submit panna remind pannu da",
            "language": "tanglish",
            "code_mixed": True,
            "dialect": "Chennai",
            "normalized_text": "Remind me to submit my assignment tomorrow.",
            "intent": "CREATE_REMINDER",
            "entities": {"task": "assignment", "date": "tomorrow"},
            "confidence": {"intent": 0.95, "dialect": 0.88},
            "latency_ms": {"total": 45.2},
        }

        handoff = LDMHandoff.from_dict(pipeline_output)
        self.assertEqual(handoff.normalized_text, "Remind me to submit my assignment tomorrow.")
        self.assertEqual(handoff.intent, "CREATE_REMINDER")
        self.assertEqual(handoff.entities["date"], "tomorrow")
        self.assertEqual(handoff.raw_transcript, "dei nalaiku assignment submit panna remind pannu da")

    def test_json_serialization(self):
        handoff = LDMHandoff(normalized_text="Hello world", intent="GENERAL_QUERY")
        j = handoff.to_json()
        self.assertIn("Hello world", j)
        self.assertIn("GENERAL_QUERY", j)


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(include_metadata=True)

    def test_normalized_input_used_not_raw_dialect(self):
        # Ensure only normalized semantic text and context metadata are exposed
        handoff = LDMHandoff(
            normalized_text="Remind me to submit my assignment tomorrow.",
            intent="CREATE_REMINDER",
            entities={"task": "assignment", "date": "tomorrow"},
            raw_transcript="dei nalaiku assignment submit panna remind pannu da",
        )

        prompt_text = self.builder.build_text(handoff)
        # Raw dialect words must NOT be part of the prompt
        self.assertNotIn("dei", prompt_text.lower())
        self.assertNotIn("pannu", prompt_text.lower())
        # Normalized text must be present
        self.assertIn("Remind me to submit my assignment tomorrow.", prompt_text)
        self.assertIn("CREATE_REMINDER", prompt_text)
        self.assertIn("task='assignment'", prompt_text)

    def test_chat_messages_structure(self):
        handoff = LDMHandoff(normalized_text="Call mother.", intent="MAKE_CALL", entities={"person": "mother"})
        messages = self.builder.build_messages(handoff)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Call mother.", messages[1]["content"])
        self.assertIn("Intent: MAKE_CALL", messages[1]["content"])

    def test_instruct_format(self):
        builder = PromptBuilder(format_type="instruct", include_metadata=False)
        text = builder.build_text(LDMHandoff(normalized_text="Turn on wifi."))
        self.assertIn("### Instruction:\nTurn on wifi.", text)
        self.assertIn("### Response:", text)


class TestLocalEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MockLLMEngine(model_name="test-mock", context_length=1024)

    def test_engine_properties(self):
        self.assertEqual(self.engine.model_name, "test-mock")
        self.assertEqual(self.engine.backend, "mock")
        self.assertEqual(self.engine.device, "cpu")
        self.assertEqual(self.engine.context_length, 1024)

    def test_generate_from_handoff(self):
        handoff = LDMHandoff(
            normalized_text="Remind me to submit my assignment tomorrow.",
            intent="CREATE_REMINDER",
            entities={"task": "assignment", "date": "tomorrow"},
        )
        response = self.engine.generate(handoff)

        self.assertIsInstance(response, LLMResponse)
        self.assertIn("reminder", response.text.lower())
        self.assertIn("assignment", response.text.lower())
        self.assertGreater(response.tokens_generated, 0)
        self.assertGreater(response.latency_ms, 0.0)
        self.assertGreaterEqual(response.memory_peak_mb, 0.0)
        self.assertEqual(response.backend, "mock")

    def test_generate_from_raw_string(self):
        response = self.engine.generate("What is the capital of Tamil Nadu?")
        self.assertIn("What is the capital of Tamil Nadu?", response.text)
        self.assertGreater(response.tokens_generated, 0)

    def test_streaming_generation(self):
        handoff = LDMHandoff(
            normalized_text="Call mother.",
            intent="MAKE_CALL",
            entities={"person": "mother"},
        )
        stream = list(self.engine.stream_generate(handoff))
        self.assertGreater(len(stream), 1)
        reconstructed = "".join(stream)
        self.assertIn("mother", reconstructed)
        self.assertIn("call", reconstructed.lower())

    def test_memory_tracker(self):
        with MemoryTracker("cpu") as tracker:
            # Allocate memory
            data = [i for i in range(100000)]
        peak_mb = tracker.get_peak_mb()
        self.assertGreater(peak_mb, 0.0)


class TestModelLoader(unittest.TestCase):
    def test_load_from_yaml(self):
        config_path = _ROOT / "configs" / "llm.yaml"
        llm = ModelLoader.load(config_path)
        self.assertIsInstance(llm, BaseLLM)
        self.assertEqual(llm.backend, "mock")

    def test_load_with_dict_override(self):
        override_cfg = {
            "model": {
                "name": "custom-mock-v2",
                "backend": "mock",
                "device": "cpu",
                "max_context_length": 4096,
            },
            "generation": {"max_new_tokens": 64},
            "prompt": {"format": "instruct"},
        }
        llm = load_llm(override_cfg)
        self.assertEqual(llm.model_name, "custom-mock-v2")
        self.assertEqual(llm.context_length, 4096)

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            ModelLoader.load({"model": {"backend": "unsupported_backend_xyz"}})


class TestLDMToLLMIntegration(unittest.TestCase):
    """
    Verifies the end-to-end adapter flow:
    LDM Output -> LDMHandoff -> LLM Interface -> Response
    """
    def test_adapter_flow(self):
        # 1. Mock LDM pipeline result
        ldm_result = {
            "transcript": "dei en appa ku call pannu",
            "language": "tanglish",
            "dialect": "Chennai",
            "normalized_text": "Call my father.",
            "intent": "MAKE_CALL",
            "entities": {"person": "father"},
            "confidence": {"intent": 0.95},
        }

        # 2. Convert to standardized handoff
        handoff = LDMHandoff.from_dict(ldm_result)

        # 3. Pass to LLM engine
        llm = load_llm()
        response = llm.generate(handoff)

        # 4. Verify response
        self.assertIn("father", response.text.lower())
        self.assertIn("call", response.text.lower())
        self.assertNotIn("dei", response.text.lower())  # Raw dialect not leaked
        self.assertGreater(response.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
