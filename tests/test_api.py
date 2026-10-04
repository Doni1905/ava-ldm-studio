"""
tests/test_api.py
=================
Integration and unit tests for the AVA LDM Studio REST API.

Tests:
- GET  /health
- POST /detect-language
- POST /detect-dialect
- POST /normalize
- POST /analyze (Primary endpoint: text and multipart audio)
- POST /transcribe
- Error handling and input validation
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aiohttp import FormData, test_utils, web

# Safe UTF-8 reconfiguration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.routes import LDMRoutes
from src.api.server import create_app
from src.api.service import LDMService


class TestLDMAPI(test_utils.AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        # Create service with mock ASR and dialect engine to keep test runs fast and deterministic
        self.service = LDMService(use_lazy_models=True)
        # Mock pipeline ASR and dialect for fast testing without loading 500MB weights
        mock_pipeline = MagicMock()
        mock_pipeline.asr_engine.transcribe_batch.return_value = [{"text": "nalaiku assignment submit panna remind pannu"}]
        mock_pipeline.dialect_clf.predict_file.return_value = {"dialect": "Chennai", "confidence": 0.95}

        # Mock process() output on pipeline
        mock_output = MagicMock()
        mock_output.transcript = "nalaiku assignment submit panna remind pannu"
        mock_output.language = "tanglish"
        mock_output.dialect = "Chennai"
        mock_output.code_mixed = True
        mock_output.normalized_text = "Remind me to submit my assignment tomorrow."
        mock_output.intent = "CREATE_REMINDER"
        mock_output.entities = {"task": "assignment", "date": "tomorrow"}
        mock_output.confidence = {"language": 0.9, "dialect": 0.95, "intent": 0.95}
        mock_output.latency_ms = {"asr": 12.5, "normalization": 0.8, "total": 15.0}
        mock_pipeline.process.return_value = mock_output

        self.service._pipeline = mock_pipeline
        return create_app(service=self.service)

    async def test_health_check(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "ava-ldm-studio")
        self.assertIn("capabilities", data)

    async def test_detect_language(self):
        payload = {"text": "nalaiku assignment submit panna remind pannu"}
        resp = await self.client.post("/detect-language", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("language", data)
        self.assertIn("code_mixed", data)
        self.assertIsInstance(data["code_mixed"], bool)

    async def test_detect_language_empty_error(self):
        resp = await self.client.post("/detect-language", json={"text": ""})
        self.assertEqual(resp.status, 400)

    async def test_detect_dialect_from_text(self):
        payload = {"text": "dei machi vaada Chennai la meet pannuvom"}
        resp = await self.client.post("/detect-dialect", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["dialect"], "Chennai")

    async def test_normalize_endpoint(self):
        payload = {"text": "dei nalaiku assignment submit panna remind pannu", "dialect": "Chennai"}
        resp = await self.client.post("/normalize", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("normalized_text", data)
        self.assertNotIn("dei", data["normalized_text"].lower())
        self.assertIn("assignment", data["normalized_text"].lower())

    async def test_analyze_text_json(self):
        # Primary endpoint tested via JSON payload
        payload = {"text": "dei nalaiku assignment submit panna remind pannu"}
        resp = await self.client.post("/analyze", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()

        # Validate all required fields in the prompt
        required_fields = [
            "transcript",
            "language",
            "dialect",
            "code_mixed",
            "normalized_text",
            "intent",
            "entities",
            "confidence",
            "latency_ms",
        ]
        for field in required_fields:
            self.assertIn(field, data, f"Field '{field}' missing from /analyze response")

        self.assertEqual(data["intent"], "CREATE_REMINDER")
        self.assertEqual(data["entities"].get("date"), "tomorrow")

    async def test_analyze_audio_multipart(self):
        # Primary endpoint tested via multipart audio upload
        # Build dummy WAV header + silence
        import numpy as np
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        dummy_audio = np.zeros(16000, dtype=np.float32)
        sf.write(temp_path, dummy_audio, 16000)

        try:
            with open(temp_path, "rb") as audio_file:
                form = FormData()
                form.add_field("audio", audio_file, filename="test_clip.wav", content_type="audio/wav")
                resp = await self.client.post("/analyze", data=form)

            self.assertEqual(resp.status, 200)
            data = await resp.json()
            self.assertEqual(data["transcript"], "nalaiku assignment submit panna remind pannu")
            self.assertEqual(data["intent"], "CREATE_REMINDER")
            self.assertEqual(data["dialect"], "Chennai")
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    async def test_analyze_invalid_empty(self):
        resp = await self.client.post("/analyze", json={})
        self.assertEqual(resp.status, 400)

    async def test_transcribe_endpoint(self):
        import numpy as np
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        dummy_audio = np.zeros(16000, dtype=np.float32)
        sf.write(temp_path, dummy_audio, 16000)

        try:
            with open(temp_path, "rb") as audio_file:
                form = FormData()
                form.add_field("audio", audio_file, filename="speech.wav", content_type="audio/wav")
                resp = await self.client.post("/transcribe", data=form)

            self.assertEqual(resp.status, 200)
            data = await resp.json()
            self.assertIn("transcript", data)
            self.assertIn("duration_seconds", data)
            self.assertIn("latency_ms", data)
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
