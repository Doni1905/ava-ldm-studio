"""
src/api/routes.py
=================
HTTP routes and handlers for the AVA LDM API using aiohttp.web.

Endpoints:
- POST /transcribe
- POST /detect-language
- POST /detect-dialect
- POST /normalize
- POST /analyze      (Primary)
- POST /evaluate
- GET  /health
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from aiohttp import web

from .service import LDMService

logger = logging.getLogger(__name__)


class LDMRoutes:
    """Configures and binds routes to an LDMService instance."""

    def __init__(self, service: Optional[LDMService] = None) -> None:
        self.service = service or LDMService()

    def setup_routes(self, app: web.Application) -> None:
        """Register API routes onto aiohttp Application."""
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/transcribe", self.handle_transcribe)
        app.router.add_post("/detect-language", self.handle_detect_language)
        app.router.add_post("/detect-dialect", self.handle_detect_dialect)
        app.router.add_post("/normalize", self.handle_normalize)
        app.router.add_post("/analyze", self.handle_analyze)
        app.router.add_post("/evaluate", self.handle_evaluate)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health - Service health check."""
        return web.json_response({
            "status": "healthy",
            "service": "ava-ldm-studio",
            "version": "1.0.0",
            "capabilities": [
                "ASR",
                "Language Detection",
                "Dialect Classification",
                "Linguistic Normalization",
                "Intent & Entity Extraction",
                "Personalization",
            ],
        })

    async def handle_transcribe(self, request: web.Request) -> web.Response:
        """POST /transcribe - Transcribe speech audio to text."""
        audio_file, cleanup_fn = await self._extract_audio_path(request)
        if not audio_file:
            return web.json_response({"error": "No audio file or valid audio_path provided"}, status=400)

        try:
            result = self.service.transcribe_audio_file(audio_file)
            return web.json_response(result)
        except Exception as e:
            logger.exception("Transcribe failed")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            cleanup_fn()

    async def handle_detect_language(self, request: web.Request) -> web.Response:
        """POST /detect-language - Detect language, script, and code-mixing."""
        data = await self._parse_json(request)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "Field 'text' is required"}, status=400)

        try:
            result = self.service.detect_language(text)
            return web.json_response(result)
        except Exception as e:
            logger.exception("Detect language failed")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_detect_dialect(self, request: web.Request) -> web.Response:
        """POST /detect-dialect - Detect regional dialect from audio or text."""
        user_id = request.query.get("user_id")

        if request.content_type.startswith("multipart/form-data"):
            audio_file, cleanup_fn = await self._extract_audio_path(request)
            if not audio_file:
                return web.json_response({"error": "Failed to read audio file from upload"}, status=400)
            try:
                result = self.service.detect_dialect(audio_path=audio_file, user_id=user_id)
                return web.json_response(result)
            finally:
                cleanup_fn()

        # Handle JSON body
        data = await self._parse_json(request)
        text = data.get("text")
        audio_path = data.get("audio_path")
        u_id = data.get("user_id", user_id)

        if not text and not audio_path:
            return web.json_response({"error": "Provide either 'text', 'audio_path', or upload an audio file"}, status=400)

        try:
            result = self.service.detect_dialect(audio_path=audio_path, text=text, user_id=u_id)
            return web.json_response(result)
        except Exception as e:
            logger.exception("Detect dialect failed")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_normalize(self, request: web.Request) -> web.Response:
        """POST /normalize - Normalize informal Tanglish/dialect text."""
        data = await self._parse_json(request)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "Field 'text' is required"}, status=400)

        dialect = data.get("dialect", "Standard")
        user_id = data.get("user_id")

        try:
            result = self.service.normalize_text(text=text, dialect=dialect, user_id=user_id)
            return web.json_response(result)
        except Exception as e:
            logger.exception("Normalize failed")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_analyze(self, request: web.Request) -> web.Response:
        """
        POST /analyze - Primary endpoint.
        Takes audio or text and returns complete structured linguistic understanding.
        """
        user_id = request.query.get("user_id")
        explicit_dialect = request.query.get("dialect")

        # Case 1: Multipart audio upload
        if request.content_type.startswith("multipart/form-data"):
            audio_file, cleanup_fn = await self._extract_audio_path(request)
            if not audio_file:
                return web.json_response({"error": "No audio file found in multipart upload"}, status=400)

            try:
                result = self.service.analyze_audio(
                    audio_path=audio_file,
                    user_id=user_id,
                    explicit_dialect=explicit_dialect,
                )
                return web.json_response(result)
            except Exception as e:
                logger.exception("Analyze audio failed")
                return web.json_response({"error": str(e)}, status=500)
            finally:
                cleanup_fn()

        # Case 2: JSON payload
        data = await self._parse_json(request)
        u_id = data.get("user_id", user_id)
        exp_dial = data.get("dialect", explicit_dialect)

        # 2a: audio_path provided in JSON
        if "audio_path" in data and data["audio_path"]:
            path = Path(data["audio_path"])
            if not path.exists():
                return web.json_response({"error": f"Audio file not found: {path}"}, status=404)
            try:
                result = self.service.analyze_audio(audio_path=path, user_id=u_id, explicit_dialect=exp_dial)
                return web.json_response(result)
            except Exception as e:
                logger.exception("Analyze audio from path failed")
                return web.json_response({"error": str(e)}, status=500)

        # 2b: direct text input provided in JSON
        if "text" in data and data["text"]:
            try:
                result = self.service.analyze_text(text=data["text"], user_id=u_id, explicit_dialect=exp_dial)
                return web.json_response(result)
            except Exception as e:
                logger.exception("Analyze text failed")
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response(
            {"error": "Invalid request. Upload an audio file or provide JSON with 'text' or 'audio_path'"},
            status=400,
        )

    async def handle_evaluate(self, request: web.Request) -> web.Response:
        """POST /evaluate - Run empirical benchmark evaluation."""
        data = await self._parse_json(request)
        out_dir = data.get("output_dir", "results")
        try:
            result = self.service.evaluate_benchmark(output_dir=out_dir)
            return web.json_response(result)
        except Exception as e:
            logger.exception("Evaluation failed")
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    async def _parse_json(request: web.Request) -> Dict[str, Any]:
        """Safely parse JSON request body, returning empty dict if missing or invalid."""
        if not request.can_read_body:
            return {}
        try:
            return await request.json()
        except Exception:
            return {}

    @staticmethod
    async def _extract_audio_path(request: web.Request) -> Tuple[Optional[Path], Any]:
        """
        Extract audio from multipart/form-data into a temporary WAV file.
        Returns (temp_file_path, cleanup_callable).
        """
        temp_file = None

        def cleanup():
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        if not request.content_type.startswith("multipart/form-data"):
            # Check if JSON audio_path was sent
            try:
                data = await request.json()
                if "audio_path" in data and os.path.exists(data["audio_path"]):
                    return Path(data["audio_path"]), lambda: None
            except Exception:
                pass
            return None, cleanup

        reader = await request.multipart()
        field = await reader.next()

        while field is not None:
            if field.name in ("audio", "file", "waveform"):
                # Save field content to temp file
                suffix = ".wav"
                if field.filename and "." in field.filename:
                    suffix = "." + field.filename.rsplit(".", 1)[-1]

                fd, temp_file = tempfile.mkstemp(suffix=suffix)
                with open(fd, "wb") as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                return Path(temp_file), cleanup

            field = await reader.next()

        return None, cleanup
