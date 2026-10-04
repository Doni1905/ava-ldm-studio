"""
src/api/service.py
==================
Modular service layer connecting HTTP endpoints to the AVA LDM pipeline.

Allows components (ASR, Dialect Classifier, Normalizer, Intent Validator, LLM)
to be replaced or mocked independently for Android integration and testing.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import soundfile as sf

from src.intent import IntentValidator
from src.ldm.normalizer import LinguisticNormalizer
from src.ldm.pipeline import LdmPipeline
from src.linguistic.code_mix_detector import CodeMixDetector
from src.linguistic.expression_mapper import ExpressionMapper
from src.linguistic.language_detector import LanguageDetector
from src.linguistic.slang_mapper import SlangMapper
from src.personalization import PreferenceStore

logger = logging.getLogger(__name__)


class LDMService:
    """
    Central business logic coordinator for the AVA LDM API.
    Decoupled from HTTP framework and replaceable per-component.
    """

    def __init__(
        self,
        pipeline: Optional[LdmPipeline] = None,
        preference_store: Optional[PreferenceStore] = None,
        use_lazy_models: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._use_lazy = use_lazy_models
        self.pref_store = preference_store or PreferenceStore()

        # Direct modular sub-components for lightweight single-task endpoints
        self.lang_detector = LanguageDetector.default()
        self.cm_detector = CodeMixDetector.default()
        self.normalizer = LinguisticNormalizer.default()
        self.expr_mapper = ExpressionMapper.default()
        self.slang_mapper = SlangMapper.default()
        self.intent_validator = IntentValidator.default()

    @property
    def pipeline(self) -> LdmPipeline:
        """Lazy-initialize full LdmPipeline if needed."""
        if self._pipeline is None:
            logger.info("Initializing full LdmPipeline for LDMService...")
            self._pipeline = LdmPipeline()
        return self._pipeline

    # ------------------------------------------------------------------
    # 1. Transcribe (ASR Only)
    # ------------------------------------------------------------------

    def transcribe_audio_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribe an audio file using the configured ASR engine."""
        t0 = time.perf_counter()
        audio_array, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio_mono = audio_array.mean(axis=1)
        duration_s = len(audio_mono) / sr

        asr_res = self.pipeline.asr_engine.transcribe_batch([audio_mono], [sr])[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "transcript": asr_res["text"],
            "duration_seconds": round(duration_s, 2),
            "sample_rate": sr,
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # 2. Detect Language & Code-Mix
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language, Tanglish presence, and code-mixing metrics."""
        t0 = time.perf_counter()
        lang_res = self.lang_detector.detect(text)
        cm_res = self.cm_detector.detect(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "language": lang_res.language,
            "code_mixed": cm_res.code_mixed,
            "confidence": round(lang_res.confidence, 4),
            "tamil_ratio": round(lang_res.tamil_ratio, 4),
            "english_ratio": round(lang_res.english_ratio, 4),
            "cmi": round(cm_res.cmi, 4),
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # 3. Detect Dialect
    # ------------------------------------------------------------------

    def detect_dialect(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect regional Tamil dialect from audio or lexical text markers.
        Applies personalization fallback when input speech is standard/neutral.
        """
        t0 = time.perf_counter()
        dialect = "Standard"
        confidence = 0.80

        if audio_path and Path(audio_path).exists():
            res = self.pipeline.dialect_clf.predict_file(str(audio_path))
            dialect = res.get("dialect", "Standard")
            confidence = res.get("confidence", 0.80)
        elif text:
            # Lexical marker classification
            from scripts.run_evaluation import detect_text_dialect
            dialect = detect_text_dialect(text)
            confidence = 0.90 if dialect != "Standard" else 0.75

        # Personalization resolution
        decision_source = "speech_detection"
        if user_id:
            dialect, decision_source = self.pref_store.resolve_effective_dialect(
                user_id=user_id,
                detected_dialect=dialect,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "dialect": dialect,
            "confidence": round(confidence, 4),
            "source": decision_source,
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # 4. Normalize
    # ------------------------------------------------------------------

    def normalize_text(
        self,
        text: str,
        dialect: str = "Standard",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Normalize informal Tanglish/dialect speech into standard semantic English.
        """
        t0 = time.perf_counter()

        # Step 1: User personalization overrides (custom vocab & expressions)
        working_text = text
        if user_id:
            working_text = self.pref_store.apply_personalization(user_id, working_text)

        # Step 2: Dialect adaptation & base normalization
        norm_res = self.normalizer.normalize(working_text, dialect=dialect)

        # Step 3: Expression and slang dictionary mapping
        expr_out = self.expr_mapper.map(norm_res.normalized_text).output_text
        final_norm = self.slang_mapper.apply(expr_out, drop_fillers=True)

        if final_norm:
            final_norm = final_norm[0].upper() + final_norm[1:]
            if final_norm[-1] not in ".!?":
                final_norm += "."

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "original_text": text,
            "normalized_text": final_norm,
            "dialect": dialect,
            "all_entities_preserved": norm_res.preservation.all_preserved,
            "warnings": norm_res.preservation.warnings,
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # 5. Primary Endpoint: Analyze (End-to-End)
    # ------------------------------------------------------------------

    def analyze_audio(
        self,
        audio_path: Union[str, Path],
        user_id: Optional[str] = None,
        explicit_dialect: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Primary end-to-end endpoint:
        Audio -> ASR -> Lang/CodeMix -> Dialect -> Normalization -> Intent/Entities
        """
        pipe_output = self.pipeline.process(audio_path)

        # If user personalization is provided, apply user resolution
        effective_dialect = pipe_output.dialect
        if user_id or explicit_dialect:
            effective_dialect, _ = self.pref_store.resolve_effective_dialect(
                user_id=user_id or "default_user",
                detected_dialect=pipe_output.dialect,
                explicit_dialect=explicit_dialect,
            )

        return {
            "transcript": pipe_output.transcript,
            "language": pipe_output.language,
            "dialect": effective_dialect,
            "code_mixed": pipe_output.code_mixed,
            "normalized_text": pipe_output.normalized_text,
            "intent": pipe_output.intent,
            "entities": pipe_output.entities,
            "confidence": pipe_output.confidence,
            "latency_ms": {k: round(v, 2) for k, v in pipe_output.latency_ms.items()},
        }

    def analyze_text(
        self,
        text: str,
        user_id: Optional[str] = None,
        explicit_dialect: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct text analysis for text-based playground input or Android text fallback.
        """
        t0 = time.perf_counter()
        latencies = {}

        # 1. Language & Code-Mix
        t_start = time.perf_counter()
        lang_res = self.lang_detector.detect(text)
        cm_res = self.cm_detector.detect(text)
        latencies["linguistic_detection"] = (time.perf_counter() - t_start) * 1000

        # 2. Dialect
        t_start = time.perf_counter()
        from scripts.run_evaluation import detect_text_dialect
        detected_dialect = detect_text_dialect(text)
        effective_dialect, _ = self.pref_store.resolve_effective_dialect(
            user_id=user_id or "default_user",
            detected_dialect=detected_dialect,
            explicit_dialect=explicit_dialect,
        )
        latencies["dialect_classification"] = (time.perf_counter() - t_start) * 1000

        # 3. Normalization
        t_start = time.perf_counter()
        norm_dict = self.normalize_text(text, dialect=effective_dialect, user_id=user_id)
        norm_text = norm_dict["normalized_text"]
        latencies["normalization"] = (time.perf_counter() - t_start) * 1000

        # 4. Intent & Entity Extraction
        t_start = time.perf_counter()
        intent_res = self.intent_validator.process(norm_text)
        latencies["intent_extraction"] = (time.perf_counter() - t_start) * 1000

        latencies["total"] = (time.perf_counter() - t0) * 1000

        return {
            "transcript": text,
            "language": lang_res.language,
            "dialect": effective_dialect,
            "code_mixed": cm_res.code_mixed,
            "normalized_text": norm_text,
            "intent": intent_res.intent,
            "entities": intent_res.entities,
            "confidence": {
                "language": round(lang_res.confidence, 4),
                "dialect": 0.90 if detected_dialect != "Standard" else 0.80,
                "intent": round(intent_res.confidence, 4),
            },
            "latency_ms": {k: round(v, 2) for k, v in latencies.items()},
        }

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------

    def evaluate_benchmark(self, output_dir: str = "results") -> Dict[str, Any]:
        """Run the standard evaluation benchmark and return summary metrics."""
        from scripts.run_evaluation import run_benchmark
        return run_benchmark(output_dir=output_dir)
