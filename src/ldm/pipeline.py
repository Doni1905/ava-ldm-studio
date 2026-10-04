"""
src/ldm/pipeline.py
===================
Unified LDM Master Pipeline.

Integrates:
1. Audio (not processed separately if ASR engine can handle it, but we can pass raw audio)
2. ASR (Whisper)
3. Language Detection (M5)
4. Code-Mix Detection (M5)
5. Dialect Classification (M6)
6. Linguistic Normalization (M7)
7. Slang/Informal Mapping (M8)
8. Intent Detection & Entity Extraction (M9)

Produces the final Structured Linguistic Understanding payload.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

# 1. ASR
from src.asr.whisper_engine import WhisperEngine
# 2. Language & Code-Mix
from src.linguistic.language_detector import LanguageDetector
from src.linguistic.code_mix_detector import CodeMixDetector
# 3. Dialect
from src.dialect.inference import DialectInferencePipeline
# 4. Normalization
from src.ldm.normalizer import LinguisticNormalizer
# 5. Slang/Informal Mapping
from src.linguistic.slang_mapper import SlangMapper
from src.linguistic.expression_mapper import ExpressionMapper
# 6. Intent & Entities
from src.intent.validator import IntentValidator

logger = logging.getLogger(__name__)


@dataclass
class LdmPipelineOutput:
    transcript: str
    language: str
    code_mixed: bool
    dialect: str
    normalized_text: str
    intent: str
    entities: Dict[str, str]
    confidence: Dict[str, float]
    latency_ms: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LdmPipeline:
    """Unified LDM Orchestrator"""

    def __init__(self, use_cpu: bool = True, project_root: str | Path = None):
        logger.info("Initializing LDM Master Pipeline...")
        t0 = time.perf_counter()
        
        root = Path(project_root) if project_root else Path.cwd()
        
        # Load configs
        import yaml
        from src.dialect.labels import DialectLabelRegistry
        from src.dialect.classifier import DialectClassifier
        
        with open(root / "configs" / "asr.yaml", encoding="utf-8") as f:
            asr_cfg = yaml.safe_load(f)
            
        with open(root / "configs" / "dialect.yaml", encoding="utf-8") as f:
            dialect_cfg = yaml.safe_load(f)
            
        # 1. ASR
        self.asr_engine = WhisperEngine(config=asr_cfg)
        
        # 2 & 3. Linguistic
        self.lang_detector = LanguageDetector.default()
        self.cm_detector = CodeMixDetector.default()
        
        # 4. Dialect
        registry = DialectLabelRegistry(config=dialect_cfg)
        model = DialectClassifier(config=dialect_cfg, num_classes=registry.num_classes)
        self.dialect_clf = DialectInferencePipeline(model=model, label_registry=registry, config=dialect_cfg)
        
        # 5. Normalization
        self.normalizer = LinguisticNormalizer.default()
        
        # 6. Slang mapping
        self.expr_mapper = ExpressionMapper.default(project_root=root)
        self.slang_mapper = SlangMapper.default(project_root=root)
        
        # 7. Intent
        self.intent_validator = IntentValidator.default()
        
        t1 = time.perf_counter()
        logger.info(f"Pipeline initialized in {(t1 - t0)*1000:.1f}ms")

    def process(self, audio_path: str | Path) -> LdmPipelineOutput:
        """Process an audio file end-to-end to produce a structured intent."""
        latencies = {}
        
        # 1. ASR
        t0 = time.perf_counter()
        import soundfile as sf
        audio_array, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio_mono = audio_array.mean(axis=1) # convert to mono
        asr_result = self.asr_engine.transcribe_batch([audio_mono], [sr])[0]
        transcript = asr_result["text"]
        latencies["asr"] = (time.perf_counter() - t0) * 1000
        
        # 2 & 3. Language & Code-Mix Detection
        t0 = time.perf_counter()
        lang_res = self.lang_detector.detect(transcript)
        cm_res = self.cm_detector.detect(transcript)
        language = lang_res.language
        code_mixed = cm_res.code_mixed
        latencies["linguistic_detection"] = (time.perf_counter() - t0) * 1000
        
        # 4. Dialect Classification
        t0 = time.perf_counter()
        dialect_res = self.dialect_clf.predict_file(str(audio_path))
        dialect = dialect_res["dialect"]
        dialect_conf = dialect_res["confidence"]
        latencies["dialect_classification"] = (time.perf_counter() - t0) * 1000
        
        # 5. Linguistic Normalization
        t0 = time.perf_counter()
        norm_res = self.normalizer.normalize(transcript, dialect=dialect, language=language)
        interim_text = norm_res.normalized_text
        latencies["normalization"] = (time.perf_counter() - t0) * 1000
        
        # 6. Slang & Expression Mapping
        # Apply CSV-backed dictionaries on the text. 
        # (Expression first for multi-word phrases, then Slang for single tokens)
        t0 = time.perf_counter()
        # To avoid capitalizing the string early from normalization, we lower it for the dictionary mappers
        # Then we capitalize it again at the end.
        expr_out = self.expr_mapper.map(interim_text)
        final_norm_text = self.slang_mapper.apply(expr_out.output_text, drop_fillers=True)
        # Capitalize and add period
        if final_norm_text:
            final_norm_text = final_norm_text[0].upper() + final_norm_text[1:]
            if final_norm_text[-1] not in ".!?":
                final_norm_text += "."
        latencies["slang_mapping"] = (time.perf_counter() - t0) * 1000
        
        # 7. Intent & Entity Extraction
        t0 = time.perf_counter()
        intent_res = self.intent_validator.process(final_norm_text)
        latencies["intent_extraction"] = (time.perf_counter() - t0) * 1000
        
        latencies["total"] = sum(latencies.values())
        
        confidences = {
            "asr": 0.0, # Whisper engine currently doesn't return global confidence
            "language": lang_res.confidence,
            "dialect": dialect_conf,
            "intent": intent_res.confidence
        }
        
        return LdmPipelineOutput(
            transcript=transcript,
            language=language,
            code_mixed=code_mixed,
            dialect=dialect,
            normalized_text=final_norm_text,
            intent=intent_res.intent,
            entities=intent_res.entities,
            confidence=confidences,
            latency_ms=latencies
        )

    def process_text(self, text: str, dialect: str = None) -> LdmPipelineOutput:
        """Process a text transcript end-to-end (bypassing ASR)."""
        latencies = {}
        transcript = text.strip()
        
        # 1. ASR (bypassed for direct text)
        latencies["asr"] = 0.0
        
        # 2 & 3. Language & Code-Mix Detection
        t0 = time.perf_counter()
        lang_res = self.lang_detector.detect(transcript)
        cm_res = self.cm_detector.detect(transcript)
        language = lang_res.language
        code_mixed = cm_res.code_mixed
        latencies["linguistic_detection"] = (time.perf_counter() - t0) * 1000
        
        # 4. Dialect Classification
        t0 = time.perf_counter()
        if not dialect:
            lower = transcript.lower()
            chennai_markers = ["dei", "machi", "da", "bruh", "vaada"]
            madurai_markers = ["thambi", "aama", "ennanga"]
            kongu_markers = ["ayya", "yov", "la", "coimbatore"]
            nellai_markers = ["pa", "ppa", "phone pottu kudu"]
            
            if any(re.search(rf"\b{m}\b", lower) for m in chennai_markers):
                dialect = "Chennai"
            elif any(re.search(rf"\b{m}\b", lower) for m in madurai_markers):
                dialect = "Madurai"
            elif any(re.search(rf"\b{m}\b", lower) for m in kongu_markers):
                dialect = "Kongu"
            elif any(re.search(rf"\b{m}\b", lower) for m in nellai_markers):
                dialect = "Nellai"
            else:
                dialect = "Standard"
        dialect_conf = 0.90
        latencies["dialect_classification"] = (time.perf_counter() - t0) * 1000
        
        # 5. Linguistic Normalization
        t0 = time.perf_counter()
        norm_res = self.normalizer.normalize(transcript, dialect=dialect, language=language)
        interim_text = norm_res.normalized_text
        latencies["normalization"] = (time.perf_counter() - t0) * 1000
        
        # 6. Slang & Expression Mapping
        t0 = time.perf_counter()
        expr_out = self.expr_mapper.map(interim_text)
        final_norm_text = self.slang_mapper.apply(expr_out.output_text, drop_fillers=True)
        if final_norm_text:
            final_norm_text = final_norm_text[0].upper() + final_norm_text[1:]
            if final_norm_text[-1] not in ".!?":
                final_norm_text += "."
        latencies["slang_mapping"] = (time.perf_counter() - t0) * 1000
        
        # 7. Intent & Entity Extraction
        t0 = time.perf_counter()
        intent_res = self.intent_validator.process(final_norm_text)
        latencies["intent_extraction"] = (time.perf_counter() - t0) * 1000
        
        latencies["total"] = sum(latencies.values())
        
        confidences = {
            "asr": 1.0,
            "language": lang_res.confidence,
            "dialect": dialect_conf,
            "intent": intent_res.confidence
        }
        
        return LdmPipelineOutput(
            transcript=transcript,
            language=language,
            code_mixed=code_mixed,
            dialect=dialect,
            normalized_text=final_norm_text,
            intent=intent_res.intent,
            entities=intent_res.entities,
            confidence=confidences,
            latency_ms=latencies
        )

