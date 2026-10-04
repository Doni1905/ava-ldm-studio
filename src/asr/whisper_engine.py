"""
src/asr/whisper_engine.py
=========================
Whisper implementation of the ASREngine.
Uses Hugging Face Transformers pipeline.
"""

import logging
from typing import Any, Dict, List
import numpy as np
import torch
from transformers import pipeline

from .asr_engine import ASREngine

logger = logging.getLogger(__name__)

class WhisperEngine(ASREngine):
    """
    Hugging Face Whisper model wrapper.
    """
    
    def __init__(self, config: Dict[str, Any]):
        model_cfg = config.get("model", {})
        inf_cfg = config.get("inference", {})
        
        self.model_id = model_cfg.get("model_id", "openai/whisper-tiny")
        self.language = model_cfg.get("language", "tamil")
        self.task = model_cfg.get("task", "transcribe")
        self.return_timestamps = model_cfg.get("return_timestamps", True)
        
        # Device selection
        device_str = inf_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str
            
        logger.info(f"Loading Whisper model '{self.model_id}' on {self.device}...")
        
        # Initialize pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            # If PyTorch 2.0+ is used, torch_dtype can be float16 for CUDA, but stick to float32 for CPU for compatibility
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        )
        
        # Force generation config
        self.generate_kwargs = {
            "task": self.task,
            "language": self.language,
        }
        
        # We handle batching outside the pipeline for better error recovery and progress tracking,
        # but the pipeline can also accept batches natively. 
        self.batch_size = inf_cfg.get("batch_size", 4)
        logger.info("Whisper model loaded successfully.")

    def get_target_sample_rate(self) -> int:
        return 16000  # Whisper expects 16kHz

    def transcribe_batch(
        self, 
        audio_arrays: List[np.ndarray], 
        sample_rates: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Transcribe a batch of numpy arrays.
        """
        if not audio_arrays:
            return []
            
        # The transformers pipeline expects inputs in the form of {"array": np.ndarray, "sampling_rate": int}
        # or just raw arrays if sampling_rate is handled, but dictionary is safer.
        inputs = []
        for arr, sr in zip(audio_arrays, sample_rates):
            if sr != self.get_target_sample_rate():
                logger.warning(f"Expected {self.get_target_sample_rate()}Hz, got {sr}Hz. Whisper may perform poorly.")
            inputs.append({"array": arr, "sampling_rate": sr})
            
        # Run inference
        outputs = self.pipe(
            inputs,
            batch_size=len(inputs),
            generate_kwargs=self.generate_kwargs,
            return_timestamps=self.return_timestamps
        )
        
        # Pipeline returns a dict if 1 input, or list of dicts if multiple inputs.
        if isinstance(outputs, dict):
            outputs = [outputs]
            
        return outputs

