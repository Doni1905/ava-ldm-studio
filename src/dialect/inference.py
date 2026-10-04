"""
src/dialect/inference.py
=========================
Inference pipeline for Tamil dialect classification.

Input  : audio file path (WAV at 16kHz mono, or any soundfile-readable format)
Output : {
    "dialect":    str,          # predicted dialect label
    "confidence": float,        # probability of predicted class [0,1]
    "top_k": [                  # top-k predictions with probabilities
        {"dialect": str, "probability": float},
        ...
    ]
}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from .classifier import DialectClassifier
from .labels import DialectLabelRegistry

logger = logging.getLogger(__name__)


class DialectInferencePipeline:
    """
    Run dialect classification on audio files.

    Parameters
    ----------
    model : DialectClassifier
        A loaded/trained classifier.
    label_registry : DialectLabelRegistry
    config : dict
        Parsed ``configs/dialect.yaml``.
    """

    def __init__(
        self,
        model: DialectClassifier,
        label_registry: DialectLabelRegistry,
        config: Dict[str, Any],
    ) -> None:
        self.model    = model
        self.registry = label_registry
        inf_cfg       = config.get("inference", {})
        self.top_k    = inf_cfg.get("top_k", 2)
        self.max_len  = config.get("training", {}).get("max_audio_length", 80000)

        device_str = inf_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model.to(self.device)
        self.model.eval()

    def predict_file(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Predict the dialect of a single audio file.

        Returns
        -------
        dict with keys: dialect, confidence, top_k, inference_ms
        """
        t0 = time.time()

        wave = self._load_audio(str(audio_path))
        if wave is None:
            return self._error_result(str(audio_path), "Failed to load audio")

        result = self._infer_waveform(wave)
        result["audio_path"]    = str(audio_path)
        result["inference_ms"]  = round((time.time() - t0) * 1000, 1)
        return result

    def predict_batch(self, audio_paths: List[str | Path]) -> List[Dict[str, Any]]:
        """
        Predict dialects for a list of audio files.

        Returns one result dict per file.
        """
        results = []
        for path in audio_paths:
            results.append(self.predict_file(path))
        return results

    @torch.no_grad()
    def _infer_waveform(self, wave: np.ndarray) -> Dict[str, Any]:
        """Run inference on a pre-loaded mono float32 waveform."""
        wave = wave[:self.max_len]
        waveform  = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(self.device)
        attn_mask = torch.ones_like(waveform, dtype=torch.long)

        probs     = self.model.predict_proba(waveform, attn_mask).cpu().numpy()[0]
        top_idxs  = probs.argsort()[::-1][:self.top_k]

        top_k_list = [
            {"dialect": self.registry.idx_to_label(int(i)), "probability": round(float(probs[i]), 4)}
            for i in top_idxs
        ]
        best = top_k_list[0]

        return {
            "dialect":    best["dialect"],
            "confidence": best["probability"],
            "top_k":      top_k_list,
        }

    def _load_audio(self, path: str) -> Optional[np.ndarray]:
        """Load audio to mono float32. Returns None on failure."""
        try:
            wave, _ = sf.read(path, dtype="float32", always_2d=True)
            return wave.mean(axis=1)
        except Exception as e:
            logger.error(f"Cannot load audio {path}: {e}")
            return None

    @staticmethod
    def _error_result(path: str, reason: str) -> Dict[str, Any]:
        return {
            "audio_path":   path,
            "dialect":      "unknown",
            "confidence":   0.0,
            "top_k":        [],
            "error":        reason,
            "inference_ms": 0.0,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Dict[str, Any],
        label_registry: DialectLabelRegistry,
    ) -> "DialectInferencePipeline":
        """Load model from checkpoint and return a ready-to-use pipeline."""
        model = DialectClassifier.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            num_classes=label_registry.num_classes,
        )
        return cls(model=model, label_registry=label_registry, config=config)
