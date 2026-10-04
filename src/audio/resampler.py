"""
src/audio/resampler.py
=======================
Audio resampling for AVA LDM Studio.

Design
------
* Primary engine: ``soxr`` — industry-standard high-quality resampler,
  very low memory footprint, CPU-only, no C extension compile required.
* Fallback: ``scipy.signal.resample_poly`` — pure scipy, always available.
* Input: numpy float32 array (1-D mono or 2-D multi-channel).
* Output: numpy float32 array at ``target_sample_rate``.
* No-op when input already at target rate (returns a view, not a copy).

Memory efficiency
-----------------
soxr processes audio in streaming fashion internally, keeping memory
usage proportional to the chunk size, not the full file.

Thread safety
-------------
Each ``AudioResampler`` instance owns its own state. Safe to call
``resample()`` concurrently from different threads with different instances.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Available engines (checked at import time)
_SOXR_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import soxr as _soxr
    _SOXR_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.signal import resample_poly as _resample_poly
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

if not _SOXR_AVAILABLE and not _SCIPY_AVAILABLE:
    raise ImportError(
        "No resampling engine available. "
        "Install either 'soxr' or 'scipy': pip install soxr"
    )


# soxr quality constants
_SOXR_QUALITY_MAP = {
    "VHQ": "VHQ",   # Very High Quality — 144 dB SNR
    "HQ":  "HQ",    # High Quality      — 120 dB SNR
    "MQ":  "MQ",    # Medium Quality    — 96 dB SNR
    "LQ":  "LQ",    # Low Quality
    "QQ":  "QQ",    # Quick & Quiet (fastest)
}


class AudioResampler:
    """
    Resample audio to a target sample rate.

    Parameters
    ----------
    target_sample_rate : int
        Output sample rate in Hz.
    engine : str
        ``"soxr"`` (default) or ``"scipy"``.
    soxr_quality : str
        soxr quality preset: VHQ | HQ | MQ | LQ | QQ.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        engine: str = "soxr",
        soxr_quality: str = "HQ",
    ) -> None:
        self.target_sr = target_sample_rate
        self.soxr_quality = _SOXR_QUALITY_MAP.get(soxr_quality, "HQ")

        # Resolve engine
        if engine == "soxr" and _SOXR_AVAILABLE:
            self._engine = "soxr"
        elif _SCIPY_AVAILABLE:
            if engine == "soxr":
                logger.warning("soxr not available — falling back to scipy.")
            self._engine = "scipy"
        else:
            self._engine = "soxr"  # will fail on first call — caught above

        logger.debug(
            "AudioResampler: target=%d Hz, engine=%s, quality=%s",
            self.target_sr, self._engine, self.soxr_quality,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
    ) -> np.ndarray:
        """
        Resample *waveform* from *orig_sr* to ``self.target_sr``.

        Parameters
        ----------
        waveform : np.ndarray
            Shape ``(samples,)`` for mono or ``(channels, samples)`` for multi-channel.
            Must be float32 or float64.
        orig_sr : int
            Source sample rate in Hz.

        Returns
        -------
        np.ndarray
            Resampled waveform, float32, same shape convention as input.
        """
        if orig_sr == self.target_sr:
            logger.debug("No resampling needed (%d Hz)", orig_sr)
            return waveform.astype(np.float32, copy=False)

        if orig_sr <= 0 or self.target_sr <= 0:
            raise ValueError(f"Invalid sample rates: {orig_sr} -> {self.target_sr}")

        is_2d = waveform.ndim == 2
        if waveform.ndim == 1:
            arr = waveform
        elif waveform.ndim == 2:
            arr = waveform  # (channels, samples)
        else:
            raise ValueError(f"Expected 1D or 2D waveform, got shape {waveform.shape}")

        logger.debug(
            "Resample %d -> %d Hz using %s, shape=%s",
            orig_sr, self.target_sr, self._engine, arr.shape,
        )

        if self._engine == "soxr":
            resampled = self._soxr_resample(arr, orig_sr)
        else:
            resampled = self._scipy_resample(arr, orig_sr)

        return resampled.astype(np.float32, copy=False)

    def is_noop(self, orig_sr: int) -> bool:
        """Return True if no resampling is needed."""
        return orig_sr == self.target_sr

    # ------------------------------------------------------------------
    # Engine implementations
    # ------------------------------------------------------------------

    def _soxr_resample(self, arr: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample using soxr. Handles both mono and multi-channel."""
        if arr.ndim == 1:
            return _soxr.resample(arr, orig_sr, self.target_sr, quality=self.soxr_quality)
        # Multi-channel: soxr expects (samples, channels)
        arr_T = arr.T  # (samples, channels)
        out_T = _soxr.resample(arr_T, orig_sr, self.target_sr, quality=self.soxr_quality)
        return out_T.T  # back to (channels, samples)

    def _scipy_resample(self, arr: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample using scipy.signal.resample_poly.
        Computes the GCD to find the minimal up/down factors.
        """
        g = math.gcd(orig_sr, self.target_sr)
        up   = self.target_sr // g
        down = orig_sr // g

        if arr.ndim == 1:
            return _resample_poly(arr, up, down).astype(np.float32)
        # Multi-channel: process axis=-1 (samples axis)
        return _resample_poly(arr, up, down, axis=-1).astype(np.float32)

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "AudioResampler":
        """Construct from the audio.yaml config dict."""
        a = cfg.get("audio", cfg)
        return cls(
            target_sample_rate=a.get("target_sample_rate", 16000),
            engine=a.get("resampler_engine", "soxr"),
            soxr_quality=a.get("soxr_quality", "HQ"),
        )

