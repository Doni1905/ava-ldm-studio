"""
src/audio/normalizer.py
========================
Audio amplitude normalization for AVA LDM Studio.

Strategies
----------
``peak``  — Scale so that the maximum absolute sample = target_dbfs.
            Fast, deterministic, avoids clipping. Default.
``rms``   — Scale so that the RMS of the full file = target_dbfs.
            Gives consistent perceived loudness across speakers.
``none``  — Pass through unchanged.

Safe-guards
-----------
* Silent files (RMS below ``silence_threshold_dbfs``) are returned
  unchanged with a SILENT status — never divide-by-zero.
* After scaling, if peak exceeds ``clip_guard_dbfs``, the waveform is
  hard-clipped at that threshold instead of applying a gain > 1.0
  (protects against RMS normalization on extremely dynamic audio).
* All operations are in-place on a copy — original ``AudioData.waveform``
  is never mutated.

Returns
-------
``NormalizeResult`` with the scaled waveform, applied gain_db, and status.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .validator import _to_dbfs

logger = logging.getLogger(__name__)

# Normalizer status values
NORM_OK     = "OK"
NORM_SILENT = "SILENT"   # file was too quiet to normalize
NORM_CLAMP  = "CLAMPED"  # post-norm clip guard triggered


@dataclass
class NormalizeResult:
    """Result of normalizing one audio file."""
    waveform: np.ndarray   # float32, shape (channels, samples)  — COPY
    gain_db: float         # dB gain applied (0.0 = no change)
    status: str            # OK | SILENT | CLAMPED
    peak_in_dbfs: float    # peak before normalization
    peak_out_dbfs: float   # peak after normalization
    rms_in_dbfs: float     # RMS before normalization


class AudioNormalizer:
    """
    Normalize audio amplitude.

    Parameters
    ----------
    strategy : str
        ``"peak"`` | ``"rms"`` | ``"none"``.
    peak_target_dbfs : float
        Target peak level in dBFS for peak normalization.
    rms_target_dbfs : float
        Target RMS level in dBFS for rms normalization.
    clip_guard_dbfs : float
        After normalization, hard-clip at this peak level.
    silence_threshold_dbfs : float
        Files with RMS below this are not normalized (returned as-is).
    """

    def __init__(
        self,
        strategy: str = "peak",
        peak_target_dbfs: float = -3.0,
        rms_target_dbfs: float = -20.0,
        clip_guard_dbfs: float = -0.1,
        silence_threshold_dbfs: float = -60.0,
    ) -> None:
        if strategy not in ("peak", "rms", "none"):
            raise ValueError(f"Unknown normalization strategy: {strategy!r}")
        self.strategy = strategy
        self.peak_target_linear  = _dbfs_to_linear(peak_target_dbfs)
        self.rms_target_linear   = _dbfs_to_linear(rms_target_dbfs)
        self.clip_guard_linear   = _dbfs_to_linear(clip_guard_dbfs)
        self.silence_thresh_rms  = _dbfs_to_linear(silence_threshold_dbfs)

        logger.debug(
            "AudioNormalizer: strategy=%s, peak_target=%.1f dBFS, rms_target=%.1f dBFS",
            strategy, peak_target_dbfs, rms_target_dbfs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, waveform: np.ndarray) -> NormalizeResult:
        """
        Normalize *waveform* according to the configured strategy.

        Parameters
        ----------
        waveform : np.ndarray
            Shape ``(channels, samples)``, float32. Not mutated.

        Returns
        -------
        NormalizeResult
        """
        out = waveform.copy()
        mono = out.mean(axis=0) if out.ndim == 2 else out

        peak_in  = float(np.max(np.abs(mono)))
        rms_in   = float(np.sqrt(np.mean(mono ** 2)))
        peak_in_dbfs = _to_dbfs(peak_in)
        rms_in_dbfs  = _to_dbfs(rms_in)

        # Silent file guard
        if rms_in < self.silence_thresh_rms:
            logger.debug("Silent file — normalization skipped (RMS=%.1f dBFS)", rms_in_dbfs)
            return NormalizeResult(
                waveform=out.astype(np.float32),
                gain_db=0.0,
                status=NORM_SILENT,
                peak_in_dbfs=peak_in_dbfs,
                peak_out_dbfs=peak_in_dbfs,
                rms_in_dbfs=rms_in_dbfs,
            )

        if self.strategy == "none":
            return NormalizeResult(
                waveform=out.astype(np.float32),
                gain_db=0.0,
                status=NORM_OK,
                peak_in_dbfs=peak_in_dbfs,
                peak_out_dbfs=peak_in_dbfs,
                rms_in_dbfs=rms_in_dbfs,
            )

        # Compute gain
        if self.strategy == "peak":
            gain = self.peak_target_linear / (peak_in + 1e-12)
        else:  # rms
            gain = self.rms_target_linear / (rms_in + 1e-12)

        gain_db = _to_dbfs(gain) if gain > 0 else float("-inf")

        # Apply gain
        out = out * gain

        # Clip guard
        clamped = False
        out_peak = float(np.max(np.abs(out)))
        if out_peak > self.clip_guard_linear:
            out = np.clip(out, -self.clip_guard_linear, self.clip_guard_linear)
            clamped = True
            logger.debug(
                "Clip guard applied: peak was %.4f, clamped to %.4f",
                out_peak, self.clip_guard_linear,
            )

        out = out.astype(np.float32)
        peak_out_dbfs = _to_dbfs(float(np.max(np.abs(out))))

        logger.debug(
            "Normalized: gain=%.2f dB, peak %.1f -> %.1f dBFS, %s",
            gain_db, peak_in_dbfs, peak_out_dbfs,
            "CLAMPED" if clamped else "OK",
        )

        return NormalizeResult(
            waveform=out,
            gain_db=gain_db,
            status=NORM_CLAMP if clamped else NORM_OK,
            peak_in_dbfs=peak_in_dbfs,
            peak_out_dbfs=peak_out_dbfs,
            rms_in_dbfs=rms_in_dbfs,
        )

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "AudioNormalizer":
        """Construct from the audio.yaml config dict."""
        n = cfg.get("normalization", cfg)
        return cls(
            strategy=n.get("strategy", "peak"),
            peak_target_dbfs=n.get("peak_target_dbfs", -3.0),
            rms_target_dbfs=n.get("rms_target_dbfs", -20.0),
            clip_guard_dbfs=n.get("clip_guard_dbfs", -0.1),
            silence_threshold_dbfs=n.get("silence_threshold_dbfs", -60.0),
        )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude. -inf → 0."""
    if dbfs == float("-inf"):
        return 0.0
    return 10 ** (dbfs / 20.0)

