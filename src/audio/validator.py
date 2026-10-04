"""
src/audio/validator.py
=======================
Audio quality validation for AVA LDM Studio.

Validates an ``AudioData`` object (produced by ``AudioLoader``) against
quality thresholds defined in ``configs/audio.yaml``.

Validation checks (in order)
------------------------------
1.  Duration hard limits (min/max) — REJECT if violated.
2.  Duration soft limits (warn_min/warn_max) — WARN if violated.
3.  Sample rate plausibility (> 0, no absurdly high values).
4.  Channel count (within configured maximum).
5.  Peak amplitude (silent file detection, clipping detection).
6.  NaN / Inf detection — always REJECT.
7.  Constant DC-offset only signal (no actual audio content).

Status values
-------------
``PASS``   — all checks passed, file may be processed.
``WARN``   — soft-limit violations; file is processed with a note.
``REJECT`` — hard violation; file is skipped by the pipeline.

Returns a ``ValidationResult`` dataclass.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .loader import AudioData

logger = logging.getLogger(__name__)

# Status constants
PASS   = "PASS"
WARN   = "WARN"
REJECT = "REJECT"


@dataclass
class ValidationResult:
    """Result of validating one audio file."""
    status: str           # PASS | WARN | REJECT
    messages: list[str] = field(default_factory=list)
    peak_amplitude: float = float("nan")
    rms_dbfs: float       = float("nan")
    is_clipped: bool      = False
    is_silent: bool       = False
    has_nan_inf: bool     = False

    def add(self, level: str, msg: str) -> None:
        self.messages.append(f"[{level}] {msg}")
        if level == REJECT and self.status != REJECT:
            self.status = REJECT
        elif level == WARN and self.status == PASS:
            self.status = WARN

    @property
    def error_summary(self) -> str:
        return "; ".join(self.messages) if self.messages else ""

    def __str__(self) -> str:
        return f"ValidationResult(status={self.status}, messages={self.messages})"


class AudioValidator:
    """
    Validate an ``AudioData`` object against quality thresholds.

    Parameters
    ----------
    cfg : dict
        Full audio.yaml configuration (or just the ``validation`` section).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        v = cfg.get("validation", cfg)
        self.min_dur:  float = v.get("min_duration_s", 0.1)
        self.max_dur:  float = v.get("max_duration_s", 60.0)
        self.warn_min: float = v.get("warn_min_duration_s", 0.5)
        self.warn_max: float = v.get("warn_max_duration_s", 30.0)
        self.min_peak: float = v.get("min_peak_amplitude", 1e-6)
        self.max_peak: float = v.get("max_peak_amplitude", 1.1)
        self.max_ch:   int   = v.get("max_input_channels", 8)
        allowed = v.get("allowed_input_sample_rates", [])
        self.allowed_sr: set[int] = set(allowed) if allowed else set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, audio: AudioData) -> ValidationResult:
        """
        Run all validation checks on *audio*.

        Parameters
        ----------
        audio : AudioData

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(status=PASS)
        mono = audio.waveform_mono

        # 1. NaN / Inf check (always first — corrupts all other metrics)
        self._check_nan_inf(mono, result)
        if result.has_nan_inf:
            return result  # no point checking further

        # 2. Duration hard limits
        self._check_duration_hard(audio.duration, result)
        if result.status == REJECT:
            return result

        # 3. Duration soft limits
        self._check_duration_soft(audio.duration, result)

        # 4. Sample rate
        self._check_sample_rate(audio.sample_rate, result)

        # 5. Channel count
        self._check_channels(audio.n_channels, result)

        # 6. Amplitude (peak + RMS)
        self._check_amplitude(mono, result)

        # 7. Constant signal (DC offset only)
        self._check_constant_signal(mono, result)

        if result.status == PASS:
            logger.debug("PASS '%s' (%.3f s)", audio.file_path.name, audio.duration)
        else:
            logger.info(
                "%s '%s': %s",
                result.status, audio.file_path.name, result.error_summary,
            )

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_nan_inf(self, mono: np.ndarray, result: ValidationResult) -> None:
        if not np.isfinite(mono).all():
            result.has_nan_inf = True
            result.add(REJECT, "Audio contains NaN or Inf values (corrupted file)")

    def _check_duration_hard(self, dur: float, result: ValidationResult) -> None:
        if dur < self.min_dur:
            result.add(
                REJECT,
                f"Duration {dur:.3f}s < hard min {self.min_dur}s (too short/corrupted)",
            )
        elif dur > self.max_dur:
            result.add(
                REJECT,
                f"Duration {dur:.1f}s > hard max {self.max_dur}s (suspiciously long)",
            )

    def _check_duration_soft(self, dur: float, result: ValidationResult) -> None:
        if dur < self.warn_min:
            result.add(WARN, f"Duration {dur:.3f}s < recommended min {self.warn_min}s")
        elif dur > self.warn_max:
            result.add(WARN, f"Duration {dur:.1f}s > recommended max {self.warn_max}s")

    def _check_sample_rate(self, sr: int, result: ValidationResult) -> None:
        if sr <= 0 or sr > 384_000:
            result.add(REJECT, f"Implausible sample rate: {sr} Hz")
        elif self.allowed_sr and sr not in self.allowed_sr:
            result.add(WARN, f"Sample rate {sr} Hz not in allowed list {sorted(self.allowed_sr)}")

    def _check_channels(self, n_ch: int, result: ValidationResult) -> None:
        if n_ch > self.max_ch:
            result.add(REJECT, f"Too many channels: {n_ch} > max {self.max_ch}")
        elif n_ch < 1:
            result.add(REJECT, f"Invalid channel count: {n_ch}")

    def _check_amplitude(self, mono: np.ndarray, result: ValidationResult) -> None:
        peak = float(np.max(np.abs(mono)))
        result.peak_amplitude = peak

        # RMS in dBFS
        rms = float(np.sqrt(np.mean(mono ** 2)))
        result.rms_dbfs = _to_dbfs(rms)

        if peak < self.min_peak:
            result.is_silent = True
            result.add(REJECT, f"Peak amplitude {peak:.2e} < threshold {self.min_peak:.2e} (silent/DC)")
        elif peak > self.max_peak:
            result.is_clipped = True
            result.add(WARN, f"Peak amplitude {peak:.4f} > {self.max_peak:.2f} (possible clipping)")

    def _check_constant_signal(self, mono: np.ndarray, result: ValidationResult) -> None:
        """Reject files where std dev is effectively zero (pure DC)."""
        std = float(np.std(mono))
        if std < 1e-9:
            result.add(REJECT, f"Signal has no variation (std={std:.2e}) — DC offset only")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _to_dbfs(linear: float) -> float:
    """Convert linear amplitude to dBFS. Returns -inf for zero."""
    if linear <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(linear)

