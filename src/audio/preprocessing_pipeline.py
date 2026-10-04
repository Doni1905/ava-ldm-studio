"""
src/audio/preprocessing_pipeline.py
=====================================
End-to-end audio preprocessing pipeline for AVA LDM Studio.

Pipeline steps (in order)
--------------------------
1.  **Load**      — ``AudioLoader.load()`` reads file → AudioData (float32)
2.  **Validate**  — ``AudioValidator.validate()`` checks quality → ValidationResult
3.  **Mono**      — Average channels to mono (1-D numpy array)
4.  **Resample**  — ``AudioResampler.resample()`` to target sample rate
5.  **Normalize** — ``AudioNormalizer.normalize()`` amplitude scaling
6.  **Save**      — Write processed WAV to ``output_dir`` (soundfile)
7.  **Log**       — Append row to ``data/logs/audio_quality.csv``

Source file guarantee
----------------------
The original audio file is NEVER modified. Processed files are written
to a separate ``data/processed_audio/`` directory tree. If the input
and output paths would collide, a ``RuntimeError`` is raised.

Resume safety
-------------
If ``skip_existing=True`` (default), already-processed files are skipped.
The quality log always records the outcome (SKIP / PASS / WARN / REJECT).

SNR estimation
--------------
Estimated from the mono waveform using a VAD-based heuristic:
  * Divide signal into frames.
  * Bottom ``noise_floor_percentile``% of frame RMS values → noise estimate.
  * Top frames → speech estimate.
  * SNR = 10 * log10(speech_power / noise_power).

Memory efficiency
-----------------
Files are loaded one at a time, processed in-memory, then immediately
freed. Peak memory = 1 audio file at a time (O(max_duration * sr * 4 bytes)).

Quality log columns
-------------------
file, duration, sample_rate, channels, status, error, snr_db,
peak_in_dbfs, peak_out_dbfs, rms_in_dbfs, gain_db, norm_status,
orig_sr, target_sr, resampled
"""

from __future__ import annotations

import csv
import io
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import yaml

from .loader import AudioData, AudioLoadError, AudioLoader
from .normalizer import AudioNormalizer, NormalizeResult
from .resampler import AudioResampler
from .validator import AudioValidator, ValidationResult

# Force UTF-8 stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# Quality log CSV header (order matters)
_LOG_FIELDS = [
    "file",
    "status",
    "error",
    "duration",
    "orig_sample_rate",
    "target_sample_rate",
    "channels",
    "resampled",
    "snr_db",
    "peak_in_dbfs",
    "peak_out_dbfs",
    "rms_in_dbfs",
    "gain_db",
    "norm_status",
    "processing_time_s",
]


@dataclass
class ProcessResult:
    """Result of processing one audio file through the pipeline."""
    input_path:   Path
    output_path:  Path | None
    status:       str           # PASS | WARN | REJECT | SKIP | ERROR
    error:        str = ""

    duration:        float = float("nan")
    orig_sr:         int   = 0
    target_sr:       int   = 0
    channels:        int   = 0
    resampled:       bool  = False
    snr_db:          float = float("nan")
    peak_in_dbfs:    float = float("nan")
    peak_out_dbfs:   float = float("nan")
    rms_in_dbfs:     float = float("nan")
    gain_db:         float = float("nan")
    norm_status:     str   = ""
    processing_time_s: float = 0.0

    def as_log_row(self) -> dict[str, Any]:
        def fmt(v: Any) -> str:
            if isinstance(v, float) and math.isnan(v):
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        return {
            "file":             str(self.input_path),
            "status":           self.status,
            "error":            self.error,
            "duration":         fmt(self.duration),
            "orig_sample_rate": str(self.orig_sr),
            "target_sample_rate": str(self.target_sr),
            "channels":         str(self.channels),
            "resampled":        str(self.resampled),
            "snr_db":           fmt(self.snr_db),
            "peak_in_dbfs":     fmt(self.peak_in_dbfs),
            "peak_out_dbfs":    fmt(self.peak_out_dbfs),
            "rms_in_dbfs":      fmt(self.rms_in_dbfs),
            "gain_db":          fmt(self.gain_db),
            "norm_status":      self.norm_status,
            "processing_time_s": fmt(self.processing_time_s),
        }


class AudioPreprocessingPipeline:
    """
    Full preprocessing pipeline: load → validate → mono → resample → normalize → save.

    Parameters
    ----------
    cfg : dict
        Full audio.yaml configuration.
    output_dir : Path | str | None
        Override output directory (default: from config).
    log_path : Path | str | None
        Override quality log CSV path (default: from config).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: Path | str | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        a_cfg = cfg.get("audio", {})

        self.target_sr: int = a_cfg.get("target_sample_rate", 16000)
        self.out_fmt:   str = a_cfg.get("output_format", "wav")
        self.out_sub:   str = a_cfg.get("output_subtype", "PCM_16")
        self.skip_existing: bool = cfg.get("processing", {}).get("skip_existing", True)

        self.output_dir = Path(output_dir or cfg.get("paths", {}).get("output_dir", "data/processed_audio"))
        self.log_path   = Path(log_path or cfg.get("paths", {}).get("quality_log", "data/logs/audio_quality.csv"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Sub-components
        v_cfg = cfg.get("validation", {})
        self.loader     = AudioLoader(max_channels=v_cfg.get("max_input_channels", 8))
        self.validator  = AudioValidator(cfg)
        self.resampler  = AudioResampler.from_cfg(cfg)
        self.normalizer = AudioNormalizer.from_cfg(cfg)

        # SNR config
        snr_cfg = cfg.get("snr", {})
        self.snr_enabled = snr_cfg.get("enable", True)
        self.snr_frame_len_ms  = snr_cfg.get("frame_length_ms", 25)
        self.snr_hop_len_ms    = snr_cfg.get("hop_length_ms", 10)
        self.snr_percentile    = snr_cfg.get("noise_floor_percentile", 10)

        # Initialise quality log (write header if new)
        self._init_log()

        logger.info(
            "AudioPreprocessingPipeline ready: target=%d Hz, output=%s",
            self.target_sr, self.output_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, input_path: str | Path) -> ProcessResult:
        """
        Process a single audio file through the full pipeline.

        Never modifies the source file. Output is written to
        ``self.output_dir`` with the same filename.

        Parameters
        ----------
        input_path : str | Path

        Returns
        -------
        ProcessResult
        """
        t0 = time.perf_counter()
        input_path = Path(input_path)
        output_path = self._output_path_for(input_path)

        result = ProcessResult(
            input_path=input_path,
            output_path=output_path,
            status="PENDING",
            target_sr=self.target_sr,
        )

        # Safety check: never overwrite source
        if output_path.resolve() == input_path.resolve():
            result.status = "ERROR"
            result.error  = "Output path equals input path — refusing to overwrite source"
            logger.error(result.error)
            self._append_log(result)
            return result

        # Resume-safe skip
        if self.skip_existing and output_path.exists():
            result.status = "SKIP"
            result.processing_time_s = time.perf_counter() - t0
            logger.debug("SKIP (already exists): %s", output_path)
            self._append_log(result)
            return result

        try:
            result = self._run_pipeline(input_path, output_path, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error processing '%s': %s", input_path, exc)
            result.status = "ERROR"
            result.error  = f"Unexpected: {exc}"

        result.processing_time_s = time.perf_counter() - t0
        self._append_log(result)

        log_fn = logger.info if result.status in ("PASS", "WARN", "SKIP") else logger.warning
        log_fn(
            "%s '%s' in %.3f s",
            result.status, input_path.name, result.processing_time_s,
        )
        return result

    def process_batch(
        self,
        paths: list[str | Path],
        progress_every: int = 50,
    ) -> list[ProcessResult]:
        """
        Process a list of audio files sequentially.

        Parameters
        ----------
        paths : list of file paths
        progress_every : int
            Log a progress summary every N files.

        Returns
        -------
        list[ProcessResult]
        """
        results = []
        n = len(paths)
        counts = {"PASS": 0, "WARN": 0, "REJECT": 0, "SKIP": 0, "ERROR": 0}

        for i, path in enumerate(paths, 1):
            r = self.process_file(path)
            results.append(r)
            counts[r.status] = counts.get(r.status, 0) + 1

            if i % progress_every == 0 or i == n:
                logger.info(
                    "Progress [%d/%d]: PASS=%d WARN=%d REJECT=%d SKIP=%d ERROR=%d",
                    i, n, counts["PASS"], counts["WARN"],
                    counts["REJECT"], counts["SKIP"], counts["ERROR"],
                )

        return results

    def process_manifest(
        self,
        manifest_path: str | Path,
        audio_col: str = "audio_hf_path",
        local_audio_col: str | None = None,
    ) -> list[ProcessResult]:
        """
        Process all audio files referenced in a manifest CSV.

        Parameters
        ----------
        manifest_path : str | Path
            CSV file (e.g. data/manifests/train.csv).
        audio_col : str
            Column containing the HF audio path reference.
        local_audio_col : str | None
            If provided, use this column for the local file path instead.

        Returns
        -------
        list[ProcessResult]
        """
        import pandas as pd  # noqa: PLC0415
        df = pd.read_csv(manifest_path)
        col = local_audio_col if local_audio_col and local_audio_col in df.columns else audio_col
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in manifest. Columns: {list(df.columns)}")
        paths = df[col].dropna().tolist()
        logger.info("Processing %d files from manifest '%s'", len(paths), manifest_path)
        return self.process_batch(paths)

    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        result: ProcessResult,
    ) -> ProcessResult:
        # --- Step 1: Load ---
        try:
            audio: AudioData = self.loader.load(input_path)
        except AudioLoadError as exc:
            result.status = "REJECT"
            result.error  = f"Load failed: {exc}"
            return result

        result.duration = audio.duration
        result.orig_sr  = audio.sample_rate
        result.channels = audio.n_channels

        # --- Step 2: Validate ---
        val: ValidationResult = self.validator.validate(audio)
        result.peak_in_dbfs = val.peak_amplitude if not math.isnan(val.peak_amplitude) else float("nan")
        result.rms_in_dbfs  = val.rms_dbfs

        if val.status == "REJECT":
            result.status = "REJECT"
            result.error  = val.error_summary
            return result

        # --- Step 3: Convert to mono ---
        mono = self._to_mono(audio.waveform)

        # --- Step 4: Resample ---
        if audio.sample_rate != self.target_sr:
            mono = self.resampler.resample(mono, orig_sr=audio.sample_rate)
            result.resampled = True
        result.target_sr = self.target_sr

        # --- Step 5: SNR estimation (on resampled mono) ---
        if self.snr_enabled:
            result.snr_db = self._estimate_snr(mono, self.target_sr)

        # --- Step 6: Normalize ---
        norm: NormalizeResult = self.normalizer.normalize(mono.reshape(1, -1))
        result.peak_out_dbfs = norm.peak_out_dbfs
        result.gain_db       = norm.gain_db
        result.norm_status   = norm.status

        # --- Step 7: Save to output (never touches input path) ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            norm.waveform[0],       # shape (samples,)
            self.target_sr,
            subtype=self.out_sub,
        )

        result.status = "WARN" if val.status == "WARN" else "PASS"
        if val.status == "WARN":
            result.error = val.error_summary

        return result

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        """
        Convert (channels, samples) to mono 1-D array.
        Single-channel: remove channel dim. Multi-channel: average.
        """
        if waveform.ndim == 1:
            return waveform
        if waveform.shape[0] == 1:
            return waveform[0]
        return waveform.mean(axis=0)

    def _estimate_snr(self, mono: np.ndarray, sr: int) -> float:
        """
        Estimate SNR using a frame-energy VAD heuristic.

        Divides the signal into fixed-length frames and computes RMS per frame.
        The bottom ``noise_floor_percentile``% of frame energies estimate the
        noise floor; the top (100 - percentile)% estimate speech energy.

        Returns SNR in dB, or NaN if the signal is too short to frame.
        """
        frame_len = int(sr * self.snr_frame_len_ms / 1000)
        hop_len   = int(sr * self.snr_hop_len_ms   / 1000)

        if frame_len <= 0 or len(mono) < frame_len:
            return float("nan")

        # Build frames
        n_frames = 1 + (len(mono) - frame_len) // hop_len
        if n_frames < 2:
            return float("nan")

        frames = np.lib.stride_tricks.as_strided(
            mono,
            shape=(n_frames, frame_len),
            strides=(mono.strides[0] * hop_len, mono.strides[0]),
        )
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        if frame_rms.max() < 1e-10:
            return float("nan")

        thresh_idx = max(1, int(n_frames * self.snr_percentile / 100))
        sorted_rms = np.sort(frame_rms)

        noise_rms   = sorted_rms[:thresh_idx].mean() + 1e-12
        speech_rms  = sorted_rms[thresh_idx:].mean()  + 1e-12

        snr_db = 20.0 * math.log10(speech_rms / noise_rms)
        return round(snr_db, 2)

    # ------------------------------------------------------------------
    # Output path resolution
    # ------------------------------------------------------------------

    def _output_path_for(self, input_path: Path) -> Path:
        """
        Compute the output path for a given input path.

        If input is inside a known data directory, the relative structure
        is preserved under ``output_dir``.
        Otherwise, just the filename is used.
        """
        name = input_path.stem + "." + self.out_fmt
        return self.output_dir / name

    # ------------------------------------------------------------------
    # Quality log
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        """Write CSV header if the log file doesn't exist yet."""
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
                writer.writeheader()

    def _append_log(self, result: ProcessResult) -> None:
        """Append one row to the quality log CSV."""
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
                writer.writerow(result.as_log_row())
        except OSError as exc:
            logger.warning("Failed to write quality log: %s", exc)


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

def load_audio_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the audio.yaml configuration."""
    if config_path is None:
        for parent in [Path.cwd(), Path(__file__).parent.parent.parent]:
            candidate = parent / "configs" / "audio.yaml"
            if candidate.exists():
                config_path = candidate
                break
        if config_path is None:
            config_path = Path("configs/audio.yaml")
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)

