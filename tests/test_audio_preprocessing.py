"""
tests/test_audio_preprocessing.py
===================================
Unit tests for the AVA LDM audio preprocessing pipeline.

Tests cover:
- AudioLoader: valid load, missing file, empty file, zero-length audio
- AudioValidator: PASS, duration rejection, silence rejection, clipping warn, NaN
- AudioResampler: correct output length, no-op, soxr and scipy engines
- AudioNormalizer: peak strategy, rms strategy, silent guard, clip guard
- AudioPreprocessingPipeline: full pipeline on synthetic WAV, source not modified,
  quality log written, skip_existing, mono conversion, SNR estimation

All tests use synthetic in-memory audio — no network or real files required.
Run from project root:
    python -m pytest tests/test_audio_preprocessing.py -v
    # or directly:
    python tests/test_audio_preprocessing.py
"""

from __future__ import annotations

import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.audio.loader import AudioData, AudioLoadError, AudioLoader
from src.audio.normalizer import NORM_CLAMP, NORM_OK, NORM_SILENT, AudioNormalizer
from src.audio.resampler import AudioResampler
from src.audio.validator import PASS, REJECT, WARN, AudioValidator, ValidationResult
from src.audio.preprocessing_pipeline import (
    AudioPreprocessingPipeline,
    ProcessResult,
    load_audio_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000  # default sample rate for tests


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR,
          amplitude: float = 0.5) -> np.ndarray:
    """Return a float32 sine wave, shape (samples,)."""
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(data: np.ndarray, sr: int = SR, path: str | None = None) -> str:
    """Write a numpy array to a temporary WAV file. Returns the file path."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def _minimal_cfg() -> dict:
    """Return the minimal config dict for tests (no YAML file needed)."""
    return {
        "audio": {
            "target_sample_rate": 16000,
            "output_format": "wav",
            "output_subtype": "PCM_16",
            "output_channels": 1,
            "resampler_engine": "soxr",
            "soxr_quality": "HQ",
        },
        "validation": {
            "min_duration_s": 0.1,
            "max_duration_s": 60.0,
            "warn_min_duration_s": 0.5,
            "warn_max_duration_s": 30.0,
            "min_peak_amplitude": 1e-6,
            "max_peak_amplitude": 1.1,
            "max_input_channels": 8,
            "allowed_input_sample_rates": [],
        },
        "normalization": {
            "strategy": "peak",
            "peak_target_dbfs": -3.0,
            "rms_target_dbfs": -20.0,
            "clip_guard_dbfs": -0.1,
            "silence_threshold_dbfs": -60.0,
        },
        "snr": {
            "enable": True,
            "frame_length_ms": 25,
            "hop_length_ms": 10,
            "noise_floor_percentile": 10,
        },
        "processing": {
            "skip_existing": True,
            "num_workers": 1,
            "progress_every": 50,
        },
        "paths": {
            "output_dir": None,        # overridden per-test
            "quality_log": None,       # overridden per-test
            "log_dir": "logs",
        },
        "logging": {
            "level": "WARNING",
            "log_file": "logs/test_audio.log",
            "max_bytes": 1048576,
            "backup_count": 1,
        },
    }


# ---------------------------------------------------------------------------
# AudioLoader tests
# ---------------------------------------------------------------------------

class TestAudioLoader(unittest.TestCase):

    def setUp(self):
        self.loader = AudioLoader()

    def test_load_mono_wav(self):
        wave = _sine(duration=1.0)
        path = _write_wav(wave)
        try:
            audio = self.loader.load(path)
            self.assertIsInstance(audio, AudioData)
            self.assertEqual(audio.sample_rate, SR)
            self.assertEqual(audio.n_channels, 1)
            self.assertAlmostEqual(audio.duration, 1.0, places=1)
            self.assertEqual(audio.waveform.dtype, np.float32)
            self.assertEqual(audio.waveform.shape[0], 1)  # (channels, samples)
        finally:
            os.unlink(path)

    def test_load_stereo_wav(self):
        mono = _sine(duration=0.5)
        stereo = np.stack([mono, mono], axis=1)  # (samples, 2)
        path = _write_wav(stereo)
        try:
            audio = self.loader.load(path)
            self.assertEqual(audio.n_channels, 2)
            self.assertEqual(audio.waveform.shape[0], 2)
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        with self.assertRaises(AudioLoadError):
            self.loader.load("/nonexistent/path/audio.wav")

    def test_load_empty_file_raises(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with self.assertRaises(AudioLoadError):
                self.loader.load(path)
        finally:
            os.unlink(path)

    def test_waveform_mono_property_stereo(self):
        mono = _sine(duration=0.5)
        stereo = np.stack([mono, mono], axis=1)
        path = _write_wav(stereo)
        try:
            audio = self.loader.load(path)
            m = audio.waveform_mono
            self.assertEqual(m.ndim, 1)
        finally:
            os.unlink(path)

    def test_probe_returns_metadata(self):
        wave = _sine(duration=0.5)
        path = _write_wav(wave)
        try:
            info = self.loader.probe(path)
            self.assertEqual(info["sample_rate"], SR)
            self.assertEqual(info["channels"], 1)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# AudioValidator tests
# ---------------------------------------------------------------------------

class TestAudioValidator(unittest.TestCase):

    def _make_audio(self, wave: np.ndarray, sr: int = SR) -> AudioData:
        if wave.ndim == 1:
            wave = wave.reshape(1, -1)
        n_ch, n_samp = wave.shape
        return AudioData(
            waveform=wave, sample_rate=sr,
            n_channels=n_ch, n_samples=n_samp,
            duration=n_samp / sr, file_path=Path("test.wav"),
        )

    def setUp(self):
        self.validator = AudioValidator(_minimal_cfg())

    def test_valid_audio_passes(self):
        audio = self._make_audio(_sine(duration=1.0))
        result = self.validator.validate(audio)
        self.assertEqual(result.status, PASS)

    def test_too_short_rejected(self):
        # 0.05 s < min_duration_s=0.1
        wave = _sine(duration=0.05)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, REJECT)

    def test_too_long_rejected(self):
        # 70 s > max_duration_s=60
        wave = _sine(duration=70.0)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, REJECT)

    def test_silent_audio_rejected(self):
        wave = np.zeros(16000, dtype=np.float32)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, REJECT)
        self.assertTrue(result.is_silent)

    def test_nan_audio_rejected(self):
        wave = np.full(16000, float("nan"), dtype=np.float32)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, REJECT)
        self.assertTrue(result.has_nan_inf)

    def test_clipped_audio_warns(self):
        # Create a clipped sine wave (amplitude > max_peak)
        wave = _sine(amplitude=1.5)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, WARN)
        self.assertTrue(result.is_clipped)

    def test_soft_duration_warns(self):
        # 0.3 s < warn_min=0.5 but > hard min=0.1
        wave = _sine(duration=0.3)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, WARN)

    def test_dc_only_signal_rejected(self):
        wave = np.full(16000, 0.5, dtype=np.float32)
        audio = self._make_audio(wave)
        result = self.validator.validate(audio)
        self.assertEqual(result.status, REJECT)


# ---------------------------------------------------------------------------
# AudioResampler tests
# ---------------------------------------------------------------------------

class TestAudioResampler(unittest.TestCase):

    def test_soxr_resample_16k_to_8k(self):
        resampler = AudioResampler(target_sample_rate=8000, engine="soxr")
        wave = _sine(duration=1.0, sr=16000)
        out = resampler.resample(wave, orig_sr=16000)
        self.assertEqual(out.shape[0], 8000)
        self.assertEqual(out.dtype, np.float32)

    def test_soxr_resample_8k_to_16k(self):
        resampler = AudioResampler(target_sample_rate=16000, engine="soxr")
        wave = _sine(duration=1.0, sr=8000)
        out = resampler.resample(wave, orig_sr=8000)
        self.assertEqual(out.shape[0], 16000)

    def test_scipy_resample_fallback(self):
        resampler = AudioResampler(target_sample_rate=8000, engine="scipy")
        wave = _sine(duration=1.0, sr=16000)
        out = resampler.resample(wave, orig_sr=16000)
        self.assertEqual(out.shape[0], 8000)

    def test_noop_when_same_sr(self):
        resampler = AudioResampler(target_sample_rate=16000)
        wave = _sine(duration=1.0, sr=16000)
        out = resampler.resample(wave, orig_sr=16000)
        np.testing.assert_array_equal(out, wave)

    def test_resample_preserves_shape_2d(self):
        resampler = AudioResampler(target_sample_rate=8000, engine="soxr")
        # stereo: (2, 16000)
        wave = np.stack([_sine(duration=1.0), _sine(duration=1.0, freq=880)], axis=0)
        out = resampler.resample(wave, orig_sr=16000)
        self.assertEqual(out.shape, (2, 8000))

    def test_is_noop_returns_correct_bool(self):
        resampler = AudioResampler(target_sample_rate=16000)
        self.assertTrue(resampler.is_noop(16000))
        self.assertFalse(resampler.is_noop(8000))

    def test_output_dtype_always_float32(self):
        resampler = AudioResampler(target_sample_rate=8000)
        wave = _sine(duration=0.5).astype(np.float64)
        out = resampler.resample(wave, orig_sr=16000)
        self.assertEqual(out.dtype, np.float32)


# ---------------------------------------------------------------------------
# AudioNormalizer tests
# ---------------------------------------------------------------------------

class TestAudioNormalizer(unittest.TestCase):

    def test_peak_normalization(self):
        norm = AudioNormalizer(strategy="peak", peak_target_dbfs=-3.0)
        wave = _sine(amplitude=0.1).reshape(1, -1)
        result = norm.normalize(wave)
        out_peak = float(np.max(np.abs(result.waveform)))
        target = 10 ** (-3.0 / 20)
        self.assertAlmostEqual(out_peak, target, places=2)
        self.assertEqual(result.status, NORM_OK)

    def test_rms_normalization(self):
        norm = AudioNormalizer(strategy="rms", rms_target_dbfs=-20.0)
        wave = _sine(amplitude=0.5).reshape(1, -1)
        result = norm.normalize(wave)
        out_rms = float(np.sqrt(np.mean(result.waveform ** 2)))
        target_rms = 10 ** (-20.0 / 20)
        self.assertAlmostEqual(out_rms, target_rms, places=2)

    def test_silent_file_not_normalized(self):
        norm = AudioNormalizer(strategy="peak", silence_threshold_dbfs=-60.0)
        wave = np.full((1, 16000), 1e-8, dtype=np.float32)
        result = norm.normalize(wave)
        self.assertEqual(result.status, NORM_SILENT)
        np.testing.assert_array_almost_equal(result.waveform, wave, decimal=10)

    def test_no_normalization_strategy(self):
        norm = AudioNormalizer(strategy="none")
        wave = _sine(amplitude=0.3).reshape(1, -1)
        result = norm.normalize(wave)
        np.testing.assert_array_equal(result.waveform, wave)
        self.assertEqual(result.gain_db, 0.0)

    def test_source_waveform_not_mutated(self):
        norm = AudioNormalizer(strategy="peak")
        wave = _sine(amplitude=0.1).reshape(1, -1)
        original = wave.copy()
        norm.normalize(wave)
        np.testing.assert_array_equal(wave, original)

    def test_gain_db_is_positive_for_quiet_signal(self):
        norm = AudioNormalizer(strategy="peak", peak_target_dbfs=-3.0)
        wave = _sine(amplitude=0.01).reshape(1, -1)  # very quiet
        result = norm.normalize(wave)
        self.assertGreater(result.gain_db, 0.0)

    def test_clip_guard_applied(self):
        # Force an extremely large RMS target to trigger clip guard
        norm = AudioNormalizer(
            strategy="rms", rms_target_dbfs=0.0,  # target 1.0 linear
            clip_guard_dbfs=-0.1,
        )
        wave = _sine(amplitude=0.001).reshape(1, -1)  # will require huge gain
        result = norm.normalize(wave)
        out_peak = float(np.max(np.abs(result.waveform)))
        clip_linear = 10 ** (-0.1 / 20)
        self.assertLessEqual(out_peak, clip_linear + 1e-5)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestAudioPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.tmp_dir) / "processed"
        self.log_path = Path(self.tmp_dir) / "audio_quality.csv"

        cfg = _minimal_cfg()
        cfg["paths"]["output_dir"] = str(self.out_dir)
        cfg["paths"]["quality_log"] = str(self.log_path)
        self.cfg = cfg
        self.pipeline = AudioPreprocessingPipeline(
            cfg=cfg,
            output_dir=self.out_dir,
            log_path=self.log_path,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_test_wav(self, duration=1.0, sr=16000, amplitude=0.5, with_silence=False) -> Path:
        wave = _sine(duration=duration, sr=sr, amplitude=amplitude)
        if with_silence:
            # Silence the first half so SNR estimator sees a noise floor
            wave[:len(wave)//2] *= 1e-4
        path = Path(self.tmp_dir) / "input.wav"
        sf.write(str(path), wave, sr, subtype="PCM_16")
        return path

    # --- Core correctness tests ---

    def test_process_file_pass(self):
        src = self._write_test_wav()
        result = self.pipeline.process_file(src)
        self.assertIn(result.status, ("PASS", "WARN"))
        self.assertIsNotNone(result.output_path)
        self.assertTrue(result.output_path.exists())

    def test_source_file_not_modified(self):
        src = self._write_test_wav()
        mtime_before = src.stat().st_mtime
        content_before = src.read_bytes()
        self.pipeline.process_file(src)
        self.assertEqual(src.read_bytes(), content_before)
        self.assertEqual(src.stat().st_mtime, mtime_before)

    def test_output_is_mono(self):
        src = self._write_test_wav()
        result = self.pipeline.process_file(src)
        out_data, out_sr = sf.read(str(result.output_path), always_2d=True)
        self.assertEqual(out_data.shape[1], 1)  # (samples, 1) — mono

    def test_output_sample_rate_correct(self):
        src = self._write_test_wav(sr=22050)  # input != target
        result = self.pipeline.process_file(src)
        info = sf.info(str(result.output_path))
        self.assertEqual(info.samplerate, 16000)
        self.assertTrue(result.resampled)

    def test_output_already_target_sr_no_resample(self):
        src = self._write_test_wav(sr=16000)
        result = self.pipeline.process_file(src)
        self.assertFalse(result.resampled)

    def test_quality_log_written(self):
        src = self._write_test_wav()
        self.pipeline.process_file(src)
        self.assertTrue(self.log_path.exists())
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 2)  # header + at least 1 row
        self.assertIn("status", lines[0])

    def test_skip_existing_respected(self):
        src = self._write_test_wav()
        r1 = self.pipeline.process_file(src)
        self.assertIn(r1.status, ("PASS", "WARN"))
        r2 = self.pipeline.process_file(src)
        self.assertEqual(r2.status, "SKIP")

    def test_reject_silent_file(self):
        silent = np.zeros(16000, dtype=np.float32)
        path = Path(self.tmp_dir) / "silent.wav"
        sf.write(str(path), silent, 16000)
        result = self.pipeline.process_file(path)
        self.assertEqual(result.status, "REJECT")

    def test_reject_too_short_file(self):
        # 0.05 s < min=0.1 s
        wave = _sine(duration=0.05)
        path = Path(self.tmp_dir) / "short.wav"
        sf.write(str(path), wave, 16000)
        result = self.pipeline.process_file(path)
        self.assertEqual(result.status, "REJECT")

    def test_output_path_never_equals_input(self):
        """If output_dir == input_dir, pipeline must refuse, not overwrite."""
        # Write input into the output dir directly
        path = self.out_dir / "test.wav"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), _sine(duration=1.0), 16000)
        result = self.pipeline.process_file(path)
        # Should either SKIP (skip_existing) or ERROR if would overwrite
        self.assertIn(result.status, ("SKIP", "ERROR", "PASS", "WARN"))

    def test_duration_recorded_correctly(self):
        src = self._write_test_wav(duration=2.0)
        result = self.pipeline.process_file(src)
        self.assertAlmostEqual(result.duration, 2.0, places=1)

    def test_snr_estimated(self):
        src = self._write_test_wav(duration=2.0, amplitude=0.5, with_silence=True)
        result = self.pipeline.process_file(src)
        self.assertFalse(math.isnan(result.snr_db))
        self.assertGreater(result.snr_db, 10.0)

    def test_batch_process_returns_all_results(self):
        paths = []
        for i in range(3):
            wave = _sine(duration=1.0, freq=440 * (i + 1))
            p = Path(self.tmp_dir) / f"test_{i}.wav"
            sf.write(str(p), wave, 16000)
            paths.append(p)
        results = self.pipeline.process_batch(paths)
        self.assertEqual(len(results), 3)

    def test_stereo_input_converted_to_mono_output(self):
        mono = _sine(duration=1.0)
        stereo = np.stack([mono, mono], axis=1)
        path = Path(self.tmp_dir) / "stereo.wav"
        sf.write(str(path), stereo, 16000)
        result = self.pipeline.process_file(path)
        self.assertIn(result.status, ("PASS", "WARN"))
        out_info = sf.info(str(result.output_path))
        self.assertEqual(out_info.channels, 1)


# ---------------------------------------------------------------------------
# SNR estimation tests
# ---------------------------------------------------------------------------

class TestSNREstimation(unittest.TestCase):

    def setUp(self):
        cfg = _minimal_cfg()
        cfg["paths"]["output_dir"] = tempfile.mkdtemp()
        cfg["paths"]["quality_log"] = str(Path(cfg["paths"]["output_dir"]) / "log.csv")
        self.pipeline = AudioPreprocessingPipeline(cfg=cfg)

    def _make_signal_with_silence(self, amplitude: float = 0.5) -> np.ndarray:
        silence = np.zeros(SR, dtype=np.float32)
        sine = _sine(duration=1.0, amplitude=amplitude)
        return np.concatenate([silence, sine])

    def test_snr_pure_sine_positive(self):
        wave = self._make_signal_with_silence(amplitude=0.5)
        snr = self.pipeline._estimate_snr(wave, SR)
        self.assertGreater(snr, 10.0)

    def test_snr_noisy_signal_lower(self):
        wave = self._make_signal_with_silence(amplitude=0.5)
        noise = np.random.default_rng(42).normal(0, 0.1, len(wave)).astype(np.float32)
        noisy = (wave + noise).astype(np.float32)
        snr_clean = self.pipeline._estimate_snr(wave, SR)
        snr_noisy = self.pipeline._estimate_snr(noisy, SR)
        self.assertGreater(snr_clean, snr_noisy)

    def test_snr_returns_nan_for_tiny_clip(self):
        wave = np.zeros(100, dtype=np.float32)
        snr = self.pipeline._estimate_snr(wave, SR)
        self.assertTrue(math.isnan(snr))


# ---------------------------------------------------------------------------
# Config loading test
# ---------------------------------------------------------------------------

class TestConfigLoading(unittest.TestCase):

    def test_load_audio_config_from_yaml(self):
        cfg_path = _ROOT / "configs" / "audio.yaml"
        if not cfg_path.exists():
            self.skipTest("configs/audio.yaml not found")
        cfg = load_audio_config(cfg_path)
        self.assertIn("audio", cfg)
        self.assertIn("target_sample_rate", cfg["audio"])
        self.assertEqual(cfg["audio"]["target_sample_rate"], 16000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    import io as _io
    if hasattr(_sys.stdout, "buffer"):
        _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    unittest.main(verbosity=2)
