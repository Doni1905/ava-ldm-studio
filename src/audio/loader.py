"""
src/audio/loader.py
====================
Audio file loader for AVA LDM Studio.

Design
------
* Primary I/O engine: ``soundfile`` (libsndfile backend — no torchcodec needed).
* Always returns ``numpy.ndarray`` in float32, shape ``(channels, samples)``.
* Preserves original file — never writes to the source path.
* Detects corrupt files via format validation before full decode.
* Memory-efficient: reads directly into float32 (no intermediate int conversion).

Supported formats (via libsndfile)
------------------------------------
WAV, FLAC, OGG, AIFF, W64, RF64, and all PCM subtypes.

Returns
-------
``AudioData`` dataclass with:
    waveform    : np.ndarray  shape (channels, samples), float32
    sample_rate : int
    n_channels  : int
    n_samples   : int
    duration    : float       seconds
    file_path   : Path
    format_info : soundfile.info object
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif", ".w64", ".rf64",
    ".mp3",  # via libsndfile if compiled with mp3 support
}


@dataclass
class AudioData:
    """Container for a loaded audio file."""
    waveform: np.ndarray        # shape: (channels, samples), float32
    sample_rate: int
    n_channels: int
    n_samples: int
    duration: float             # seconds
    file_path: Path
    format_info: Any = field(repr=False, default=None)

    @property
    def is_mono(self) -> bool:
        return self.n_channels == 1

    @property
    def waveform_mono(self) -> np.ndarray:
        """Return mono waveform (1-D array). Averages channels if stereo."""
        if self.n_channels == 1:
            return self.waveform[0]
        return self.waveform.mean(axis=0)


class AudioLoadError(Exception):
    """Raised when a file cannot be loaded."""


class AudioLoader:
    """
    Load audio files into float32 numpy arrays via soundfile.

    Parameters
    ----------
    max_channels : int
        Reject files with more than this many channels.
    """

    def __init__(self, max_channels: int = 8) -> None:
        self.max_channels = max_channels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> AudioData:
        """
        Load an audio file and return an ``AudioData`` object.

        Raises ``AudioLoadError`` for:
        - Missing / empty files
        - Unsupported format or codec
        - Too many channels
        - Zero-length audio

        Parameters
        ----------
        file_path : str | Path

        Returns
        -------
        AudioData
        """
        path = Path(file_path)
        self._check_path(path)

        try:
            info = sf.info(str(path))
        except Exception as exc:
            raise AudioLoadError(
                f"Cannot read audio metadata from '{path}': {exc}"
            ) from exc

        self._check_info(info, path)

        try:
            # always_2d=True → shape (samples, channels)
            data, sr = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise AudioLoadError(
                f"Failed to decode audio from '{path}': {exc}"
            ) from exc

        # Transpose to (channels, samples)
        waveform = data.T.astype(np.float32, copy=False)

        n_channels, n_samples = waveform.shape
        duration = n_samples / sr if sr > 0 else 0.0

        if n_samples == 0:
            raise AudioLoadError(f"Zero-length audio in '{path}'")

        logger.debug(
            "Loaded '%s': %d ch, %d Hz, %.3f s, %s",
            path.name, n_channels, sr, duration, info.subtype,
        )

        return AudioData(
            waveform=waveform,
            sample_rate=sr,
            n_channels=n_channels,
            n_samples=n_samples,
            duration=duration,
            file_path=path,
            format_info=info,
        )

    def probe(self, file_path: str | Path) -> dict[str, Any]:
        """
        Return metadata without decoding audio bytes.
        Fast and memory-efficient — reads only the file header.
        """
        path = Path(file_path)
        try:
            info = sf.info(str(path))
            return {
                "file": str(path),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration": info.frames / info.samplerate if info.samplerate else 0.0,
                "format": info.format,
                "subtype": info.subtype,
            }
        except Exception as exc:
            return {
                "file": str(path),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_path(self, path: Path) -> None:
        if not path.exists():
            raise AudioLoadError(f"File not found: '{path}'")
        if not path.is_file():
            raise AudioLoadError(f"Not a file: '{path}'")
        if path.stat().st_size == 0:
            raise AudioLoadError(f"File is empty (0 bytes): '{path}'")
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            logger.warning("Extension '%s' not in known list; attempting load anyway.", ext)

    def _check_info(self, info: Any, path: Path) -> None:
        if info.channels > self.max_channels:
            raise AudioLoadError(
                f"Too many channels ({info.channels} > {self.max_channels}) in '{path}'"
            )
        if info.samplerate <= 0:
            raise AudioLoadError(f"Invalid sample rate {info.samplerate} in '{path}'")
        if info.frames == 0:
            raise AudioLoadError(f"Zero frames reported for '{path}'")

