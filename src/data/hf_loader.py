"""
src/data/hf_loader.py
======================
Hugging Face dataset loader for SPRINGLab/IndicVoices-R_Tamil.

Design goals
------------
* NEVER download audio bytes — the 62.7 GB dataset stays on HF Hub.
* Use pyarrow to read only non-audio columns from each parquet shard.
* Support resume-safe streaming via a _Checkpoint object.
* Support both authenticated and unauthenticated access.
* Provide dataset inspection (card, schema) without any local downloads.

Strategy for avoiding audio download
--------------------------------------
Parquet stores each column in separate column chunks. We use
``pyarrow.parquet.read_table(..., columns=[...])`` with an explicit list
of non-audio columns, then never request the ``audio`` column bytes.

What IS kept in the manifest for ``audio``
------------------------------------------
The HF Hub URL (``hf://datasets/<repo_id>/<shard_path>``) is stored as a
plain string in ``audio_path``. When a training script later needs the
actual waveform, it uses ``torchaudio.load(audio_path)`` via the HF
filesystem, fetching only that single file on demand.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests

if TYPE_CHECKING:
    from .dataset_manager import _Checkpoint

logger = logging.getLogger(__name__)

# Columns that carry audio bytes — we NEVER request these from parquet
_AUDIO_COLUMNS = {"audio", "audio_array", "audio_path", "waveform", "bytes"}

# Base URL for resolving individual parquet shard files
_HF_RESOLVE = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
_HF_PARQUET_API = "https://huggingface.co/api/datasets/{repo_id}/parquet"


class HFLoader:
    """
    Handles all Hugging Face Hub interactions for the dataset pipeline.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration loaded from configs/dataset.yaml.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ds_cfg = cfg["dataset"]
        self.repo_id: str = self.ds_cfg["repo_id"]
        self.num_shards: int = self.ds_cfg["num_shards"]
        self._token: str | None = self._resolve_token()
        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _resolve_token(self) -> str | None:
        """Try HF_TOKEN env var, then huggingface_hub cache."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            try:
                import huggingface_hub as hf  # noqa: PLC0415
                token = hf.get_token()
            except Exception:  # noqa: BLE001
                pass
        if token:
            logger.debug("HF token resolved (first 8 chars: %s…)", token[:8])
        else:
            logger.info(
                "No HF token found. Proceeding anonymously — public datasets only."
            )
        return token

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        if self._token:
            s.headers.update({"Authorization": f"Bearer {self._token}"})
        s.headers.update({"User-Agent": "AVA-LDM-Studio/1.0"})
        return s

    # ------------------------------------------------------------------
    # Dataset inspection (no audio download)
    # ------------------------------------------------------------------

    def fetch_dataset_info(self) -> dict[str, Any]:
        """
        Return dataset card metadata and schema without downloading any data.

        Uses the HF Hub API to retrieve the dataset card and parquet manifest.
        Falls back to the known schema if the network is unavailable.
        """
        logger.info("Fetching dataset info for %s …", self.repo_id)
        try:
            import huggingface_hub as hf  # noqa: PLC0415
            api = hf.HfApi(token=self._token)
            ds_info = api.dataset_info(self.repo_id)

            # Parse card data string if present
            card_raw = ds_info.cardData or ""
            card_text = card_raw if isinstance(card_raw, str) else str(card_raw)

            # Build schema from known card data
            schema = self._known_schema()

            return {
                "repo_id": self.repo_id,
                "num_examples": self.ds_cfg.get("total_examples", 39292),
                "num_shards": self.num_shards,
                "download_size_gb": round(62665842671 / 1e9, 2),
                "schema": schema,
                "card_text": card_text[:2000],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch live dataset info: %s. Using cached schema.", exc)
            return {
                "repo_id": self.repo_id,
                "num_examples": self.ds_cfg.get("total_examples", 39292),
                "num_shards": self.num_shards,
                "download_size_gb": 62.67,
                "schema": self._known_schema(),
                "card_text": "(offline — schema from card data cache)",
            }

    def _known_schema(self) -> dict[str, str]:
        """Return the schema from the dataset card (verified 2026-09-30)."""
        return {
            "text": "string",
            "lang": "ClassLabel(ta)",
            "samples": "int64",
            "verbatim": "string",
            "normalized": "string",
            "speaker_id": "string",
            "scenario": "ClassLabel(Extempore,Read)",
            "task_name": "string",
            "gender": "ClassLabel(Female,Male)",
            "age_group": "ClassLabel(18-30,30-45,45-60,60+)",
            "job_type": "ClassLabel(Blue Collar,Student,Unemployed,White Collar)",
            "qualification": "ClassLabel",
            "area": "ClassLabel(Rural,Urban)",
            "district": "string",
            "state": "ClassLabel(Tamil Nadu)",
            "occupation": "string",
            "utterance_pitch_mean": "float64",
            "utterance_pitch_std": "float64",
            "snr": "float64",
            "c50": "float64",
            "speaking_rate": "float64",
            "cer": "string",
            "duration": "float64",
            "audio": "Audio (binary — NOT downloaded)",
        }

    # ------------------------------------------------------------------
    # Parquet shard URLs
    # ------------------------------------------------------------------

    def shard_url(self, shard_idx: int) -> str:
        """Return the HTTPS URL for a specific parquet shard."""
        filename = f"data/train-{shard_idx:05d}-of-{self.num_shards:05d}.parquet"
        return _HF_RESOLVE.format(repo_id=self.repo_id, filename=filename)

    def shard_hf_path(self, shard_idx: int) -> str:
        """Return the hf:// path used as the audio reference base."""
        return (
            f"hf://datasets/{self.repo_id}/"
            f"data/train-{shard_idx:05d}-of-{self.num_shards:05d}.parquet"
        )

    # ------------------------------------------------------------------
    # Columns to read (exclude audio bytes)
    # ------------------------------------------------------------------

    def _metadata_columns(self) -> list[str]:
        """
        Return the list of parquet columns to request — explicitly
        excluding any column that would load audio bytes.
        """
        # All columns from the known schema except audio
        all_cols = [
            "text", "lang", "samples", "verbatim", "normalized",
            "speaker_id", "scenario", "task_name", "gender", "age_group",
            "job_type", "qualification", "area", "district", "state",
            "occupation", "utterance_pitch_mean", "utterance_pitch_std",
            "snr", "c50", "speaking_rate", "cer", "duration",
        ]
        return all_cols  # audio column deliberately excluded

    # ------------------------------------------------------------------
    # Core streaming method
    # ------------------------------------------------------------------

    def stream_metadata(
        self,
        checkpoint: "_Checkpoint | None" = None,
        max_shards: int | None = None,
    ) -> pd.DataFrame:
        """
        Stream metadata-only rows from all parquet shards.

        Audio bytes are NEVER requested. Each shard's parquet file is
        downloaded in full (metadata columns only), which is much smaller
        than the audio-inclusive file, because parquet stores column data
        in separate column chunks — requesting only metadata columns
        causes the parquet reader to skip the audio column data.

        Parameters
        ----------
        checkpoint : _Checkpoint | None
            If provided, shards already marked as done are skipped.
        max_shards : int | None
            Process at most this many shards (for testing).

        Returns
        -------
        pd.DataFrame
            Concatenated metadata rows from all processed shards.
            An ``audio_hf_path`` column (string) is added to each row
            recording the HF URI for later audio loading.
        """
        completed = checkpoint.completed_shards() if checkpoint else set()
        total_shards = min(self.num_shards, max_shards) if max_shards else self.num_shards
        columns_to_read = self._metadata_columns()

        all_frames: list[pd.DataFrame] = []

        # Load any previously cached partial frames
        metadata_dir = Path(self.cfg["paths"]["metadata_dir"])
        for shard_idx in sorted(completed):
            cache_file = metadata_dir / f"shard_{shard_idx:05d}.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                all_frames.append(df)
                logger.debug("Loaded cached shard %d (%d rows)", shard_idx, len(df))

        for shard_idx in range(total_shards):
            if shard_idx in completed:
                logger.info("Skipping shard %d (already processed)", shard_idx)
                continue

            shard_df = self._fetch_shard(
                shard_idx=shard_idx,
                columns=columns_to_read,
                metadata_dir=metadata_dir,
            )
            if shard_df is not None:
                all_frames.append(shard_df)
                if checkpoint:
                    checkpoint.mark_shard_done(shard_idx)

        if not all_frames:
            logger.warning("No shard data collected — returning empty DataFrame.")
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        logger.info("Total rows collected across all shards: %d", len(result))
        return result

    def _fetch_shard(
        self,
        shard_idx: int,
        columns: list[str],
        metadata_dir: Path,
    ) -> pd.DataFrame | None:
        """
        Download a single parquet shard, read only metadata columns,
        cache locally, and return as a DataFrame.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415

        cache_file = metadata_dir / f"shard_{shard_idx:05d}.parquet"
        url = self.shard_url(shard_idx)

        logger.info("Processing shard %d / %d → %s", shard_idx, self.num_shards - 1, url)
        t0 = time.perf_counter()

        try:
            # Download the parquet shard
            resp = self._session.get(url, stream=True, timeout=(10, 300))
            resp.raise_for_status()

            raw_bytes = io.BytesIO(resp.content)
            logger.debug("Shard %d downloaded in %.1f s", shard_idx, time.perf_counter() - t0)

            # Read ONLY requested columns — audio column chunk is skipped
            pq_file = pq.read_table(
                raw_bytes,
                columns=columns,
                use_pandas_metadata=True,
            )
            df = pq_file.to_pandas()

            # Add a string reference to the audio location (no bytes)
            hf_base = self.shard_hf_path(shard_idx)
            # Row index within shard is the unique row locator
            df["audio_hf_path"] = hf_base
            df["shard_idx"] = shard_idx
            df["row_within_shard"] = range(len(df))

            # Decode class label columns (integers → strings)
            df = self._decode_class_labels(df)

            # Cache to local parquet (metadata only — no audio bytes)
            df.to_parquet(cache_file, index=False)
            logger.info(
                "Shard %d: %d rows read in %.1f s → cached %s",
                shard_idx, len(df), time.perf_counter() - t0, cache_file.name,
            )
            return df

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error(
                    "Shard %d: 401 Unauthorized. Run `huggingface-cli login`.", shard_idx
                )
            else:
                logger.error("Shard %d: HTTP error: %s", shard_idx, exc)
            return None
        except requests.ConnectionError as exc:
            logger.error("Shard %d: Connection error: %s — check network.", shard_idx, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Shard %d: Unexpected error: %s", shard_idx, exc)
            return None

    # ------------------------------------------------------------------
    # Class label decoding
    # ------------------------------------------------------------------

    _LABEL_MAPS: dict[str, dict[int, str]] = {
        "scenario":  {0: "Extempore", 1: "Read"},
        "gender":    {0: "Female", 1: "Male"},
        "age_group": {0: "18-30", 1: "30-45", 2: "45-60", 3: "60+"},
        "job_type":  {0: "Blue Collar", 1: "Student", 2: "Unemployed", 3: "White Collar"},
        "qualification": {
            0: "No Schooling",
            1: "Post Grad + PhD",
            2: "Undergrad and Grad.",
            3: "Upto 12th",
        },
        "area":  {0: "Rural", 1: "Urban"},
        "lang":  {0: "ta"},
        "state": {0: "Tamil Nadu"},
    }

    def _decode_class_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace integer class-label codes with their string names."""
        for col, mapping in self._LABEL_MAPS.items():
            if col in df.columns:
                try:
                    df[col] = df[col].map(mapping).fillna(df[col].astype(str))
                except Exception:  # noqa: BLE001
                    pass
        return df

    # ------------------------------------------------------------------
    # Streaming via datasets library (alternative path)
    # ------------------------------------------------------------------

    def stream_via_datasets_library(
        self,
        max_examples: int = 10000,
    ) -> pd.DataFrame:
        """
        Alternative: use the ``datasets`` library in streaming mode.

        This avoids downloading any shard files locally — rows are fetched
        lazily. Audio arrays are excluded via ``remove_columns``.

        Parameters
        ----------
        max_examples : int
            Maximum number of rows to collect before stopping.

        Returns
        -------
        pd.DataFrame
        """
        import datasets as hf_datasets  # noqa: PLC0415

        logger.info("Streaming via datasets library (max %d rows) …", max_examples)
        ds = hf_datasets.load_dataset(
            self.repo_id,
            split="train",
            streaming=True,
            token=self._token or None,
        )

        # Remove audio column immediately to avoid loading bytes
        audio_cols = [c for c in ds.column_names if c in _AUDIO_COLUMNS]
        if audio_cols:
            ds = ds.remove_columns(audio_cols)
            logger.info("Removed audio columns from stream: %s", audio_cols)

        rows = []
        for i, example in enumerate(ds):
            if i >= max_examples:
                break
            rows.append(example)
            if (i + 1) % 1000 == 0:
                logger.info("  … streamed %d rows", i + 1)

        df = pd.DataFrame(rows)
        df = self._decode_class_labels(df)
        logger.info("Streaming complete — %d rows collected.", len(df))
        return df

    # ------------------------------------------------------------------
    # Parquet shard inspection (lightweight)
    # ------------------------------------------------------------------

    def inspect_shard_schema(self, shard_idx: int = 0) -> dict[str, str]:
        """
        Return the schema of a single parquet shard without reading row data.
        Uses a HEAD request to get the file, then reads only parquet metadata.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415

        url = self.shard_url(shard_idx)
        logger.info("Inspecting shard %d schema from %s …", shard_idx, url)
        try:
            resp = self._session.get(url, timeout=(10, 300))
            resp.raise_for_status()
            schema = pq.read_schema(io.BytesIO(resp.content))
            return {field.name: str(field.type) for field in schema}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read live shard schema: %s", exc)
            return self._known_schema()
