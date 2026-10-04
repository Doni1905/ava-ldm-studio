"""
src/data/metadata_selector.py
==============================
Column selection, quality filtering, and dataset statistics.

Responsibilities
----------------
* Keep only the 12 required columns (+ internal tracking columns).
* Drop the ~12 columns not needed by AVA LDM.
* Apply quality filters (duration bounds, SNR floor, non-empty text).
* Compute and log dataset statistics.

Required columns (from configs/dataset.yaml → dataset.required_columns):
    audio, verbatim, normalized, speaker_id, district, scenario,
    task_name, gender, age_group, area, duration, snr

Note: ``audio`` bytes are never in the DataFrame — only ``audio_hf_path``
(the string HF URI) is retained as the audio reference.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class MetadataSelector:
    """
    Applies column selection and quality filters to the raw streaming DataFrame.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration from configs/dataset.yaml.
    """

    # Columns always kept for internal pipeline tracking
    _INTERNAL_COLS = {"audio_hf_path", "shard_idx", "row_within_shard"}

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ds_cfg = cfg["dataset"]
        self.sel_cfg = cfg["selection"]

        # Required columns for the AVA working dataset
        self.required: list[str] = self.ds_cfg["required_columns"]
        # Remove 'audio' from required — we use 'audio_hf_path' instead
        self.keep_cols: set[str] = (
            set(self.required) - {"audio"} | self._INTERNAL_COLS
        )

        # Quality filter params
        filt = self.sel_cfg.get("filters", {})
        self.min_duration: float = filt.get("min_duration", 1.0)
        self.max_duration: float = filt.get("max_duration", 30.0)
        self.min_snr: float = filt.get("min_snr", 5.0)
        self.require_verbatim: bool = filt.get("require_verbatim", True)
        self.require_normalized: bool = filt.get("require_normalized", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column selection and all quality filters to *df*.

        Steps
        -----
        1. Select only required + internal columns (drop everything else).
        2. Coerce numeric columns to float.
        3. Filter by duration bounds.
        4. Filter by SNR floor.
        5. Filter by non-empty verbatim / normalized text.
        6. Drop exact duplicates on (speaker_id, verbatim).

        Returns
        -------
        pd.DataFrame  (copy, original untouched)
        """
        if df.empty:
            logger.warning("select_and_filter received an empty DataFrame.")
            return df.copy()

        n0 = len(df)
        logger.info("MetadataSelector — input rows: %d", n0)

        df = df.copy()

        # Step 1: Column selection
        df = self._select_columns(df)

        # Step 2: Coerce numeric types
        df = self._coerce_numerics(df)

        # Step 3: Duration filter
        df = self._filter_duration(df)
        logger.info("After duration filter [%.1f–%.1f s]: %d rows", self.min_duration, self.max_duration, len(df))

        # Step 4: SNR filter
        df = self._filter_snr(df)
        logger.info("After SNR filter [>= %.1f dB]: %d rows", self.min_snr, len(df))

        # Step 5: Text completeness
        df = self._filter_text(df)
        logger.info("After text completeness filter: %d rows", len(df))

        # Step 6: Deduplicate on (speaker_id, verbatim)
        df = self._deduplicate(df)
        logger.info("After deduplication: %d rows", len(df))

        logger.info(
            "MetadataSelector — retained %d / %d rows (%.1f%%)",
            len(df), n0, 100 * len(df) / max(n0, 1),
        )
        return df.reset_index(drop=True)

    def compute_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute and log summary statistics for a manifest DataFrame.

        Returns a dictionary suitable for JSON serialisation.
        """
        stats: dict[str, Any] = {"total_rows": len(df)}

        for col in ("gender", "age_group", "area", "scenario", "district"):
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                stats[f"{col}_distribution"] = counts

        for col in ("duration", "snr"):
            if col in df.columns:
                stats[f"{col}_stats"] = {
                    "mean":   round(float(df[col].mean()), 3),
                    "median": round(float(df[col].median()), 3),
                    "std":    round(float(df[col].std()), 3),
                    "min":    round(float(df[col].min()), 3),
                    "max":    round(float(df[col].max()), 3),
                }

        if "speaker_id" in df.columns:
            stats["unique_speakers"] = int(df["speaker_id"].nunique())

        if "task_name" in df.columns:
            stats["unique_tasks"] = int(df["task_name"].nunique())
            stats["task_distribution"] = df["task_name"].value_counts().head(20).to_dict()

        self._log_statistics(stats)
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only required + internal columns, warn about missing ones."""
        available = set(df.columns)
        desired = self.keep_cols

        present = desired & available
        missing = desired - available - self._INTERNAL_COLS
        if missing:
            logger.warning(
                "Required columns not found in DataFrame (will be absent in manifest): %s",
                sorted(missing),
            )

        to_drop = available - present
        if to_drop:
            logger.debug("Dropping columns: %s", sorted(to_drop))

        return df[sorted(present)]

    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("duration", "snr"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _filter_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        if "duration" not in df.columns:
            logger.warning("'duration' column missing — skipping duration filter.")
            return df
        mask = (
            df["duration"].notna()
            & (df["duration"] >= self.min_duration)
            & (df["duration"] <= self.max_duration)
        )
        return df[mask]

    def _filter_snr(self, df: pd.DataFrame) -> pd.DataFrame:
        if "snr" not in df.columns:
            logger.warning("'snr' column missing — skipping SNR filter.")
            return df
        mask = df["snr"].notna() & (df["snr"] >= self.min_snr)
        return df[mask]

    def _filter_text(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series([True] * len(df), index=df.index)
        if self.require_verbatim and "verbatim" in df.columns:
            mask &= df["verbatim"].notna() & (df["verbatim"].str.strip() != "")
        if self.require_normalized and "normalized" in df.columns:
            mask &= df["normalized"].notna() & (df["normalized"].str.strip() != "")
        return df[mask]

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        dedup_cols = [c for c in ("speaker_id", "verbatim") if c in df.columns]
        if not dedup_cols:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        dropped = before - len(df)
        if dropped:
            logger.info("Deduplication removed %d exact duplicate rows.", dropped)
        return df

    def _log_statistics(self, stats: dict[str, Any]) -> None:
        logger.info("=== Dataset Statistics ===")
        logger.info("  Total rows   : %d", stats.get("total_rows", 0))
        logger.info("  Unique speakers: %d", stats.get("unique_speakers", 0))
        logger.info("  Unique tasks : %d", stats.get("unique_tasks", 0))
        for col in ("gender", "age_group", "area", "scenario"):
            dist = stats.get(f"{col}_distribution", {})
            if dist:
                logger.info("  %s: %s", col, dist)
        for col in ("duration", "snr"):
            s = stats.get(f"{col}_stats", {})
            if s:
                logger.info(
                    "  %s: mean=%.2f  median=%.2f  [%.2f–%.2f]",
                    col, s["mean"], s["median"], s["min"], s["max"],
                )
