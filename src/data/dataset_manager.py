"""
src/data/dataset_manager.py
============================
High-level orchestrator for the AVA LDM dataset pipeline.

Responsibilities
----------------
* Load and validate the pipeline configuration (configs/dataset.yaml).
* Expose a clean API used by all CLI scripts.
* Route calls to hf_loader, metadata_selector, sampler, and splitter.
* Manage the resume-safe checkpoint so interrupted runs can restart
  from the last successfully processed shard.
* Emit structured log entries at every stage boundary.

Usage
-----
    from src.data.dataset_manager import DatasetManager

    dm = DatasetManager()                      # loads configs/dataset.yaml
    dm.inspect()                               # print dataset card & schema
    manifest = dm.build_manifest()             # stream + filter + stratify
    splits = dm.create_splits(manifest)        # speaker-disjoint splits
"""

from __future__ import annotations

import io as _io
import sys as _sys

# Force UTF-8 stdout/stderr so Unicode characters print on Windows without errors.
if hasattr(_sys.stdout, "buffer"):
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "buffer"):
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .hf_loader import HFLoader
from .metadata_selector import MetadataSelector
from .sampler import StratifiedSampler
from .splitter import SpeakerDisjointSplitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Run from the project root or pass the correct path."
        )
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Basic validation
    required_sections = {"dataset", "selection", "splitting", "paths", "logging"}
    missing = required_sections - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    return cfg


def _setup_logging(cfg: dict[str, Any], verbose: bool = False) -> None:
    """Configure root logger from config."""
    log_cfg = cfg.get("logging", {})
    level_name = "DEBUG" if verbose else log_cfg.get("level", "INFO")
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / Path(log_cfg.get("log_file", "logs/dataset_pipeline.log")).name

    from logging.handlers import RotatingFileHandler
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        RotatingFileHandler(
            log_file,
            maxBytes=log_cfg.get("max_bytes", 10 * 1024 * 1024),
            backupCount=log_cfg.get("backup_count", 3),
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume-safe)
# ---------------------------------------------------------------------------

class _Checkpoint:
    """Persists pipeline progress so runs can be resumed after interruption."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning("Checkpoint file corrupt — starting fresh.")
        return {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def mark_shard_done(self, shard_idx: int) -> None:
        done = set(self._data.get("completed_shards", []))
        done.add(shard_idx)
        self._data["completed_shards"] = sorted(done)
        self._save()

    def completed_shards(self) -> set[int]:
        return set(self._data.get("completed_shards", []))

    def reset(self) -> None:
        self._data = {}
        if self.path.exists():
            self.path.unlink()
        logger.info("Checkpoint reset.")


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    Orchestrates the entire dataset pipeline for AVA LDM Studio.

    Parameters
    ----------
    config_path : str | Path
        Path to configs/dataset.yaml (default: auto-discovered from cwd).
    verbose : bool
        Enable DEBUG-level logging.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        if config_path is None:
            # Walk up from cwd looking for configs/dataset.yaml
            for parent in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
                candidate = parent / "configs" / "dataset.yaml"
                if candidate.exists():
                    config_path = candidate
                    break
            if config_path is None:
                config_path = Path("configs/dataset.yaml")

        self.cfg = _load_config(config_path)
        _setup_logging(self.cfg, verbose=verbose)
        logger.info("DatasetManager initialised. Config: %s", config_path)

        self._ensure_dirs()

        self.loader = HFLoader(self.cfg)
        self.selector = MetadataSelector(self.cfg)
        self.sampler = StratifiedSampler(self.cfg)
        self.splitter = SpeakerDisjointSplitter(self.cfg)
        self.checkpoint = _Checkpoint(
            Path(self.cfg["paths"]["checkpoint_file"])
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        paths = self.cfg["paths"]
        for key in ("cache_dir", "shard_dir", "metadata_dir", "manifest_dir", "log_dir"):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        """
        Print and return dataset card information and schema.
        Does NOT download audio data.
        """
        logger.info("=== INSPECT MODE — no audio will be downloaded ===")
        info = self.loader.fetch_dataset_info()
        self._print_inspect(info)
        return info

    def build_manifest(
        self,
        force: bool = False,
        max_shards: int | None = None,
    ) -> pd.DataFrame:
        """
        Stream metadata from all parquet shards (skipping audio bytes),
        apply quality filters, run stratified sampling, and write
        data/manifests/selected_manifest.csv.

        Parameters
        ----------
        force : bool
            If True, ignore any existing checkpoint and reprocess from scratch.
        max_shards : int | None
            Limit processing to the first N shards (useful for testing).

        Returns
        -------
        pd.DataFrame
            The selected 5 000-row manifest (no audio bytes).
        """
        if force:
            self.checkpoint.reset()

        manifest_path = Path(self.cfg["paths"]["selected_manifest"])
        if manifest_path.exists() and not force:
            logger.info(
                "Manifest already exists at %s. Use force=True to rebuild.", manifest_path
            )
            return pd.read_csv(manifest_path)

        t0 = time.perf_counter()
        logger.info("=== BUILD MANIFEST ===")

        # Step 1: Stream metadata from shards (no audio bytes)
        raw_df = self.loader.stream_metadata(
            checkpoint=self.checkpoint,
            max_shards=max_shards,
        )
        logger.info("Raw rows collected from shards: %d", len(raw_df))

        # Step 2: Select required columns, apply quality filters
        filtered_df = self.selector.select_and_filter(raw_df)
        logger.info("Rows after column selection & quality filtering: %d", len(filtered_df))

        # Step 3: Stratified sample → target N
        selected_df = self.sampler.sample(filtered_df)
        logger.info("Rows after stratified sampling: %d", len(selected_df))

        # Step 4: Write manifest
        selected_df.to_csv(manifest_path, index=False)
        logger.info("Manifest written → %s", manifest_path)

        elapsed = time.perf_counter() - t0
        logger.info("build_manifest completed in %.1f s", elapsed)

        # Persist stats to checkpoint
        self.checkpoint.set("manifest_rows", len(selected_df))
        self.checkpoint.set("manifest_path", str(manifest_path))

        return selected_df

    def create_splits(
        self,
        manifest: pd.DataFrame | None = None,
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Create speaker-disjoint train / validation / test splits.

        Parameters
        ----------
        manifest : pd.DataFrame | None
            If None, loads data/manifests/selected_manifest.csv.
        force : bool
            Overwrite existing split files.

        Returns
        -------
        dict with keys 'train', 'val', 'test', each a pd.DataFrame.
        """
        train_path = Path(self.cfg["paths"]["train_manifest"])
        val_path   = Path(self.cfg["paths"]["val_manifest"])
        test_path  = Path(self.cfg["paths"]["test_manifest"])

        if all(p.exists() for p in (train_path, val_path, test_path)) and not force:
            logger.info("Split files already exist. Use force=True to rebuild.")
            return {
                "train": pd.read_csv(train_path),
                "val":   pd.read_csv(val_path),
                "test":  pd.read_csv(test_path),
            }

        if manifest is None:
            manifest_path = Path(self.cfg["paths"]["selected_manifest"])
            if not manifest_path.exists():
                raise FileNotFoundError(
                    "No manifest found. Run build_manifest() first."
                )
            manifest = pd.read_csv(manifest_path)

        logger.info("=== CREATE SPLITS ===")
        splits = self.splitter.split(manifest)

        splits["train"].to_csv(train_path, index=False)
        splits["val"].to_csv(val_path, index=False)
        splits["test"].to_csv(test_path, index=False)

        logger.info(
            "Splits written — train: %d | val: %d | test: %d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )

        # Verify speaker disjointness
        self.splitter.verify_disjoint(splits)
        self.checkpoint.set("splits_created", True)

        return splits

    def get_statistics(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return summary statistics for a manifest DataFrame."""
        if df is None:
            path = Path(self.cfg["paths"]["selected_manifest"])
            if not path.exists():
                raise FileNotFoundError("No manifest. Run build_manifest() first.")
            df = pd.read_csv(path)
        return self.selector.compute_statistics(df)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _print_inspect(self, info: dict[str, Any]) -> None:
        sep = "=" * 60
        print(sep)
        print(f"  Dataset : {info.get('repo_id', 'N/A')}")
        print(f"  Examples: {info.get('num_examples', 'N/A'):,}")
        print(f"  Shards  : {info.get('num_shards', 'N/A')}")
        print(f"  Size    : {info.get('download_size_gb', 'N/A')} GB")
        print(sep)
        print("  Schema (name -> type):")
        for col, dtype in info.get("schema", {}).items():
            marker = "  ★" if col in self.cfg["dataset"]["required_columns"] else "   "
            print(f"{marker}  {col:<30} {dtype}")
        print(sep)
        print("  ★ = included in AVA working dataset")
