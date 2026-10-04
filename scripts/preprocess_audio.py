#!/usr/bin/env python3
"""
scripts/preprocess_audio.py
============================
CLI entry point for the AVA LDM audio preprocessing pipeline.

What this script does
---------------------
1. Reads a manifest CSV (data/manifests/train.csv etc.) or a directory
   of audio files.
2. Runs each audio file through:
      Load → Validate → Mono → Resample → Normalize → Save
3. Writes processed WAV files to data/processed_audio/ (NEVER overwrites
   source audio).
4. Writes data/logs/audio_quality.csv with per-file quality metrics.

Source audio guarantee
----------------------
Original audio files are NEVER modified.
Processed audio is always written to a separate output directory.

Usage
-----
    # Process a manifest (local audio column):
    python scripts/preprocess_audio.py --manifest data/manifests/train.csv

    # Process a single file:
    python scripts/preprocess_audio.py --file path/to/audio.wav

    # Process a directory of WAV files:
    python scripts/preprocess_audio.py --dir path/to/wavs/

    # Dry run (validate only, no output written):
    python scripts/preprocess_audio.py --manifest data/manifests/train.csv --dry-run

    # Force reprocess (ignore existing outputs):
    python scripts/preprocess_audio.py --manifest data/manifests/train.csv --force

    # Custom output directory:
    python scripts/preprocess_audio.py --dir audio/ --output-dir data/processed_audio/custom/

    # Show a summary at the end:
    python scripts/preprocess_audio.py --manifest data/manifests/train.csv --summary

    # Verbose debug logging:
    python scripts/preprocess_audio.py --file audio.wav --verbose
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".w64"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AVA LDM Audio Preprocessing — never modifies source audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source specification (mutually exclusive)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", "-m", metavar="CSV",
                     help="Manifest CSV with audio paths.")
    src.add_argument("--file", "-f", metavar="AUDIO",
                     help="Single audio file to process.")
    src.add_argument("--dir", "-d", metavar="DIR",
                     help="Directory of audio files to process recursively.")

    p.add_argument("--audio-col", default="audio_hf_path",
                   help="CSV column containing audio paths (default: audio_hf_path).")
    p.add_argument("--local-audio-col", default=None,
                   help="Alternative CSV column for local audio file paths.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for processed audio (default: from config).")
    p.add_argument("--log-path", default=None,
                   help="Path for quality log CSV (default: from config).")
    p.add_argument("--config", default="configs/audio.yaml",
                   help="Path to audio.yaml config.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate only — do not write output files.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if output already exists.")
    p.add_argument("--summary", action="store_true",
                   help="Print per-status summary table at end.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging.")
    return p.parse_args()


def _setup_logging(cfg: dict, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_file = cfg.get("logging", {}).get("log_file", "logs/audio_preprocessing.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler  # noqa: PLC0415
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler(
            log_file,
            maxBytes=cfg.get("logging", {}).get("max_bytes", 10 * 1024 * 1024),
            backupCount=cfg.get("logging", {}).get("backup_count", 3),
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


def _collect_from_dir(directory: str) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
    )
    return files


def _print_summary(results: list) -> None:
    counts: Counter = Counter(r.status for r in results)
    total = len(results)
    print("\n" + "=" * 55)
    print(f"  {'Status':<12} {'Count':>8}   {'%':>6}")
    print("  " + "-" * 35)
    for status in ("PASS", "WARN", "REJECT", "SKIP", "ERROR"):
        n = counts.get(status, 0)
        pct = 100 * n / max(total, 1)
        print(f"  {status:<12} {n:>8,}   {pct:>5.1f}%")
    print("  " + "-" * 35)
    print(f"  {'TOTAL':<12} {total:>8,}")
    print("=" * 55)

    # Rejection reasons
    rejections = [r for r in results if r.status in ("REJECT", "ERROR") and r.error]
    if rejections:
        print(f"\n  Top rejection reasons (first {min(5, len(rejections))}):")
        for r in rejections[:5]:
            print(f"    {r.input_path.name}: {r.error}")


def main() -> None:
    args = parse_args()

    from src.audio.preprocessing_pipeline import (  # noqa: PLC0415
        AudioPreprocessingPipeline,
        load_audio_config,
    )

    cfg = load_audio_config(args.config)
    _setup_logging(cfg, verbose=args.verbose)

    if args.force:
        cfg.setdefault("processing", {})["skip_existing"] = False

    if args.dry_run:
        logger.info("DRY RUN mode — no output files will be written.")
        # Monkey-patch the save step
        from src.audio import preprocessing_pipeline as _pp  # noqa: PLC0415
        _orig = _pp.sf.write
        _pp.sf.write = lambda *a, **kw: None  # no-op

    pipeline = AudioPreprocessingPipeline(
        cfg=cfg,
        output_dir=args.output_dir,
        log_path=args.log_path,
    )

    t0 = time.perf_counter()
    print("=" * 55)
    print("  AVA LDM Audio Preprocessing Pipeline")
    print("=" * 55)
    print(f"  Target SR : {pipeline.target_sr} Hz")
    print(f"  Output    : {pipeline.output_dir}")
    print(f"  Quality log: {pipeline.log_path}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Skip exist: {pipeline.skip_existing}")

    results: list = []

    if args.file:
        print(f"  Mode: single file")
        r = pipeline.process_file(args.file)
        results = [r]

    elif args.dir:
        paths = _collect_from_dir(args.dir)
        print(f"  Mode: directory ({len(paths)} audio files found)")
        results = pipeline.process_batch(paths)

    else:  # manifest
        print(f"  Mode: manifest ({args.manifest})")
        results = pipeline.process_manifest(
            manifest_path=args.manifest,
            audio_col=args.audio_col,
            local_audio_col=args.local_audio_col,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n  Processed {len(results)} files in {elapsed:.1f} s")
    print(f"  Quality log: {pipeline.log_path}")

    if args.summary or True:  # always show summary
        _print_summary(results)


if __name__ == "__main__":
    main()

