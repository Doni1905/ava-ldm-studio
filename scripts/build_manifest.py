#!/usr/bin/env python3
"""
scripts/build_manifest.py
==========================
Stream metadata from SPRINGLab/IndicVoices-R_Tamil parquet shards,
apply quality filters, run stratified sampling, and write
data/manifests/selected_manifest.csv.

Audio bytes are NEVER downloaded. The resulting manifest contains:
    - All 12 required AVA LDM columns (except raw audio bytes)
    - ``audio_hf_path``  : hf:// URI pointing to the shard parquet file
    - ``shard_idx``      : which shard this row came from
    - ``row_within_shard``: row position within that shard
    - ``stratum_key``    : stratification cell label (for auditing)
    - ``selection_rank`` : final selection rank (1 = selected first)

Resume safety
-------------
Each shard's metadata is cached to data/cache/metadata/shard_NNNNN.parquet
after processing. If the script is interrupted, re-running it will skip
already-processed shards and resume from the next one.

Use --force to clear the checkpoint and restart from scratch.

Usage
-----
    # Full run (all 14 shards → 5 000 selected rows):
    python scripts/build_manifest.py

    # Quick test run (first 2 shards only):
    python scripts/build_manifest.py --max-shards 2

    # Force rebuild (ignore checkpoint):
    python scripts/build_manifest.py --force

    # Use datasets library streaming instead of direct parquet download:
    python scripts/build_manifest.py --streaming

    # Verbose:
    python scripts/build_manifest.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build AVA LDM manifest from IndicVoices-R Tamil (no audio download)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Clear checkpoint and rebuild from scratch.",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N shards (default: all 14).",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use the HF datasets library in streaming mode instead of "
            "direct parquet download. Slower but avoids storing shard files."
        ),
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics after building the manifest.",
    )
    p.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset.yaml config.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.data.dataset_manager import DatasetManager

    dm = DatasetManager(config_path=args.config, verbose=args.verbose)

    print("=" * 65)
    print("  AVA LDM Studio — Build Manifest")
    print("=" * 65)
    print(f"  Dataset    : {dm.cfg['dataset']['repo_id']}")
    print(f"  Target rows: {dm.cfg['selection']['target_n']:,}")
    print(f"  Shards     : {args.max_shards or dm.cfg['dataset']['num_shards']}")
    print(f"  Mode       : {'HF streaming' if args.streaming else 'direct parquet'}")
    print(f"  Force      : {args.force}")
    print("=" * 65)

    t0 = time.perf_counter()

    if args.streaming:
        # --- Streaming path via datasets library ---
        print("\n[1/3] Streaming metadata via datasets library …")
        raw_df = dm.loader.stream_via_datasets_library(
            max_examples=dm.cfg["selection"].get("stream_cap", 50000),
        )

        print(f"[2/3] Applying column selection & quality filters …")
        filtered = dm.selector.select_and_filter(raw_df)

        print(f"[3/3] Stratified sampling → {dm.cfg['selection']['target_n']:,} rows …")
        selected = dm.sampler.sample(filtered)

        # Write manifest
        manifest_path = Path(dm.cfg["paths"]["selected_manifest"])
        selected.to_csv(manifest_path, index=False)
        print(f"\n✔ Manifest written → {manifest_path}")

    else:
        # --- Direct parquet path (resume-safe) ---
        manifest = dm.build_manifest(
            force=args.force,
            max_shards=args.max_shards,
        )
        print(f"\n✔ Manifest ready: {len(manifest):,} rows")

        manifest_path = Path(dm.cfg["paths"]["selected_manifest"])

    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  File   : {manifest_path.resolve()}")

    if args.stats:
        print("\n=== Dataset Statistics ===")
        import pandas as pd
        df = pd.read_csv(manifest_path)
        stats = dm.get_statistics(df)
        print(json.dumps(stats, indent=2, default=str))

    print(
        "\nNext step:\n"
        "  python scripts/create_splits.py\n"
    )


if __name__ == "__main__":
    main()
