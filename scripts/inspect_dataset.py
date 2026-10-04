#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
===========================
Inspect the SPRINGLab/IndicVoices-R_Tamil dataset WITHOUT downloading
any audio data.

What this script does
---------------------
1.  Fetches the dataset card and schema from the HF Hub API.
2.  Prints the full schema with a ★ marker next to columns that will be
    retained in the AVA working dataset.
3.  Optionally probes a single parquet shard to verify the live schema.
4.  Prints a summary of the 14 parquet shards.

Usage
-----
    # From project root with venv activated:
    python scripts/inspect_dataset.py

    # Probe live parquet shard schema (requires network + optional HF token):
    python scripts/inspect_dataset.py --probe-shard 0

    # JSON output (for CI / scripting):
    python scripts/inspect_dataset.py --json

    # Verbose debug logging:
    python scripts/inspect_dataset.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src`` can be imported
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect IndicVoices-R Tamil dataset (no audio download)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--probe-shard",
        type=int,
        default=None,
        metavar="N",
        help="Probe the live schema of shard N (0-13). Requires network.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of pretty-print.",
    )
    p.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset.yaml config (default: configs/dataset.yaml).",
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
    info = dm.inspect()

    if args.probe_shard is not None:
        print(f"\nProbing live shard {args.probe_shard} schema …")
        live_schema = dm.loader.inspect_shard_schema(args.probe_shard)
        info["live_shard_schema"] = live_schema
        print("Live shard schema:")
        for col, dtype in live_schema.items():
            print(f"  {col:<35} {dtype}")

    if args.json:
        # Sanitise for JSON
        safe = {k: v for k, v in info.items() if k != "card_text"}
        print(json.dumps(safe, indent=2, default=str))
    else:
        print(
            f"\nCard excerpt:\n{'─' * 60}\n"
            + info.get("card_text", "")[:800]
            + "\n" + "─" * 60
        )
        print("\nShard files (all parquet, no audio download needed):")
        for i in range(info.get("num_shards", 14)):
            url = dm.loader.shard_url(i)
            print(f"  [{i:02d}] {url}")

    print(
        f"\n✔ Inspection complete. "
        f"{info.get('num_examples', '?'):,} total examples across "
        f"{info.get('num_shards', '?')} shards ({info.get('download_size_gb', '?')} GB total — "
        f"NOT downloading)."
    )


if __name__ == "__main__":
    main()
