#!/usr/bin/env python3
"""
scripts/create_splits.py
=========================
Create speaker-disjoint train / validation / test splits from the
selected manifest (data/manifests/selected_manifest.csv).

Outputs
-------
    data/manifests/train.csv       (~80 % of rows)
    data/manifests/validation.csv  (~10 % of rows)
    data/manifests/test.csv        (~10 % of rows)

Speaker disjointness guarantee
-------------------------------
Every speaker_id in the dataset is assigned to exactly ONE split.
No speaker_id appears in more than one of train / validation / test.
This is verified by a hard assertion after splitting — if violated,
the script exits with a non-zero return code and an error message.

Algorithm summary
-----------------
See src/data/splitter.py for the full algorithm description.
Short version:
  1. Group utterances by speaker.
  2. Compute per-speaker demographic profile.
  3. Sort + shuffle speakers within demographic strata.
  4. Assign speakers to splits to hit the target fractions.
  5. Map split labels back to utterance rows.
  6. Assert disjointness.

Usage
-----
    python scripts/create_splits.py

    # Use a different manifest:
    python scripts/create_splits.py --manifest data/manifests/selected_manifest.csv

    # Force rebuild (overwrite existing splits):
    python scripts/create_splits.py --force

    # Print per-split statistics:
    python scripts/create_splits.py --stats

    # Verify an existing set of splits without rebuilding:
    python scripts/create_splits.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create speaker-disjoint train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to the selected manifest CSV "
            "(default: data/manifests/selected_manifest.csv from config)."
        ),
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing split files.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print per-split statistics table.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify disjointness of existing splits (do not rebuild).",
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


def _print_split_table(splits: dict) -> None:
    import pandas as pd

    print("\n" + "=" * 65)
    print(f"  {'Split':<12} {'Rows':>8} {'Speakers':>10} {'%':>6}")
    print("  " + "-" * 40)
    total_rows = sum(len(v) for v in splits.values())
    for name, df in splits.items():
        spk = df["speaker_id"].nunique() if "speaker_id" in df.columns else "N/A"
        pct = 100 * len(df) / max(total_rows, 1)
        print(f"  {name.upper():<12} {len(df):>8,} {str(spk):>10} {pct:>5.1f}%")
    print("  " + "-" * 40)
    print(f"  {'TOTAL':<12} {total_rows:>8,}")
    print("=" * 65)

    # Per-split distributions
    for split_name, df in splits.items():
        print(f"\n  [{split_name.upper()}] demographic distribution:")
        for col in ("gender", "age_group", "area", "scenario"):
            if col in df.columns:
                dist = df[col].value_counts().to_dict()
                print(f"    {col:>12}: {dist}")


def main() -> None:
    args = parse_args()
    import pandas as pd

    from src.data.dataset_manager import DatasetManager

    dm = DatasetManager(config_path=args.config, verbose=args.verbose)

    # ----------------------------------------------------------------
    # Verify-only mode
    # ----------------------------------------------------------------
    if args.verify_only:
        print("=== VERIFY-ONLY MODE ===")
        paths = dm.cfg["paths"]
        splits = {}
        for name, key in [("train", "train_manifest"), ("val", "val_manifest"), ("test", "test_manifest")]:
            p = Path(paths[key])
            if not p.exists():
                print(f"  ✘ Missing: {p}")
                sys.exit(1)
            splits[name] = pd.read_csv(p)
            print(f"  ✔ Loaded {name}: {len(splits[name]):,} rows")

        try:
            dm.splitter.verify_disjoint(splits)
            print("\n  ✔ Speaker disjointness: VERIFIED")
        except AssertionError as exc:
            print(f"\n  ✘ Disjointness violation: {exc}")
            sys.exit(1)

        if args.stats:
            _print_split_table(splits)
        return

    # ----------------------------------------------------------------
    # Normal split creation
    # ----------------------------------------------------------------
    print("=" * 65)
    print("  AVA LDM Studio — Create Splits")
    print("=" * 65)

    manifest_path = Path(args.manifest) if args.manifest else Path(dm.cfg["paths"]["selected_manifest"])
    if not manifest_path.exists():
        print(f"  ✘ Manifest not found: {manifest_path}")
        print("  Run `python scripts/build_manifest.py` first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"  Manifest loaded: {len(manifest):,} rows from {manifest_path}")

    splits = dm.create_splits(manifest=manifest, force=args.force)

    print(f"\n  ✔ Splits created:")
    for name, df in splits.items():
        key_map = {"train": "train_manifest", "val": "val_manifest", "test": "test_manifest"}
        out_path = Path(dm.cfg["paths"][key_map[name]])
        print(f"    {name.upper():>5}: {len(df):,} rows → {out_path}")

    if args.stats:
        _print_split_table(splits)

    print(
        "\nSpeaker disjointness: ✔ VERIFIED\n"
        "Splits are ready for use in LDM training.\n"
    )


if __name__ == "__main__":
    main()
