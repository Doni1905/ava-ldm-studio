#!/usr/bin/env python3
"""
scripts/train_dialect.py
=========================
Train the Tamil dialect classifier head.

Usage
-----
    python scripts/train_dialect.py
    python scripts/train_dialect.py --config configs/dialect.yaml
    python scripts/train_dialect.py --train data/manifests/train.csv \\
                                    --val   data/manifests/validation.csv
"""
import io, sys, argparse, logging, yaml
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dialect import DialectLabelRegistry, DialectTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Tamil Dialect Classifier")
    p.add_argument("--config",  default="configs/dialect.yaml")
    p.add_argument("--train",   default=None, help="Override train manifest path")
    p.add_argument("--val",     default=None, help="Override val manifest path")
    p.add_argument("--audio-col", default="file", help="Column containing audio paths")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    registry = DialectLabelRegistry(config=cfg)
    trainer  = DialectTrainer(config=cfg, label_registry=registry)

    train_path = args.train or cfg.get("paths", {}).get("train_manifest", "data/manifests/train.csv")
    val_path   = args.val   or cfg.get("paths", {}).get("val_manifest",   "data/manifests/validation.csv")

    print(f"Training dialect classifier")
    print(f"  Labels : {registry.names}")
    print(f"  Train  : {train_path}")
    print(f"  Val    : {val_path}")

    result = trainer.train(
        train_manifest=train_path,
        val_manifest=val_path,
        audio_col=args.audio_col,
    )
    print(f"\nTraining complete. Best val_acc = {result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
