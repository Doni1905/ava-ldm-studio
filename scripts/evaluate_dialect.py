#!/usr/bin/env python3
"""
scripts/evaluate_dialect.py
============================
Evaluate a trained Tamil dialect classifier on a test manifest.

Usage
-----
    python scripts/evaluate_dialect.py --checkpoint models/dialect/best_model.pt
    python scripts/evaluate_dialect.py --checkpoint models/dialect/best_model.pt \\
                                       --manifest data/manifests/test.csv \\
                                       --output results/dialect/test_eval/
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

from src.dialect import DialectClassifier, DialectEvaluator, DialectLabelRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Tamil Dialect Classifier")
    p.add_argument("--config",     default="configs/dialect.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint (.pt)")
    p.add_argument("--manifest",   default=None, help="Path to test manifest CSV")
    p.add_argument("--output",     default="results/dialect/eval/", help="Output directory")
    p.add_argument("--audio-col",  default="file")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    registry  = DialectLabelRegistry(config=cfg)
    model     = DialectClassifier.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=cfg,
        num_classes=registry.num_classes,
    )
    evaluator = DialectEvaluator(config=cfg, label_registry=registry)

    manifest = args.manifest or cfg.get("paths", {}).get("test_manifest", "data/manifests/test.csv")
    result   = evaluator.evaluate(
        model=model,
        manifest_path=manifest,
        audio_col=args.audio_col,
        output_dir=args.output,
    )

    print(f"\n=== Dialect Evaluation Results ===")
    print(f"  Accuracy  : {result['accuracy']:.4f}")
    print(f"  Macro F1  : {result['macro_f1']:.4f}")
    print(f"  N samples : {result['n_samples']}")
    print(f"\nPer-class breakdown:")
    for lbl, metrics in result["per_class"].items():
        print(f"  {lbl:<12}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  n={int(metrics['support'])}")
    print(f"\n{result['confusion_matrix_str']}")


if __name__ == "__main__":
    main()
