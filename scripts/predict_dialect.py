#!/usr/bin/env python3
"""
scripts/predict_dialect.py
===========================
Run dialect inference on individual audio files or a manifest.

Usage
-----
    # Single file
    python scripts/predict_dialect.py --checkpoint models/dialect/best_model.pt \\
                                      --file path/to/audio.wav

    # Batch from manifest
    python scripts/predict_dialect.py --checkpoint models/dialect/best_model.pt \\
                                      --manifest data/manifests/test.csv \\
                                      --output results/dialect/predictions.csv
"""
import io, sys, argparse, logging, yaml, json
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dialect import DialectInferencePipeline, DialectLabelRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Tamil Dialect Inference")
    p.add_argument("--config",     default="configs/dialect.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint (.pt)")
    p.add_argument("--file",       default=None, help="Single audio file")
    p.add_argument("--manifest",   default=None, help="Manifest CSV (batch mode)")
    p.add_argument("--audio-col",  default="file")
    p.add_argument("--output",     default=None, help="Output CSV for batch predictions")
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

    registry = DialectLabelRegistry(config=cfg)
    pipeline = DialectInferencePipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=cfg,
        label_registry=registry,
    )

    if args.file:
        result = pipeline.predict_file(args.file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.manifest:
        import pandas as pd
        df = pd.read_csv(args.manifest)
        audio_col = args.audio_col
        if audio_col not in df.columns:
            for c in ("file", "audio_path", "audio_hf_path"):
                if c in df.columns:
                    audio_col = c
                    break

        audio_paths = df[audio_col].tolist()
        results = pipeline.predict_batch(audio_paths)

        result_df = pd.DataFrame(results)
        out_path  = args.output or "results/dialect/predictions.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        print(f"Saved {len(results)} predictions to {out_path}")

        # Print summary
        if "dialect" in result_df.columns:
            print("\nDialect distribution:")
            print(result_df["dialect"].value_counts().to_string())
        return

    print("Provide --file or --manifest. Use --help for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
