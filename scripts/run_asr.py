#!/usr/bin/env python3
"""
scripts/run_asr.py
===================
CLI entry point for AVA LDM ASR Inference.

Usage
-----
    # Process a manifest (e.g. data/manifests/test.csv):
    python scripts/run_asr.py --manifest data/manifests/test.csv

    # Process a sample dataset (first N records for testing)
    python scripts/run_asr.py --manifest data/manifests/test.csv --limit 20
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd

# UTF-8 stdout
import io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.asr import WhisperEngine, ASRInferencePipeline

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Run ASR Inference")
    p.add_argument("--manifest", required=True, help="Path to manifest CSV (must contain audio paths and reference texts).")
    p.add_argument("--config", default="configs/asr.yaml", help="Path to ASR config YAML.")
    p.add_argument("--limit", type=int, default=None, help="Limit to N samples (for testing/verification).")
    p.add_argument("--audio-col", default="audio_hf_path", help="Column containing the path to the audio file.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args()

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    cfg = load_config(args.config)
    
    # Initialize Engine (Whisper by default, could use registry for others)
    engine = WhisperEngine(cfg)
    
    # Initialize Inference Pipeline
    pipeline = ASRInferencePipeline(engine, cfg)
    
    # Handle manifest limit
    manifest_path = args.manifest
    if args.limit:
        logger.info(f"Limiting to first {args.limit} samples.")
        df = pd.read_csv(manifest_path)
        limited_df = df.head(args.limit)
        
        # Save temporary manifest
        manifest_path = str(Path(manifest_path).with_name(f"temp_limited_{Path(manifest_path).name}"))
        limited_df.to_csv(manifest_path, index=False)
        
    try:
        pipeline.run_inference(manifest_path, audio_col=args.audio_col)
    finally:
        # Cleanup temporary manifest if used
        if args.limit and os.path.exists(manifest_path):
            os.remove(manifest_path)

if __name__ == "__main__":
    main()

