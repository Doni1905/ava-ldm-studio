"""
scripts/run_ldm.py
==================
End-to-end CLI for the unified AVA LDM Pipeline.

Usage:
    python scripts/run_ldm.py --audio <path_to_audio_file>
"""

import argparse
import io
import logging
import sys
from pathlib import Path

# Ensure utf-8 output for Windows console
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ldm.pipeline import LdmPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run the AVA LDM Pipeline end-to-end.")
    parser.add_argument("--audio", type=str, default=None, help="Path to the input audio file.")
    parser.add_argument("--text", type=str, default=None, help="Direct text transcript to analyze.")
    parser.add_argument("--dialect", type=str, default=None, help="Optional dialect override (Chennai, Kongu, Madurai, Nellai, Standard).")
    parser.add_argument("--json", action="store_true", help="Output raw JSON only")
    args = parser.parse_args()
    
    if not args.audio and not args.text:
        logging.error("Please provide either --audio <file> or --text \"<transcript>\".")
        sys.exit(1)
        
    # Initialize pipeline
    pipeline = LdmPipeline()
    
    # Process
    try:
        if args.audio:
            audio_path = Path(args.audio)
            if not audio_path.exists():
                logging.error(f"Audio file not found: {audio_path}")
                sys.exit(1)
            result = pipeline.process(audio_path)
        else:
            result = pipeline.process_text(args.text, dialect=args.dialect)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)
        
    if args.json:
        print(result.to_json())
    else:
        print("\n" + "="*50)
        print(" AVA LDM PIPELINE RESULT")
        print("="*50)
        print(f"Transcript : {result.transcript}")
        print(f"Language   : {result.language}")
        print(f"Code-Mixed : {result.code_mixed}")
        print(f"Dialect    : {result.dialect} (conf: {result.confidence['dialect']:.2f})")
        print("-" * 50)
        print(f"Normalized : {result.normalized_text}")
        print(f"Intent     : {result.intent} (conf: {result.confidence['intent']:.2f})")
        
        if result.entities:
            print("Entities   :")
            for k, v in result.entities.items():
                print(f"  - {k}: {v}")
        else:
            print("Entities   : None")
            
        print("-" * 50)
        print("Latency (ms):")
        for step, ms in result.latency_ms.items():
            if step != "total":
                print(f"  {step:<25}: {ms:>7.1f}")
        print(f"  {'total':<25}: {result.latency_ms['total']:>7.1f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()

