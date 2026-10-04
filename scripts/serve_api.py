"""
scripts/serve_api.py
====================
CLI launcher for the AVA LDM Studio Local API Server.

Usage:
    python scripts/serve_api.py
    python scripts/serve_api.py --host 0.0.0.0 --port 8080
"""

import argparse
import io
import logging
import sys
from pathlib import Path

# Safe UTF-8 reconfiguration for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.server import run_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Start the AVA LDM Studio Local REST API Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to (e.g. 127.0.0.1 or 0.0.0.0 for LAN/emulator).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000).")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      AVA LDM STUDIO — LOCAL INTEGRATION API SERVER")
    print("=" * 60)
    print(f"Target Host : http://{args.host}:{args.port}")
    print("Endpoints   :")
    print("  - GET  /health")
    print("  - POST /transcribe")
    print("  - POST /detect-language")
    print("  - POST /detect-dialect")
    print("  - POST /normalize")
    print("  - POST /analyze  (Primary)")
    print("  - POST /evaluate")
    print("=" * 60 + "\n")

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
