#!/usr/bin/env python3
"""
AVA LDM Studio — Environment Checker
=====================================
Verifies that the development environment meets all requirements for
the AVA LDM Studio Python backend (LDM training / evaluation pipeline).

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --json        # machine-readable output
    python scripts/check_environment.py --fail-fast   # exit 1 on first failure

The script DOES NOT train any models or download datasets.
"""

from __future__ import annotations

# Force UTF-8 stdout/stderr on Windows so Unicode symbols render correctly.
import sys as _sys
import io as _io
if hasattr(_sys.stdout, "buffer"):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "buffer"):
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import platform
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Minimum version constraints (tuples for easy comparison)
# ---------------------------------------------------------------------------
MIN_PYTHON = (3, 11)
MIN_TORCH = (2, 0, 0)
MIN_TRANSFORMERS = (4, 35, 0)
MIN_DATASETS = (2, 14, 0)

# ---------------------------------------------------------------------------
# ANSI colours (disabled on non-TTY / Windows without ANSICON)
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty() and platform.system() != "Windows" or os.environ.get("FORCE_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t: str) -> str:  return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str) -> str:    return _c(t, "31")
def bold(t: str) -> str:   return _c(t, "1")
def cyan(t: str) -> str:   return _c(t, "36")


# ---------------------------------------------------------------------------
# Result accumulation
# ---------------------------------------------------------------------------

class CheckResult:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def __init__(self, label: str, status: str, detail: str, value: Any = None):
        self.label = label
        self.status = status
        self.detail = detail
        self.value = value  # raw value for JSON output

    def __str__(self) -> str:
        icon = {self.PASS: green("✔"), self.WARN: yellow("⚠"), self.FAIL: red("✘")}[self.status]
        status_str = {
            self.PASS: green(self.PASS),
            self.WARN: yellow(self.WARN),
            self.FAIL: red(self.FAIL),
        }[self.status]
        return f"  {icon} [{status_str}] {bold(self.label)}: {self.detail}"


results: list[CheckResult] = []


def record(label: str, status: str, detail: str, value: Any = None) -> CheckResult:
    r = CheckResult(label, status, detail, value)
    results.append(r)
    return r


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python() -> CheckResult:
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    impl = platform.python_implementation()
    if (ver.major, ver.minor) < MIN_PYTHON:
        return record(
            "Python Version", CheckResult.FAIL,
            f"{impl} {ver_str} — requires >= {'.'.join(map(str, MIN_PYTHON))}",
            ver_str,
        )
    return record("Python Version", CheckResult.PASS, f"{impl} {ver_str}", ver_str)


def check_os() -> CheckResult:
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    detail = f"{system} {release} [{machine}]"
    return record("Operating System", CheckResult.PASS, detail, detail)


def check_torch() -> CheckResult:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return record("PyTorch", CheckResult.FAIL, "NOT INSTALLED — run: pip install torch", None)

    ver_str = torch.__version__
    # Parse version ignoring dev/rc suffixes
    parts = ver_str.split("+")[0].split(".")
    try:
        ver_tuple = tuple(int(x) for x in parts[:3])
    except ValueError:
        ver_tuple = (0, 0, 0)

    if ver_tuple < MIN_TORCH:
        return record(
            "PyTorch", CheckResult.WARN,
            f"{ver_str} — recommended >= {'.'.join(map(str, MIN_TORCH))}",
            ver_str,
        )
    return record("PyTorch", CheckResult.PASS, ver_str, ver_str)


def check_cuda() -> tuple[CheckResult, CheckResult]:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        cuda_r = record("CUDA Available", CheckResult.WARN, "Cannot check — PyTorch not installed", False)
        gpu_r  = record("GPU Name",       CheckResult.WARN, "N/A", None)
        return cuda_r, gpu_r

    available = torch.cuda.is_available()
    if available:
        device_count = torch.cuda.device_count()
        cuda_ver = torch.version.cuda or "unknown"
        cuda_r = record(
            "CUDA Available", CheckResult.PASS,
            f"Yes — CUDA {cuda_ver}, {device_count} device(s)",
            True,
        )
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_r = record("GPU Name", CheckResult.PASS, gpu_name, gpu_name)
        except Exception as exc:  # noqa: BLE001
            gpu_r = record("GPU Name", CheckResult.WARN, f"Could not read GPU name: {exc}", None)
    else:
        cuda_r = record(
            "CUDA Available", CheckResult.WARN,
            "No — running on CPU only (training will be slow)",
            False,
        )
        gpu_r = record("GPU Name", CheckResult.WARN, "N/A — no CUDA device", None)

    return cuda_r, gpu_r


def check_transformers() -> CheckResult:
    try:
        import transformers  # noqa: PLC0415
    except ImportError:
        return record(
            "Transformers", CheckResult.FAIL,
            "NOT INSTALLED — run: pip install transformers",
            None,
        )
    ver_str = transformers.__version__
    parts = ver_str.split(".")
    try:
        ver_tuple = tuple(int(x) for x in parts[:3])
    except ValueError:
        ver_tuple = (0, 0, 0)

    if ver_tuple < MIN_TRANSFORMERS:
        return record(
            "Transformers", CheckResult.WARN,
            f"{ver_str} — recommended >= {'.'.join(map(str, MIN_TRANSFORMERS))}",
            ver_str,
        )
    return record("Transformers", CheckResult.PASS, ver_str, ver_str)


def check_datasets() -> CheckResult:
    try:
        import datasets  # noqa: PLC0415
    except ImportError:
        return record(
            "Datasets", CheckResult.FAIL,
            "NOT INSTALLED — run: pip install datasets",
            None,
        )
    ver_str = datasets.__version__
    parts = ver_str.split(".")
    try:
        ver_tuple = tuple(int(x) for x in parts[:3])
    except ValueError:
        ver_tuple = (0, 0, 0)

    if ver_tuple < MIN_DATASETS:
        return record(
            "Datasets", CheckResult.WARN,
            f"{ver_str} — recommended >= {'.'.join(map(str, MIN_DATASETS))}",
            ver_str,
        )
    return record("Datasets", CheckResult.PASS, ver_str, ver_str)


def check_hf_auth() -> CheckResult:
    try:
        import huggingface_hub as hf_hub  # noqa: PLC0415
    except ImportError:
        return record(
            "HF Authentication", CheckResult.FAIL,
            "huggingface_hub NOT INSTALLED — run: pip install huggingface_hub",
            None,
        )

    hub_ver = getattr(hf_hub, "__version__", "unknown")

    # Retrieve token: modern API (hub >= 0.16 / 1.x) uses get_token()
    token = None
    try:
        token = hf_hub.get_token()
    except AttributeError:
        # Very old hub — fall back to HfFolder
        try:
            token = hf_hub.HfFolder.get_token()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    # Also honour environment variables
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        return record(
            "HF Authentication", CheckResult.WARN,
            f"Not logged in (hub {hub_ver}) — run: huggingface-cli login",
            None,
        )

    try:
        user_info = hf_hub.whoami(token=token)
        username = user_info.get("name") or user_info.get("fullname") or "Unknown"
        return record(
            "HF Authentication", CheckResult.PASS,
            f"Logged in as '{username}' (hub {hub_ver})",
            username,
        )
    except Exception as exc:  # noqa: BLE001
        return record(

            "HF Authentication", CheckResult.WARN,
            f"Token found but verification failed: {exc}",
            None,
        )


def check_audio_backends() -> list[CheckResult]:
    audio_results = []

    # librosa
    try:
        import librosa  # noqa: PLC0415
        audio_results.append(
            record("librosa", CheckResult.PASS, librosa.__version__, librosa.__version__)
        )
    except ImportError:
        audio_results.append(
            record("librosa", CheckResult.FAIL, "NOT INSTALLED — run: pip install librosa", None)
        )

    # soundfile (libsndfile backend)
    try:
        import soundfile as sf  # noqa: PLC0415
        # Try a quick codec list to confirm the native lib loaded
        codecs = sf.available_formats()
        codec_count = len(codecs)
        audio_results.append(
            record(
                "soundfile", CheckResult.PASS,
                f"{sf.__version__} ({codec_count} formats available)",
                sf.__version__,
            )
        )
    except ImportError:
        audio_results.append(
            record("soundfile", CheckResult.FAIL, "NOT INSTALLED — run: pip install soundfile", None)
        )
    except Exception as exc:  # noqa: BLE001
        audio_results.append(
            record("soundfile", CheckResult.WARN, f"Installed but codec load failed: {exc}", None)
        )

    # torchaudio
    try:
        import torchaudio  # noqa: PLC0415
        backend = None
        try:
            backend = torchaudio.list_audio_backends()
        except Exception:  # noqa: BLE001
            pass
        backend_str = f"backends={backend}" if backend else ""
        audio_results.append(
            record(
                "torchaudio", CheckResult.PASS,
                f"{torchaudio.__version__} {backend_str}".strip(),
                torchaudio.__version__,
            )
        )
    except ImportError:
        audio_results.append(
            record("torchaudio", CheckResult.FAIL, "NOT INSTALLED — run: pip install torchaudio", None)
        )

    return audio_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AVA LDM Studio environment checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit with code 1 on first FAIL (still prints all WARNings)",
    )
    return p.parse_args()


def run_checks(fail_fast: bool = False) -> None:
    checks = [
        ("System", [check_python, check_os]),
        ("PyTorch & CUDA", [check_torch, check_cuda]),
        ("Hugging Face", [check_transformers, check_datasets, check_hf_auth]),
        ("Audio Backends", [check_audio_backends]),
    ]

    for section, fns in checks:
        print(f"\n{cyan(bold(f'── {section} '))}{'─' * max(0, 50 - len(section))}")
        for fn in fns:
            result = fn()
            if isinstance(result, (list, tuple)):
                for r in result:
                    print(r)
                    if fail_fast and r.status == CheckResult.FAIL:
                        print(red("\n[FAIL-FAST] Aborting on first failure."))
                        sys.exit(1)
            else:
                print(result)
                if fail_fast and result.status == CheckResult.FAIL:
                    print(red("\n[FAIL-FAST] Aborting on first failure."))
                    sys.exit(1)


def summarise() -> int:
    passed = sum(1 for r in results if r.status == CheckResult.PASS)
    warned = sum(1 for r in results if r.status == CheckResult.WARN)
    failed = sum(1 for r in results if r.status == CheckResult.FAIL)

    print(f"\n{'═' * 55}")
    print(bold("SUMMARY"))
    print(f"  {green(f'{passed} passed')}  {yellow(f'{warned} warnings')}  {red(f'{failed} failures')}")
    if failed == 0 and warned == 0:
        print(green("  ✔ Environment is fully ready."))
    elif failed == 0:
        print(yellow("  ⚠ Environment is usable but has warnings."))
    else:
        print(red("  ✘ Fix failures before proceeding."))
    print(f"{'═' * 55}\n")
    return 1 if failed > 0 else 0


def main() -> None:
    args = parse_args()

    if not args.json:
        print(bold(cyan("\n╔══════════════════════════════════════════════════╗")))
        print(bold(cyan("║     AVA LDM Studio — Environment Check           ║")))
        print(bold(cyan("╚══════════════════════════════════════════════════╝")))
        print(f"  Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Executable: {sys.executable}")

    run_checks(fail_fast=args.fail_fast)

    if args.json:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "executable": sys.executable,
            "results": [
                {
                    "label": r.label,
                    "status": r.status,
                    "detail": r.detail,
                    "value": r.value,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(0)

    exit_code = summarise()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
