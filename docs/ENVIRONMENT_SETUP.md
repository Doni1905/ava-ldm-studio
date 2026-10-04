# AVA LDM Studio — Environment Setup

> **Scope:** Python backend for the AVA Linguistic Dialect Model (LDM) —
> the training, evaluation and dataset utilities.
> The front-end (React / TanStack Start) has its own toolchain (`bun install`).
> These instructions cover **only** the Python stack.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Layout](#2-repository-layout)
3. [Virtual Environment Setup](#3-virtual-environment-setup)
4. [PyTorch Installation](#4-pytorch-installation)
5. [Install Remaining Dependencies](#5-install-remaining-dependencies)
6. [Hugging Face Authentication](#6-hugging-face-authentication)
7. [Run the Environment Check](#7-run-the-environment-check)
8. [Troubleshooting](#8-troubleshooting)
9. [Dependency Reference](#9-dependency-reference)

---

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| **Python** | 3.11 | 3.14 used in development |
| **pip** | 23+ | `python -m pip install --upgrade pip` |
| **Git** | any | for cloning |
| **CUDA toolkit** | 11.8+ *(optional)* | Required only for GPU training |
| **Node.js / Bun** | see `package.json` | Front-end only — not needed for Python work |

> [!IMPORTANT]
> The project already ships a working virtual environment at `./venv`
> (Python 3.14.2, Windows x64). If you are on the **same machine** where
> the repo was cloned, skip to [§7 Run the Environment Check](#7-run-the-environment-check).
> Only follow §3–5 if you are setting up on a **new machine** or a
> fresh clone.

---

## 2. Repository Layout

```
ava-ldm-studio/
├── venv/                    # Python virtual environment (do not commit)
├── scripts/
│   └── check_environment.py # ← environment verifier (run this first)
├── src/
│   └── lib/ldm/             # TypeScript rule-based LDM prototype
│       ├── processor.ts     # Core normalization logic
│       ├── dataset.ts       # 30-sample synthetic dataset
│       ├── evaluation.ts    # Prototype metrics
│       └── types.ts         # Shared type contracts
├── docs/
│   └── ENVIRONMENT_SETUP.md # ← this file
├── requirements.txt         # Pinned Python dependencies
├── package.json             # Front-end dependencies (Bun / npm)
└── README.md
```

---

## 3. Virtual Environment Setup

> [!NOTE]
> Skip this section if `./venv` already exists and you are on the same
> machine. The `check_environment.py` script will confirm whether the
> existing env is healthy.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

---

## 4. PyTorch Installation

PyTorch must be installed **before** the rest of the requirements because
its CUDA variant is served from a separate index. Installing `torch` from
the default PyPI index will silently install the CPU-only build.

### 4a. GPU Build (CUDA 12.6 — recommended if you have an NVIDIA GPU)

```bash
pip install torch==2.14.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 4b. CPU-only Build

```bash
pip install torch==2.14.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

> [!WARNING]
> Training on CPU is **very slow**. For anything beyond a quick smoke
> test, use a machine with an NVIDIA GPU (CUDA 11.8+) or a cloud VM
> (Google Colab, Kaggle, RunPod, etc.).

---

## 5. Install Remaining Dependencies

After PyTorch, install the rest of the pinned stack:

```bash
pip install -r requirements.txt
```

`requirements.txt` is pinned to the exact versions verified in the
development environment. You can safely run this command repeatedly —
`pip` skips packages that are already at the correct version.

---

## 6. Hugging Face Authentication

Some datasets and models on the Hub require an account. Login once:

```bash
huggingface-cli login
```

Paste your token from <https://huggingface.co/settings/tokens> (read-only
access is sufficient for evaluation; write access is needed if you plan
to publish models).

Alternatively, export the token as an environment variable so CI/CD
pipelines don't require interactive login:

```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# macOS / Linux
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 7. Run the Environment Check

Always run this before starting any training or evaluation work:

```bash
# Windows — activate the venv first
.\venv\Scripts\Activate.ps1
python scripts/check_environment.py

# macOS / Linux
source venv/bin/activate
python scripts/check_environment.py
```

### Expected output (all checks green)

```
╔══════════════════════════════════════════════════╗
║     AVA LDM Studio — Environment Check           ║
╚══════════════════════════════════════════════════╝
  Timestamp : 2026-09-30 01:20:00
  Executable: C:\...\venv\Scripts\python.exe

── System ──────────────────────────────────────────
  ✔ [PASS] Python Version: CPython 3.14.2
  ✔ [PASS] Operating System: Windows 11 [AMD64]

── PyTorch & CUDA ──────────────────────────────────
  ✔ [PASS] PyTorch: 2.14.0
  ✔ [PASS] CUDA Available: Yes — CUDA 12.6, 1 device(s)
  ✔ [PASS] GPU Name: NVIDIA GeForce RTX 3060

── Hugging Face ────────────────────────────────────
  ✔ [PASS] Transformers: 5.17.0
  ✔ [PASS] Datasets: 5.0.1
  ✔ [PASS] HF Authentication: Logged in as 'your-username'

── Audio Backends ──────────────────────────────────
  ✔ [PASS] librosa: 1.0.0
  ✔ [PASS] soundfile: 0.14.0 (59 formats available)
  ✔ [PASS] torchaudio: 2.11.0

═══════════════════════════════════════════════════
SUMMARY
  13 passed  0 warnings  0 failures
  ✔ Environment is fully ready.
═══════════════════════════════════════════════════
```

### Machine-readable (JSON) output

```bash
python scripts/check_environment.py --json
```

### Fail fast (stop on first error — useful in CI)

```bash
python scripts/check_environment.py --fail-fast
```

---

## 8. Troubleshooting

### CUDA says "Not available" even though I have an NVIDIA GPU

1. Confirm the CUDA toolkit is installed: `nvcc --version`
2. Make sure you installed the correct torch CUDA variant (§4a).
3. Check driver compatibility: `nvidia-smi`

### `soundfile` loads but reports 0 formats

The `libsndfile` native library did not load. On Windows, ensure the
`venv` was created from the same Python architecture as your system
(64-bit Python → 64-bit venv). Re-install: `pip install --force-reinstall soundfile`

### `huggingface-cli login` hangs

Paste the token directly via the `HF_TOKEN` environment variable (§6)
instead of using the interactive flow.

### `scipy` or `numba` fails to import on Python 3.14

`scipy==1.18.1` and `numba==0.67.0` are the first releases with Python
3.14 wheels. If you see a `ModuleNotFoundError` for native extensions,
upgrade pip and try: `pip install --upgrade scipy numba`

---

## 9. Dependency Reference

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.14.0 | Core DL framework |
| `torchaudio` | 2.11.0 | Audio tensor I/O |
| `transformers` | 5.17.0 | Pre-trained model access |
| `datasets` | 5.0.1 | Dataset loading & processing |
| `huggingface_hub` | 1.33.0 | Hub API / auth |
| `tokenizers` | 0.23.2 | Fast tokenization |
| `safetensors` | 0.8.0 | Safe model serialisation |
| `librosa` | 1.0.0 | Audio feature extraction |
| `soundfile` | 0.14.0 | Audio file I/O (libsndfile) |
| `soxr` | 1.1.0 | High-quality resampling |
| `numpy` | 2.5.3 | N-dimensional arrays |
| `scipy` | 1.18.1 | Signal processing |
| `scikit-learn` | 1.9.1 | ML utilities / metrics |
| `pandas` | 3.0.6 | Tabular data |
| `pyarrow` | 25.0.1 | Columnar data (datasets backend) |
| `sympy` | 1.14.0 | Symbolic math (torch dependency) |
| `numba` | 0.67.0 | JIT compilation (librosa dep) |
| `llvmlite` | 0.49.0 | LLVM bindings (numba dep) |
| `tqdm` | 4.70.1 | Progress bars |
| `pyyaml` | 6.0.3 | YAML config files |

> [!TIP]
> Run `pip list` inside the activated venv at any time to see the full
> flat list of installed packages (including transitive dependencies).
