# AVA LDM Studio — Dataset Management

> **Primary dataset**: `SPRINGLab/IndicVoices-R_Tamil`  
> **Working subset**: 5,000 samples (from 39,292 total)  
> **Audio policy**: ⚠️ The 62.7 GB audio corpus is **never downloaded** locally

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset Facts](#2-dataset-facts)
3. [Module Architecture](#3-module-architecture)
4. [Subset Selection Method](#4-subset-selection-method)
5. [Quality Filters](#5-quality-filters)
6. [Stratification Strategy](#6-stratification-strategy)
7. [Speaker-Disjoint Splitting](#7-speaker-disjoint-splitting)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Output Files](#9-output-files)
10. [Resume Safety](#10-resume-safety)
11. [Audio Access at Training Time](#11-audio-access-at-training-time)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. Overview

The AVA LDM dataset pipeline manages the creation of a curated,
demographically balanced 5,000-sample working subset of the
`SPRINGLab/IndicVoices-R_Tamil` dataset. It handles:

- **HF authentication** (token or environment variable)
- **Dataset inspection** — schema, shard list, card data — without any downloads
- **Streaming metadata access** — only the 24 metadata columns are read from each parquet shard (audio bytes are never requested)
- **Column selection** — keeping only the 12 columns needed by AVA LDM
- **Quality filtering** — duration, SNR, text completeness
- **Stratified sample selection** — multi-dimensional, bias-free
- **Speaker-disjoint splitting** — hard guarantee: no speaker appears in >1 split
- **Local manifest generation** — 4 CSV files written to `data/manifests/`
- **Resume-safe processing** — interrupted runs continue from the last shard

---

## 2. Dataset Facts

| Property | Value |
|---|---|
| HF Repo | `SPRINGLab/IndicVoices-R_Tamil` |
| Language | Tamil (ta) |
| Total examples | 39,292 |
| Parquet shards | 14 (`train-00000-of-00014.parquet` … `train-00013-of-00014.parquet`) |
| Total size | 62.67 GB (audio + metadata) |
| Scenarios | Extempore, Read |
| Districts | Multiple Tamil Nadu districts |
| Gender | Female, Male |
| Age groups | 18-30, 30-45, 45-60, 60+ |
| Area | Rural, Urban |

### Schema (all 24 columns)

| Column | Type | AVA status |
|---|---|---|
| `verbatim` | string | ★ keep |
| `normalized` | string | ★ keep |
| `speaker_id` | string | ★ keep |
| `district` | string | ★ keep |
| `scenario` | ClassLabel | ★ keep |
| `task_name` | string | ★ keep |
| `gender` | ClassLabel | ★ keep |
| `age_group` | ClassLabel | ★ keep |
| `area` | ClassLabel | ★ keep |
| `duration` | float64 | ★ keep |
| `snr` | float64 | ★ keep |
| `audio` | Audio binary | ★ path ref only (no bytes) |
| `text` | string | ✗ drop |
| `lang` | ClassLabel | ✗ drop |
| `samples` | int64 | ✗ drop |
| `job_type` | ClassLabel | ✗ drop |
| `qualification` | ClassLabel | ✗ drop |
| `state` | ClassLabel | ✗ drop |
| `occupation` | string | ✗ drop |
| `utterance_pitch_mean` | float64 | ✗ drop |
| `utterance_pitch_std` | float64 | ✗ drop |
| `c50` | float64 | ✗ drop |
| `speaking_rate` | float64 | ✗ drop |
| `cer` | string | ✗ drop |

---

## 3. Module Architecture

```
src/data/
├── __init__.py
├── dataset_manager.py   # Orchestrator — public API used by all scripts
├── hf_loader.py         # HF Hub access, parquet streaming, class-label decoding
├── metadata_selector.py # Column selection, quality filters, statistics
├── sampler.py           # Stratified proportional sampling
└── splitter.py          # Speaker-disjoint split creation + verification

configs/
└── dataset.yaml         # Single source of truth for all parameters

scripts/
├── inspect_dataset.py   # Inspect schema & shards (no download)
├── build_manifest.py    # Build selected_manifest.csv
└── create_splits.py     # Create train/val/test splits

data/
├── cache/
│   ├── shards/          # Raw shard parquet files (metadata cols only)
│   └── metadata/        # Per-shard processed DataFrames + checkpoint.json
└── manifests/
    ├── selected_manifest.csv
    ├── train.csv
    ├── validation.csv
    └── test.csv

logs/
└── dataset_pipeline.log
```

---

## 4. Subset Selection Method

> [!IMPORTANT]
> The 5,000-row subset is **NOT** the first 5,000 rows of the dataset.
> Selection is performed via multi-dimensional stratified proportional sampling.

### Step-by-step

1. **Stream metadata** from all 14 parquet shards, reading only the 24 non-audio columns. Audio bytes are skipped by the pyarrow parquet reader.

2. **Apply quality filters** (see §5).

3. **Build stratum cells** by forming the Cartesian product of:
   - `gender` (2 values)
   - `age_group` (4 values)
   - `area` (2 values)
   - `scenario` (2 values)
   - `task_name` (variable, typically 10–20 tasks)

   Populated cells are used; empty cells are ignored.

4. **Proportional allocation**: Each stratum cell receives a quota proportional to its natural frequency in the filtered pool. Rounding remainders are distributed to the largest cells.

5. **Within-cell selection**: Within each cell, speakers are enumerated in a deterministic order (sorted by `speaker_id`). A greedy **round-robin** across speakers selects rows — one row per speaker per round — until the cell quota is filled. This maximises speaker diversity within each stratum.

6. **Overflow redistribution**: If a cell has fewer available rows than its quota, the surplus seats are redistributed to cells with the most spare capacity.

7. **Final shuffle**: The concatenated selection is shuffled with the seeded RNG to eliminate any ordering bias from the concat operation.

8. **Reproducibility**: All random operations use `numpy.random.default_rng(seed=42)`. Re-running the pipeline with the same data and seed produces identical output.

---

## 5. Quality Filters

Filters are applied **before** sampling to ensure only clean, usable samples enter the working dataset.

| Filter | Condition | Rationale |
|---|---|---|
| Duration minimum | `duration >= 1.0 s` | Very short clips are unreliable for ASR training |
| Duration maximum | `duration <= 30.0 s` | Very long clips may contain multiple speakers or interruptions |
| SNR floor | `snr >= 5.0 dB` | Clips below this are too noisy for LDM normalisation |
| Non-empty verbatim | `verbatim` is not null/empty | Rows without transcription cannot be used for training |
| Non-empty normalized | `normalized` is not null/empty | Rows without normalised target cannot be used for training |
| Deduplication | unique (`speaker_id`, `verbatim`) | Remove exact repeated recordings |

All filter parameters are configurable in `configs/dataset.yaml → selection.filters`.

---

## 6. Stratification Strategy

The 7 stratification dimensions and their roles:

| Dimension | Role | Cardinality |
|---|---|---|
| `speaker_id` | Maximise speaker diversity within cells | High (hundreds) |
| `gender` | Primary stratum axis | 2 |
| `age_group` | Primary stratum axis | 4 |
| `area` | Primary stratum axis | 2 |
| `district` | Secondary diversity (monitored via Gini) | ~30+ |
| `scenario` | Primary stratum axis | 2 |
| `task_name` | Primary stratum axis | ~10-20 |

> [!NOTE]
> `district` has very high cardinality and is used as a **secondary diversity metric** rather than a primary stratum axis. After sampling, the Gini coefficient across districts is computed. If it exceeds 0.7 (highly skewed), a warning is logged.

---

## 7. Speaker-Disjoint Splitting

> [!IMPORTANT]
> No `speaker_id` appears in more than one split. This is a **hard constraint** verified by assertion after splitting.

### Algorithm

1. For each unique speaker, compute their **demographic profile** (modal value of `gender`, `age_group`, `area`, `scenario`).
2. Group speakers by profile and **shuffle within each group** using the seeded RNG.
3. **Interleave** groups via round-robin to produce a balanced ordered list of speakers.
4. Assign the first 80% of speakers to **train**, next 10% to **val**, remaining to **test** (by speaker count, not utterance count).
5. Map the speaker → split assignment back to individual utterance rows.
6. **Verify disjointness**: assert that `set(train speakers) ∩ set(val speakers) = ∅` etc.

### Target fractions

| Split | Speakers | Utterances (approx.) |
|---|---|---|
| `train` | ~80% | ~80% |
| `validation` | ~10% | ~10% |
| `test` | ~10% | ~10% |

The utterance fractions will not be exactly 80/10/10 because different speakers have different utterance counts.

---

## 8. Running the Pipeline

### Prerequisites

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# (Optional) Log in to Hugging Face for private datasets / higher rate limits
huggingface-cli login
```

### Step 1 — Inspect (no download)

```bash
python scripts/inspect_dataset.py
```

### Step 2 — Build manifest (streams metadata, selects 5,000 rows)

```bash
# Full run (all 14 shards):
python scripts/build_manifest.py

# Test run (first 2 shards only — much faster):
python scripts/build_manifest.py --max-shards 2

# Via datasets library streaming (no shard files cached locally):
python scripts/build_manifest.py --streaming

# Force rebuild from scratch:
python scripts/build_manifest.py --force

# With statistics output:
python scripts/build_manifest.py --stats
```

### Step 3 — Create splits

```bash
python scripts/create_splits.py

# With per-split statistics table:
python scripts/create_splits.py --stats

# Verify existing splits without rebuilding:
python scripts/create_splits.py --verify-only
```

### Full pipeline (3 commands)

```bash
python scripts/inspect_dataset.py
python scripts/build_manifest.py --stats
python scripts/create_splits.py --stats
```

---

## 9. Output Files

### `data/manifests/selected_manifest.csv`

The full 5,000-row working dataset. Columns:

| Column | Description |
|---|---|
| `verbatim` | Original spoken text |
| `normalized` | Normalised reference text |
| `speaker_id` | Speaker identifier |
| `district` | Tamil Nadu district |
| `scenario` | Extempore or Read |
| `task_name` | Fine-grained task label |
| `gender` | Female or Male |
| `age_group` | 18-30, 30-45, 45-60, 60+ |
| `area` | Rural or Urban |
| `duration` | Audio duration in seconds |
| `snr` | Signal-to-noise ratio in dB |
| `audio_hf_path` | `hf://` URI for audio loading |
| `shard_idx` | Source parquet shard index |
| `row_within_shard` | Row position within that shard |
| `stratum_key` | Stratification cell label |
| `selection_rank` | Order of selection (1 = first) |

### `data/manifests/train.csv` / `validation.csv` / `test.csv`

Same columns as `selected_manifest.csv`, minus `stratum_key` and `selection_rank`.

---

## 10. Resume Safety

Shard processing uses a JSON checkpoint file (`data/cache/metadata/checkpoint.json`).

```json
{
  "completed_shards": [0, 1, 2, 3],
  "manifest_rows": 5000,
  "manifest_path": "data/manifests/selected_manifest.csv"
}
```

If the script is interrupted (network error, power loss, keyboard interrupt), re-running `build_manifest.py` automatically:
1. Reads the checkpoint.
2. Loads cached metadata for completed shards from `data/cache/metadata/shard_NNNNN.parquet`.
3. Continues from the next incomplete shard.

Use `--force` to ignore the checkpoint and start over.

---

## 11. Audio Access at Training Time

Audio bytes are **never stored locally**. Instead, each row in the manifest contains an `audio_hf_path` column with a URI like:

```
hf://datasets/SPRINGLab/IndicVoices-R_Tamil/data/train-00003-of-00014.parquet
```

At training time, a data loader loads individual audio clips on demand:

```python
import torchaudio
from datasets import load_dataset

# Load a single row's audio via HF streaming (only that clip is downloaded)
ds = load_dataset(
    "SPRINGLab/IndicVoices-R_Tamil",
    split="train",
    streaming=True,
)
# Filter to the specific rows listed in the manifest
```

Or use the `row_within_shard` + `shard_idx` to fetch individual rows efficiently via the HF Parquet API.

---

## 12. Configuration Reference

All parameters live in [`configs/dataset.yaml`](../configs/dataset.yaml).

```yaml
dataset:
  target_n: 5000          # Working subset size
  random_seed: 42         # Reproducibility seed

selection:
  filters:
    min_duration: 1.0     # seconds
    max_duration: 30.0    # seconds
    min_snr: 5.0          # dB

splitting:
  train_frac: 0.80
  val_frac:   0.10
  test_frac:  0.10
```

> [!TIP]
> Change `random_seed` and rerun to produce a different but equally valid stratified subset for ablation studies.
