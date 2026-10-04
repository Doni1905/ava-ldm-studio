"""
scripts/test_pipeline_smoke.py
================================
Smoke test for the AVA LDM dataset pipeline modules.
Uses synthetic data — no network access required.

Run from project root:
    python scripts/test_pipeline_smoke.py
"""
import io
import sys
import pathlib

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import pandas as pd

from src.data.metadata_selector import MetadataSelector
from src.data.sampler import StratifiedSampler
from src.data.splitter import SpeakerDisjointSplitter

# ------------------------------------------------------------------
# Build synthetic dataset (1 000 rows, 50 speakers)
# ------------------------------------------------------------------
rng = np.random.default_rng(99)
N = 1000
speakers  = [f"spk_{i:03d}" for i in range(50)]
genders   = ["Female", "Male"]
ages      = ["18-30", "30-45", "45-60", "60+"]
areas     = ["Rural", "Urban"]
scenarios = ["Extempore", "Read"]
tasks     = ["task_A", "task_B", "task_C", "task_D", "task_E"]
districts = [f"district_{d}" for d in range(10)]

df = pd.DataFrame({
    "speaker_id":       rng.choice(speakers, N),
    "gender":           rng.choice(genders, N),
    "age_group":        rng.choice(ages, N),
    "area":             rng.choice(areas, N),
    "scenario":         rng.choice(scenarios, N),
    "task_name":        rng.choice(tasks, N),
    "district":         rng.choice(districts, N),
    "verbatim":         [f"utterance {i}" for i in range(N)],
    "normalized":       [f"normalized {i}" for i in range(N)],
    "duration":         rng.uniform(1.5, 20.0, N),
    "snr":              rng.uniform(8.0, 40.0, N),
    "audio_hf_path": [
        "hf://datasets/SPRINGLab/IndicVoices-R_Tamil/data/train-00000-of-00014.parquet"
    ] * N,
    "shard_idx":         [0] * N,
    "row_within_shard":  list(range(N)),
})

# Load real config
cfg_path = ROOT / "configs" / "dataset.yaml"
with open(cfg_path, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

# ------------------------------------------------------------------
# Test 1: MetadataSelector
# ------------------------------------------------------------------
print("=== MetadataSelector ===")
sel = MetadataSelector(cfg)
filtered = sel.select_and_filter(df)
print(f"  Input rows   : {len(df)}")
print(f"  Filtered rows: {len(filtered)}")
assert len(filtered) <= len(df), "Filtered should be <= input"
assert "verbatim" in filtered.columns, "verbatim missing"
assert "snr" in filtered.columns, "snr missing"
assert "audio" not in filtered.columns, "audio binary should be excluded"

stats = sel.compute_statistics(filtered)
assert stats["total_rows"] == len(filtered)
assert "gender_distribution" in stats
print("  PASS\n")

# ------------------------------------------------------------------
# Test 2: StratifiedSampler
# ------------------------------------------------------------------
print("=== StratifiedSampler ===")
sampler = StratifiedSampler(cfg)
sampler.target_n = 200  # smaller target for test speed

sampled = sampler.sample(filtered)
print(f"  Target: 200  | Selected: {len(sampled)}")
print(f"  Unique speakers : {sampled['speaker_id'].nunique()}")
print(f"  Gender dist     : {sampled['gender'].value_counts().to_dict()}")
print(f"  Area dist       : {sampled['area'].value_counts().to_dict()}")
print(f"  Scenario dist   : {sampled['scenario'].value_counts().to_dict()}")

assert len(sampled) <= 200, "Should not exceed target_n"
assert "stratum_key" in sampled.columns, "stratum_key column missing"
assert "selection_rank" in sampled.columns, "selection_rank column missing"

# Verify NOT simply the first 200 rows of filtered
first_200_verbatims = set(filtered["verbatim"].iloc[:200])
sampled_verbatims   = set(sampled["verbatim"])
overlap = len(first_200_verbatims & sampled_verbatims)
print(f"  Overlap with first-200 rows: {overlap}/200")
assert overlap < 200, "Selection is naively first-N — stratification not working"
print("  PASS\n")

# ------------------------------------------------------------------
# Test 3: SpeakerDisjointSplitter
# ------------------------------------------------------------------
print("=== SpeakerDisjointSplitter ===")
splitter = SpeakerDisjointSplitter(cfg)
splits = splitter.split(sampled)

train_spk = splits["train"]["speaker_id"].nunique()
val_spk   = splits["val"]["speaker_id"].nunique()
test_spk  = splits["test"]["speaker_id"].nunique()

print(f"  train : {len(splits['train'])} rows, {train_spk} speakers")
print(f"  val   : {len(splits['val'])} rows, {val_spk} speakers")
print(f"  test  : {len(splits['test'])} rows, {test_spk} speakers")

total = sum(len(v) for v in splits.values())
assert total == len(sampled), f"Row count mismatch: {total} != {len(sampled)}"

# Hard verification (raises AssertionError on violation)
splitter.verify_disjoint(splits)

# Manual double-check
all_train = set(splits["train"]["speaker_id"])
all_val   = set(splits["val"]["speaker_id"])
all_test  = set(splits["test"]["speaker_id"])
assert not (all_train & all_val),  f"train-val speaker overlap"
assert not (all_train & all_test), f"train-test speaker overlap"
assert not (all_val   & all_test), f"val-test speaker overlap"

print("  Speaker disjointness: VERIFIED")
print("  PASS\n")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("=" * 55)
print("  ALL PIPELINE TESTS PASSED")
print("=" * 55)
