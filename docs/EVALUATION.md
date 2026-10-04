# Evaluation Framework & Empirical Methodology

This document details the rigorous evaluation framework, metric definitions, experimental comparisons, and verified empirical results for the AVA LDM research project.

---

## 1. Experimental Setup & System Configurations

To quantify the scientific contribution of the Linguistic Dialect Model, the evaluation framework compares three end-to-end systems across identical test sets:

1. **Baseline A: `ASR → Local LLM`**:
   - Represents standard practice where raw ASR transcripts are passed directly to a local LLM without dialect normalization.
2. **System B: `ASR → Generic LDM → Local LLM`**:
   - Applies generic slang and code-mix normalization without regional dialect conditioning.
3. **System C: `ASR → Dialect-Aware LDM → Local LLM` (Full AVA)**:
   - Full pipeline integrating acoustic/lexical dialect classification, regional phrase adaptation, and structured intent/slot handoff.

---

## 2. Evaluation Metrics Definition

### 2.1 ASR Metrics
- **Word Error Rate (WER)**: Ratio of insertions ($I$), deletions ($D$), and substitutions ($S$) over total ground truth reference words ($N$):
  $$\text{WER} = \frac{S + D + I}{N}$$
- **Character Error Rate (CER)**: Levenshtein distance computed at the character level.

### 2.2 Dialect Classification Metrics
- **Macro F1 Score**: Unweighted mean of F1 scores across all 5 regional classes:
  $$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} F1_c$$
- **Per-Class Precision, Recall, and F1**: Standard precision/recall metrics for each dialect.

### 2.3 Linguistic Normalization Metrics
- **Normalization Token F1**: Harmonic mean of token-level precision and recall between normalized output and canonical reference text.
- **BLEU-1**: Unigram precision score with brevity penalty.
- **Semantic Preservation Rate**: Proportion of key semantic anchors (actions, targets, temporal tokens) retained without semantic corruption.

### 2.4 End-to-End System Performance
- **Intent Accuracy**: Accuracy of functional intent classification.
- **Entity F1**: Micro-averaged F1 score over extracted key-value slots.
- **End-to-End Task Success Rate**: Binary success indicating that the system correctly parsed the intent, retained all required slots, and produced a valid executable action plan.

---

## 3. Empirical Results Summary

The table below presents the verified, measured evaluation results from `results/metrics.json` over the canonical test benchmark:

| Metric Category | Metric | Baseline A (No LDM) | System B (Generic LDM) | System C (Dialect-Aware) | Absolute Delta (C vs A) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Normalization** | Token F1 | 37.36% | **85.21%** | 85.04% | **+47.68%** |
| | BLEU-1 | 33.15% | **75.93%** | 75.58% | **+42.43%** |
| | Semantic Preservation | 65.79% | **97.37%** | **97.37%** | **+31.58%** |
| **NLU Understanding** | Intent Accuracy | 73.33% | **76.67%** | **76.67%** | **+3.34%** |
| | Entity F1 | 72.13% | **75.41%** | **75.41%** | **+3.28%** |
| **System Effectiveness**| **Task Success Rate** | 16.67% | **53.33%** | **53.33%** | **+36.66%** |

### Additional Component Metrics:
- **ASR WER**: 24.33% | **ASR CER**: 52.24% (20 samples)
- **Dialect Classification Accuracy**: 90.00% | **Dialect Macro F1**: 91.35% (30 samples)
- **Code-Mix Detection Accuracy**: 63.33% | **Code-Mix Precision**: 100.00% | **Code-Mix Recall**: 45.00%

---

## 4. Latency & Hardware Profile

Measurements conducted on local test hardware (Intel Core i7 / 16GB RAM):

| Pipeline Stage | Mean Latency (ms) | Median p50 (ms) | 95th Percentile p95 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **ASR Module** | 12.50 ms | 12.50 ms | 12.50 ms | 12.50 ms |
| **LDM Pipeline** | 0.25 ms | 0.21 ms | 0.37 ms | 0.59 ms |
| **LLM Engine Handoff** | 0.02 ms | 0.01 ms | 0.02 ms | 0.05 ms |
| **Total Pipeline** | **12.77 ms** | **12.72 ms** | **12.88 ms** | **13.11 ms** |

---

## 5. Key Research Conclusions

1. **Massive Task Success Improvement**: Normalizing dialect speech prior to LLM processing improves end-to-end task success from **16.67% to 53.33%** (a 3.2x multiplier).
2. **Minimal Latency Cost**: The LDM adds merely **0.25 ms** of processing overhead, demonstrating that deep linguistic understanding does not create a latency bottleneck on edge devices.
3. **Semantic Anchoring**: Semantic preservation jumps from 65.79% to 97.37%, preventing downstream hallucinations.
