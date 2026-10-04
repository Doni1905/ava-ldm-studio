"""
src/evaluation/asr_metrics.py
=============================
ASR evaluation metrics: Word Error Rate (WER) and Character Error Rate (CER).

Calculates exact Levenshtein alignment between reference and prediction
without requiring external C/binary dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _normalize_text(text: str) -> str:
    """Normalize text for consistent ASR evaluation."""
    if not text:
        return ""
    # Lowercase and remove extra whitespace
    text = text.lower().strip()
    # Strip common punctuation
    text = re.sub(r"[^\w\s\u0B80-\u0BFF]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance and operation counts (Substitutions, Deletions, Insertions).
    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(seq1), len(seq2)
    # DP table: (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # i deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost, s, d, ins = dp[i - 1][j - 1]
                del_cost, ds, dd, dins = dp[i - 1][j]
                ins_cost, is_, id_, iins = dp[i][j - 1]

                choice_sub = (sub_cost + 1, s + 1, d, ins)
                choice_del = (del_cost + 1, ds, dd + 1, dins)
                choice_ins = (ins_cost + 1, is_, id_, iins + 1)

                dp[i][j] = min(choice_sub, choice_del, choice_ins, key=lambda x: x[0])

    return dp[n][m]


def compute_wer(reference: str, prediction: str) -> float:
    """
    Compute Word Error Rate (WER):
        WER = (S + D + I) / N_ref
    """
    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(prediction).split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    dist, _, _, _ = _levenshtein_distance(ref_tokens, hyp_tokens)
    return round(dist / len(ref_tokens), 4)


def compute_cer(reference: str, prediction: str) -> float:
    """
    Compute Character Error Rate (CER):
        CER = (S + D + I) / N_ref
    """
    ref_chars = list(_normalize_text(reference).replace(" ", ""))
    hyp_chars = list(_normalize_text(prediction).replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist, _, _, _ = _levenshtein_distance(ref_chars, hyp_chars)
    return round(dist / len(ref_chars), 4)


def evaluate_asr_batch(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    """
    Evaluate ASR metrics over a batch of sample pairs.
    """
    if not references or len(references) != len(predictions):
        return {"wer": 0.0, "cer": 0.0, "samples_evaluated": 0}

    total_wer = 0.0
    total_cer = 0.0
    n = len(references)

    for ref, hyp in zip(references, predictions):
        total_wer += compute_wer(ref, hyp)
        total_cer += compute_cer(ref, hyp)

    return {
        "wer": round(total_wer / n, 4),
        "cer": round(total_cer / n, 4),
        "samples_evaluated": n,
    }
