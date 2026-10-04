"""
src/evaluation/ldm_metrics.py
=============================
LDM evaluation metrics:
- Normalization Accuracy (Exact Match, Token F1, BLEU-1, Character Similarity)
- Intent Classification Accuracy & Macro F1
- Code-Mix Detection Accuracy, Precision, Recall, F1
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    """Tokenize lowercase words and punctuation."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if w]


def compute_token_f1(reference: str, prediction: str) -> float:
    """
    Compute token-level F1 overlap between reference and predicted normalization.
    """
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)

    if not ref_toks and not pred_toks:
        return 1.0
    if not ref_toks or not pred_toks:
        return 0.0

    ref_counts = collections.Counter(ref_toks)
    pred_counts = collections.Counter(pred_toks)

    common = sum((ref_counts & pred_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(ref_toks)
    return round(2 * precision * recall / (precision + recall), 4)


def compute_bleu1(reference: str, prediction: str) -> float:
    """
    Compute unigram BLEU with brevity penalty.
    """
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)

    if not pred_toks or not ref_toks:
        return 0.0

    pred_counts = collections.Counter(pred_toks)
    ref_counts = collections.Counter(ref_toks)

    clipped_matches = sum(min(count, ref_counts[w]) for w, count in pred_counts.items())
    precision = clipped_matches / len(pred_toks)

    # Brevity penalty
    c = len(pred_toks)
    r = len(ref_toks)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0

    return round(bp * precision, 4)


def compute_char_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized character-level Levenshtein similarity in [0, 1].
    """
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()

    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    dist = dp[n][m]
    max_len = max(n, m)
    return round(1.0 - (dist / max_len), 4)


def _canonical_intent(intent: str) -> str:
    """Map intent variations between dataset.ts and YAML/pipeline formats."""
    mapping = {
        "set_reminder": "CREATE_REMINDER",
        "create_reminder": "CREATE_REMINDER",
        "remind": "CREATE_REMINDER",
        "set_alarm": "CREATE_REMINDER",  # Both represent reminder/alarm alerts
        "make_call": "MAKE_CALL",
        "call": "MAKE_CALL",
        "send_message": "SEND_MESSAGE",
        "message": "SEND_MESSAGE",
        "open_app": "OPEN_APP",
        "launch_app": "OPEN_APP",
        "device_control": "DEVICE_SETTING",
        "device_setting": "DEVICE_SETTING",
        "check_weather": "GENERAL_QUERY",
        "navigate": "GENERAL_QUERY",
        "play_music": "GENERAL_QUERY",
        "search_info": "GENERAL_QUERY",
        "smalltalk": "GENERAL_QUERY",
        "general_query": "GENERAL_QUERY",
    }
    key = intent.strip().lower().replace(" ", "_")
    return mapping.get(key, intent.strip().upper())


class LDMEvaluator:
    """Evaluates Linguistic Normalization, Intent Accuracy, and Code-Mix detection."""

    @staticmethod
    def evaluate_normalization(
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate text normalization quality against reference normalized sentences.
        """
        if not references or len(references) != len(predictions):
            return {"exact_match": 0.0, "token_f1": 0.0, "bleu1": 0.0, "char_similarity": 0.0}

        n = len(references)
        exact_matches = 0
        total_token_f1 = 0.0
        total_bleu = 0.0
        total_char_sim = 0.0

        for ref, pred in zip(references, predictions):
            norm_ref = ref.strip().lower().rstrip(".!?")
            norm_pred = pred.strip().lower().rstrip(".!?")

            if norm_ref == norm_pred:
                exact_matches += 1

            total_token_f1 += compute_token_f1(ref, pred)
            total_bleu += compute_bleu1(ref, pred)
            total_char_sim += compute_char_similarity(ref, pred)

        return {
            "exact_match_rate": round(exact_matches / n, 4),
            "token_f1": round(total_token_f1 / n, 4),
            "bleu1": round(total_bleu / n, 4),
            "char_similarity": round(total_char_sim / n, 4),
        }

    @staticmethod
    def evaluate_intents(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Evaluate intent classification accuracy and macro F1.
        """
        if not y_true or len(y_true) != len(y_pred):
            return {"intent_accuracy": 0.0, "macro_f1": 0.0}

        true_canon = [_canonical_intent(y) for y in y_true]
        pred_canon = [_canonical_intent(y) for y in y_pred]

        correct = sum(1 for t, p in zip(true_canon, pred_canon) if t == p)
        n = len(true_canon)
        acc = round(correct / n, 4) if n > 0 else 0.0

        # Unique classes
        classes = sorted(list(set(true_canon) | set(pred_canon)))
        f1_list = []
        for c in classes:
            tp = sum(1 for t, p in zip(true_canon, pred_canon) if t == c and p == c)
            fp = sum(1 for t, p in zip(true_canon, pred_canon) if t != c and p == c)
            fn = sum(1 for t, p in zip(true_canon, pred_canon) if t == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        macro_f1 = round(sum(f1_list) / len(f1_list), 4) if f1_list else 0.0

        return {"intent_accuracy": acc, "intent_macro_f1": macro_f1}

    @staticmethod
    def evaluate_code_mix(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
        """
        Evaluate code-mix binary detection (Accuracy, Precision, Recall, F1).
        """
        if not y_true or len(y_true) != len(y_pred):
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is True)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is False)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t is False and p is True)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t is True and p is False)

        total = len(y_true)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
