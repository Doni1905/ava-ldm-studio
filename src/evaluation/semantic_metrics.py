"""
src/evaluation/semantic_metrics.py
==================================
Semantic preservation and End-to-End Task Success metrics:
- Semantic Anchor Preservation Rate
- Entity Precision, Recall, and F1
- End-to-End Task Success Rate across Baseline A, System B, and System C
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.ldm.semantic_preserver import SemanticPreserver


class SemanticEvaluator:
    """Evaluates entity survival, semantic integrity, and task success."""

    def __init__(self, preserver: Optional[SemanticPreserver] = None) -> None:
        self.preserver = preserver or SemanticPreserver.default()

    @classmethod
    def default(cls) -> "SemanticEvaluator":
        return cls()

    def evaluate_preservation_batch(
        self,
        inputs: List[str],
        normalized_texts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate semantic anchor preservation across a batch of utterances.
        Checks what fraction of extracted anchors from the input survived in the output.
        """
        if not inputs or len(inputs) != len(normalized_texts):
            return {"preservation_rate": 0.0, "total_anchors": 0, "anchors_preserved": 0}

        total_anchors = 0
        preserved_anchors = 0

        for inp, norm in zip(inputs, normalized_texts):
            extracted = self.preserver.extract(inp)
            report = self.preserver.verify(extracted, norm)

            for anchor in extracted.anchors:
                total_anchors += 1
                if anchor.value.lower() in norm.lower():
                    preserved_anchors += 1

        rate = (preserved_anchors / total_anchors) if total_anchors > 0 else 1.0

        return {
            "preservation_rate": round(rate, 4),
            "total_anchors": total_anchors,
            "anchors_preserved": preserved_anchors,
        }

    def evaluate_entities(
        self,
        extracted_entities_list: List[Dict[str, str]],
        reference_normalized_list: List[str],
    ) -> Dict[str, float]:
        """
        Measure entity extraction accuracy against reference semantics.
        """
        tp, fp, fn = 0, 0, 0

        for pred_entities, ref_norm in zip(extracted_entities_list, reference_normalized_list):
            ref_extracted = self.preserver.extract(ref_norm)
            ground_truth_vals = {a.value.lower() for a in ref_extracted.anchors}
            pred_vals = {str(v).lower() for v in pred_entities.values() if v}

            current_tp = len(pred_vals & ground_truth_vals)
            current_fp = len(pred_vals - ground_truth_vals)
            current_fn = len(ground_truth_vals - pred_vals)

            tp += current_tp
            fp += current_fp
            fn += current_fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "entity_precision": round(precision, 4),
            "entity_recall": round(recall, 4),
            "entity_f1": round(f1, 4),
        }

    @staticmethod
    def evaluate_task_success(
        predicted_intents: List[str],
        reference_intents: List[str],
        extracted_entities_list: List[Dict[str, str]],
        llm_responses: List[str],
        raw_inputs: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate whether the end-to-end user request succeeded.

        Success Criteria:
        1. Correct intent understanding.
        2. Key required entities (date, person, task, app) captured and reflected in the assistant response.
        3. LLM response is coherent and not hallucinating an unrelated refusal or misunderstanding.
        """
        n = len(raw_inputs)
        if n == 0:
            return {"task_success_rate": 0.0, "success_count": 0, "failure_count": 0}

        successes = 0
        failure_reasons = []

        for i in range(n):
            pred_intent = predicted_intents[i]
            ref_intent = reference_intents[i]
            entities = extracted_entities_list[i]
            resp = llm_responses[i].lower()
            inp = raw_inputs[i].lower()

            # 1. Intent check
            from .ldm_metrics import _canonical_intent
            intent_ok = (_canonical_intent(pred_intent) == _canonical_intent(ref_intent))

            # 2. Entity grounding in response
            # At least one key entity must appear in the assistant action, if entities exist
            entity_ok = True
            if entities:
                entity_ok = any(str(val).lower() in resp for val in entities.values() if val)

            # 3. No raw dialect hallucination (e.g. LLM confusing 'dei' or 'machi' for names)
            slang_hallucinated = any(w in resp for w in ["dei", "machi", "pannu", "da", "la", "kudu"])

            if intent_ok and entity_ok and not slang_hallucinated:
                successes += 1
            else:
                failure_reasons.append({
                    "sample_idx": i,
                    "input": inp,
                    "intent_ok": intent_ok,
                    "entity_ok": entity_ok,
                    "slang_hallucinated": slang_hallucinated,
                })

        rate = round(successes / n, 4)
        return {
            "task_success_rate": rate,
            "success_count": successes,
            "failure_count": n - successes,
            "failures": failure_reasons,
        }
