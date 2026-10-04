"""
src/dialect/evaluator.py
=========================
Evaluation engine for the Tamil dialect classifier.

Metrics reported
----------------
- Overall accuracy
- Macro F1 (unweighted mean F1 across all dialect classes)
- Per-class precision, recall, F1
- Confusion matrix (as nested list and as a printable string)
- Per-sample prediction log saved to CSV

All metrics are computed using sklearn to avoid any numerical
inconsistencies from manual implementations.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .classifier import DialectClassifier
from .labels import DialectLabelRegistry
from .trainer import DialectAudioDataset

logger = logging.getLogger(__name__)


class DialectEvaluator:
    """
    Run evaluation of a trained DialectClassifier on a test manifest.

    Parameters
    ----------
    config : dict
        Parsed ``configs/dialect.yaml``.
    label_registry : DialectLabelRegistry
    """

    def __init__(
        self,
        config: Dict[str, Any],
        label_registry: DialectLabelRegistry,
    ) -> None:
        self.config   = config
        self.registry = label_registry
        eval_cfg      = config.get("evaluation", {})
        self.batch_size = eval_cfg.get("batch_size", 16)

        device_str = config.get("inference", {}).get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

    def evaluate(
        self,
        model: DialectClassifier,
        manifest_path: str | Path,
        audio_col: str = "file",
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a manifest CSV.

        Returns a dict containing:
        - accuracy
        - macro_f1
        - per_class: { label: {precision, recall, f1, support} }
        - confusion_matrix: list[list[int]]
        - confusion_matrix_str: human-readable string
        """
        df = pd.read_csv(manifest_path)
        dataset = DialectAudioDataset(
            manifest_df=df,
            label_registry=self.registry,
            target_sr=self.config.get("model", {}).get("target_sample_rate", 16000),
            max_length=self.config.get("training", {}).get("max_audio_length", 80000),
            audio_col=audio_col,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=DialectAudioDataset.collate_fn,
        )

        all_preds, all_labels, all_probs = self._run_inference(model, loader)

        # ---- Sklearn metrics ----
        label_names = self.registry.names
        acc  = accuracy_score(all_labels, all_preds)
        mf1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm   = confusion_matrix(all_labels, all_preds, labels=list(range(self.registry.num_classes)))
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        cm_str = self._format_confusion_matrix(cm, label_names)

        per_class = {
            lbl: {
                "precision": report_dict[lbl]["precision"],
                "recall":    report_dict[lbl]["recall"],
                "f1":        report_dict[lbl]["f1-score"],
                "support":   report_dict[lbl]["support"],
            }
            for lbl in label_names
            if lbl in report_dict
        }

        result = {
            "accuracy":          round(acc, 4),
            "macro_f1":          round(mf1, 4),
            "per_class":         per_class,
            "confusion_matrix":  cm.tolist(),
            "confusion_matrix_str": cm_str,
            "n_samples":         len(all_labels),
        }

        # ---- Save artefacts ----
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(result, all_preds, all_labels, all_probs, df, output_dir)

        logger.info(
            f"Evaluation complete: accuracy={acc:.4f} macro_f1={mf1:.4f} n={len(all_labels)}"
        )
        logger.info("\n" + cm_str)

        return result

    @torch.no_grad()
    def _run_inference(
        self,
        model: DialectClassifier,
        loader: DataLoader,
    ):
        model.eval()
        model.to(self.device)

        all_preds:  List[int]   = []
        all_labels: List[int]   = []
        all_probs:  List[List[float]] = []

        for waveform, attn_mask, labels, _ in loader:
            waveform  = waveform.to(self.device)
            attn_mask = attn_mask.to(self.device)

            probs = model.predict_proba(waveform, attn_mask).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

        return all_preds, all_labels, all_probs

    def _format_confusion_matrix(self, cm: np.ndarray, labels: List[str]) -> str:
        """Return a readable ASCII confusion matrix."""
        width = max(len(l) for l in labels) + 2
        header = "Pred→ " + " ".join(f"{l:>{width}}" for l in labels)
        rows   = [f"True↓ {header}"]
        for i, row in enumerate(cm):
            cells = " ".join(f"{v:>{width}}" for v in row)
            rows.append(f"  {labels[i]:<{width}} {cells}")
        return "\n".join(rows)

    def _save_results(
        self,
        result: Dict,
        preds: List[int],
        labels: List[int],
        probs: List[List[float]],
        df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        """Save evaluation artefacts: summary JSON, per-sample CSV, confusion matrix."""
        # Summary JSON
        summary = {k: v for k, v in result.items() if k != "confusion_matrix_str"}
        with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Confusion matrix text
        with open(out_dir / "confusion_matrix.txt", "w", encoding="utf-8") as f:
            f.write(result["confusion_matrix_str"])

        # Per-sample predictions
        rows = []
        for i, (pred_idx, label_idx) in enumerate(zip(preds, labels)):
            prob_vec = probs[i]
            top_prob = max(prob_vec)
            pred_lbl = self.registry.idx_to_label(pred_idx)
            true_lbl = self.registry.idx_to_label(label_idx)
            rows.append({
                "sample_id":      df.get("sample_id", df.index)[i] if i < len(df) else i,
                "true_dialect":   true_lbl,
                "pred_dialect":   pred_lbl,
                "correct":        pred_lbl == true_lbl,
                "confidence":     round(top_prob, 4),
                **{f"prob_{self.registry.idx_to_label(j)}": round(p, 4)
                   for j, p in enumerate(prob_vec)},
            })
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

        logger.info(f"Saved evaluation artefacts to {out_dir}")
