"""
src/evaluation/dialect_metrics.py
=================================
Dialect classification evaluation metrics:
- Overall Accuracy
- Macro F1
- Per-class Precision, Recall, F1
- Confusion matrix calculation and pure-Python PNG heatmap rendering.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_label(label: str) -> str:
    """Normalize dialect label variations."""
    l = label.strip().lower()
    if "chennai" in l:
        return "Chennai"
    if "madurai" in l:
        return "Madurai"
    if "kongu" in l:
        return "Kongu"
    if "nellai" in l or "tirunelveli" in l:
        return "Nellai"
    if "standard" in l or "indian english" in l:
        return "Standard"
    return label.strip()


class DialectEvaluator:
    """Evaluates dialect classification predictions against ground truth."""

    DEFAULT_CLASSES = ["Chennai", "Madurai", "Kongu", "Nellai", "Standard"]

    def __init__(self, classes: Optional[List[str]] = None) -> None:
        self.classes = classes or self.DEFAULT_CLASSES

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Compute Accuracy, Macro F1, Per-class metrics, and Confusion Matrix.
        """
        if not y_true or len(y_true) != len(y_pred):
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "per_class": {},
                "confusion_matrix": {},
                "sample_count": 0,
            }

        norm_true = [_normalize_label(y) for y in y_true]
        norm_pred = [_normalize_label(y) for y in y_pred]

        # 1. Confusion Matrix
        # matrix[actual][predicted]
        cm: Dict[str, Dict[str, int]] = {
            c_true: {c_pred: 0 for c_pred in self.classes} for c_true in self.classes
        }

        correct = 0
        n = len(norm_true)

        for t, p in zip(norm_true, norm_pred):
            if t in cm and p in cm[t]:
                cm[t][p] += 1
            if t == p:
                correct += 1

        accuracy = round(correct / n, 4) if n > 0 else 0.0

        # 2. Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        f1_scores = []

        for c in self.classes:
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in self.classes if other != c)
            fn = sum(cm[c][other] for other in self.classes if other != c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
            if (tp + fn) > 0 or (tp + fp) > 0:
                f1_scores.append(f1)

        macro_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": cm,
            "sample_count": n,
        }

    def render_confusion_matrix_png(
        self,
        cm: Dict[str, Dict[str, int]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Renders the confusion matrix as a high-contrast heatmap PNG using pure Python.
        Requires zero third-party GUI or graphics libraries.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        classes = self.classes
        n_cls = len(classes)
        cell_size = 80
        pad_left = 120
        pad_top = 80
        pad_right = 40
        pad_bottom = 60

        width = pad_left + n_cls * cell_size + pad_right
        height = pad_top + n_cls * cell_size + pad_bottom

        # Find maximum value for normalization
        max_val = max(max(row.values()) for row in cm.values()) if cm else 1
        if max_val == 0:
            max_val = 1

        # Background color: dark slate (#1e293b / RGB 30, 41, 59)
        bg_color = (30, 41, 59)
        grid_color = (71, 85, 105)

        # Create RGB canvas
        canvas = bytearray(width * height * 3)
        for i in range(width * height):
            canvas[i * 3 : i * 3 + 3] = bytes(bg_color)

        def set_pixel(x: int, y: int, color: Tuple[int, int, int]):
            if 0 <= x < width and 0 <= y < height:
                idx = (y * width + x) * 3
                canvas[idx : idx + 3] = bytes(color)

        def draw_rect(x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
            for cy in range(y1, y2):
                for cx in range(x1, x2):
                    set_pixel(cx, cy, color)

        # Draw Heatmap Cells
        for r_idx, true_cls in enumerate(classes):
            for c_idx, pred_cls in enumerate(classes):
                val = cm[true_cls][pred_cls]
                ratio = val / max_val

                # Heatmap Color Interpolation:
                # Dark navy (26, 44, 76) -> Teal/Cyan (14, 165, 233) -> Bright White-Cyan (224, 242, 254)
                if ratio == 0:
                    color = (40, 53, 75)
                else:
                    r = int(14 + (224 - 14) * ratio)
                    g = int(100 + (242 - 100) * ratio)
                    b = int(180 + (254 - 180) * ratio)
                    color = (r, g, b)

                x1 = pad_left + c_idx * cell_size
                y1 = pad_top + r_idx * cell_size
                x2 = x1 + cell_size - 2
                y2 = y1 + cell_size - 2

                draw_rect(x1, y1, x2, y2, color)

                # Draw diagonal highlight for true positives
                if true_cls == pred_cls:
                    for px in range(x1, x2):
                        set_pixel(px, y1, (56, 189, 248))
                        set_pixel(px, y2 - 1, (56, 189, 248))
                    for py in range(y1, y2):
                        set_pixel(x1, py, (56, 189, 248))
                        set_pixel(x2 - 1, py, (56, 189, 248))

        # Write to PNG file
        self._write_png(str(output_path), width, height, bytes(canvas))

    @staticmethod
    def _write_png(filename: str, width: int, height: int, rgb_data: bytes) -> None:
        """Pure-Python PNG writer with zlib compression."""
        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        raw_scanlines = bytearray()
        row_len = width * 3
        for y in range(height):
            raw_scanlines.append(0)  # filter type 0 (None)
            raw_scanlines.extend(rgb_data[y * row_len : (y + 1) * row_len])

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw_scanlines), level=6))
            + chunk(b"IEND", b"")
        )

        with open(filename, "wb") as f:
            f.write(png)
