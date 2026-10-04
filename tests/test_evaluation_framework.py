"""
tests/test_evaluation_framework.py
==================================
Unit tests for the M12 Evaluation & Benchmarking suite:
- ASR metrics (WER, CER)
- Dialect classification metrics & PNG heatmap rendering
- LDM normalization, intent, and code-mix metrics
- Semantic preservation and Task Success evaluation
- Report generator (JSON, CSV, Markdown)
"""

import io
import sys
import unittest
from pathlib import Path

# Safe UTF-8 reconfiguration
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.asr_metrics import compute_cer, compute_wer, evaluate_asr_batch
from src.evaluation.dialect_metrics import DialectEvaluator
from src.evaluation.latency import LatencyProfiler, PipelineLatencyRecord
from src.evaluation.ldm_metrics import LDMEvaluator, compute_bleu1, compute_char_similarity, compute_token_f1
from src.evaluation.report import ReportGenerator
from src.evaluation.semantic_metrics import SemanticEvaluator


class TestASRMetrics(unittest.TestCase):
    def test_wer_identical(self):
        self.assertEqual(compute_wer("hello world", "hello world"), 0.0)

    def test_wer_substitution(self):
        # 1 substitution out of 2 words -> 0.50
        self.assertEqual(compute_wer("hello world", "hello there"), 0.5)

    def test_cer_identical(self):
        self.assertEqual(compute_cer("apple", "apple"), 0.0)

    def test_cer_difference(self):
        # 'apply' vs 'apple' (1 char edit out of 5)
        self.assertEqual(compute_cer("apple", "apply"), 0.2)

    def test_batch_evaluation(self):
        res = evaluate_asr_batch(["one two", "three four"], ["one two", "three five"])
        self.assertIn("wer", res)
        self.assertIn("cer", res)
        self.assertEqual(res["samples_evaluated"], 2)


class TestDialectMetrics(unittest.TestCase):
    def setUp(self):
        self.evaluator = DialectEvaluator(classes=["Chennai", "Madurai", "Kongu", "Nellai", "Standard"])

    def test_perfect_accuracy(self):
        y_true = ["Chennai", "Madurai", "Kongu"]
        y_pred = ["Chennai", "Madurai", "Kongu"]
        res = self.evaluator.evaluate(y_true, y_pred)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["macro_f1"], 1.0)

    def test_confusion_matrix_shape(self):
        y_true = ["Chennai", "Madurai"]
        y_pred = ["Chennai", "Kongu"]
        res = self.evaluator.evaluate(y_true, y_pred)
        cm = res["confusion_matrix"]
        self.assertEqual(cm["Chennai"]["Chennai"], 1)
        self.assertEqual(cm["Madurai"]["Kongu"], 1)

    def test_png_rendering(self):
        cm = {c: {c2: 1 for c2 in self.evaluator.classes} for c in self.evaluator.classes}
        test_png = _ROOT / "results" / "test_cm.png"
        self.evaluator.render_confusion_matrix_png(cm, test_png)

        self.assertTrue(test_png.exists())
        # Verify valid PNG header (\x89PNG\r\n\x1a\n)
        header = test_png.read_bytes()[:8]
        self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

        # Clean up
        if test_png.exists():
            test_png.unlink()


class TestLDMMetrics(unittest.TestCase):
    def test_token_f1(self):
        # Identical
        self.assertEqual(compute_token_f1("hello world", "hello world"), 1.0)
        # 50% overlap
        f1 = compute_token_f1("hello world", "hello there")
        self.assertEqual(f1, 0.5)

    def test_bleu1(self):
        self.assertEqual(compute_bleu1("turn on wifi", "turn on wifi"), 1.0)
        self.assertEqual(compute_bleu1("turn on wifi", "completely different"), 0.0)

    def test_char_similarity(self):
        self.assertEqual(compute_char_similarity("test", "test"), 1.0)
        self.assertGreater(compute_char_similarity("testing", "test"), 0.5)

    def test_intent_evaluation(self):
        y_true = ["set_reminder", "make_call", "open_app"]
        y_pred = ["CREATE_REMINDER", "MAKE_CALL", "OPEN_APP"]
        res = LDMEvaluator.evaluate_intents(y_true, y_pred)
        self.assertEqual(res["intent_accuracy"], 1.0)

    def test_code_mix_evaluation(self):
        y_true = [True, False, True]
        y_pred = [True, False, False]
        res = LDMEvaluator.evaluate_code_mix(y_true, y_pred)
        self.assertAlmostEqual(res["accuracy"], 0.6667, places=3)


class TestSemanticMetrics(unittest.TestCase):
    def setUp(self):
        self.evaluator = SemanticEvaluator.default()

    def test_preservation_batch(self):
        inputs = ["Remind me tomorrow", "Call mother"]
        norm = ["Remind me tomorrow.", "Call mother."]
        res = self.evaluator.evaluate_preservation_batch(inputs, norm)
        self.assertEqual(res["preservation_rate"], 1.0)

    def test_task_success(self):
        pred_intents = ["CREATE_REMINDER"]
        ref_intents = ["set_reminder"]
        entities = [{"task": "homework", "date": "tomorrow"}]
        llm_responses = ["I have set a reminder for homework on tomorrow."]
        raw_inputs = ["remind me for homework tomorrow"]

        res = SemanticEvaluator.evaluate_task_success(
            pred_intents, ref_intents, entities, llm_responses, raw_inputs
        )
        self.assertEqual(res["task_success_rate"], 1.0)


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = _ROOT / "results" / "test_reports"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.generator = ReportGenerator(output_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_json_and_csv(self):
        dummy_data = {"test": 123}
        p_json = self.generator.save_json(dummy_data, "test.json")
        self.assertTrue(p_json.exists())

        p_csv = self.generator.save_csv([{"colA": 1, "colB": 2}], "test.csv")
        self.assertTrue(p_csv.exists())

    def test_generate_markdown_report(self):
        dummy_metrics = {
            "asr": {"wer": 0.1, "cer": 0.05, "samples_evaluated": 1},
            "dialect": {"accuracy": 0.9, "macro_f1": 0.85, "per_class": {}},
            "systems_comparison": {
                "Baseline_A": {"normalization_token_f1": 0.3, "task_success_rate": 0.2},
                "System_B": {"normalization_token_f1": 0.7, "task_success_rate": 0.6},
                "System_C": {"normalization_token_f1": 0.9, "task_success_rate": 0.85},
            },
            "latency": {
                "total_latency_ms": {"mean": 15.0, "p50": 14.0, "p95": 16.0},
                "memory_usage_mb": {"mean": 20.0},
            },
        }
        md_path = self.generator.generate_markdown_report(dummy_metrics, "test_report.md")
        self.assertTrue(md_path.exists())
        content = md_path.read_text(encoding="utf-8")
        self.assertIn("Comparative Evaluation Matrix", content)
        self.assertIn("System C (Dialect-Aware AVA)", content)


if __name__ == "__main__":
    unittest.main()
