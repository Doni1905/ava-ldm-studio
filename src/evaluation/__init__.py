"""
src/evaluation/__init__.py
==========================
AVA LDM Evaluation and Benchmarking Suite.
"""

from .asr_metrics import compute_cer, compute_wer, evaluate_asr_batch
from .dialect_metrics import DialectEvaluator
from .latency import LatencyProfiler, PipelineLatencyRecord
from .ldm_metrics import LDMEvaluator
from .report import ReportGenerator
from .semantic_metrics import SemanticEvaluator

__all__ = [
    "compute_wer",
    "compute_cer",
    "evaluate_asr_batch",
    "DialectEvaluator",
    "LDMEvaluator",
    "SemanticEvaluator",
    "LatencyProfiler",
    "PipelineLatencyRecord",
    "ReportGenerator",
]
