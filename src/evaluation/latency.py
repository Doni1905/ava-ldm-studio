"""
src/evaluation/latency.py
=========================
Latency and memory profiling for AVA LDM pipeline components:
- ASR Latency
- LDM Latency (Language ID, Dialect, Normalization, Intent/Entity)
- LLM Latency
- Total Pipeline Latency
- Peak Memory Allocation (RAM / VRAM)
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class PipelineLatencyRecord:
    """Latency records for a single pipeline run in milliseconds."""
    asr_ms: float = 0.0
    lang_id_ms: float = 0.0
    dialect_ms: float = 0.0
    normalization_ms: float = 0.0
    intent_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def ldm_total_ms(self) -> float:
        return self.lang_id_ms + self.dialect_ms + self.normalization_ms + self.intent_ms


class LatencyProfiler:
    """Collects and computes latency percentiles and memory statistics."""

    def __init__(self) -> None:
        self.records: List[PipelineLatencyRecord] = []

    def add_record(self, record: PipelineLatencyRecord) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Compute mean, median (p50), p95, and max for each component."""
        if not self.records:
            return {}

        def stats_for(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            p50_idx = int(0.50 * (n - 1))
            p95_idx = int(0.95 * (n - 1))
            return {
                "mean": round(statistics.mean(values), 2),
                "p50": round(sorted_v[p50_idx], 2),
                "p95": round(sorted_v[p95_idx], 2),
                "max": round(max(values), 2),
            }

        return {
            "asr_latency_ms": stats_for([r.asr_ms for r in self.records]),
            "ldm_latency_ms": stats_for([r.ldm_total_ms for r in self.records]),
            "llm_latency_ms": stats_for([r.llm_ms for r in self.records]),
            "total_latency_ms": stats_for([r.total_ms for r in self.records]),
            "memory_usage_mb": stats_for([r.peak_memory_mb for r in self.records]),
        }
