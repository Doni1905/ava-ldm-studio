"""
src/llm/base.py
===============
Base classes, interfaces, and data containers for the AVA Local LLM module.

Decoupling Principle:
- The LLM receives normalized semantic input (via LDMHandoff), not raw dialect speech.
- The interface is completely decoupled from the LDM pipeline.
- Backends (Transformers, llama.cpp, Mock) can be swapped without altering callers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class LDMHandoff:
    """
    Standardized semantic payload handed off from LDM to LLM.

    The LLM receives pre-processed semantic intent and entities
    derived from dialect/code-mixed speech.
    """
    normalized_text: str
    intent: str = "GENERAL_QUERY"
    entities: Dict[str, Any] = field(default_factory=dict)
    language: str = "unknown"
    dialect: str = "Standard"
    confidence: Dict[str, float] = field(default_factory=dict)
    raw_transcript: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LDMHandoff":
        """Build LDMHandoff from a dictionary (e.g. from pipeline or API)."""
        return cls(
            normalized_text=data.get("normalized_text") or data.get("normalized") or "",
            intent=data.get("intent", "GENERAL_QUERY"),
            entities=data.get("entities", {}) or {},
            language=data.get("language", "unknown"),
            dialect=data.get("dialect", "Standard"),
            confidence=data.get("confidence", {}) or {},
            raw_transcript=data.get("transcript") or data.get("raw_transcript"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str
    tokens_generated: int
    latency_ms: float
    memory_peak_mb: float
    model_name: str
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "latency_ms": round(self.latency_ms, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "model_name": self.model_name,
            "backend": self.backend,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BaseLLM(ABC):
    """Abstract Base Class for local LLM engines."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""
        pass

    @property
    @abstractmethod
    def backend(self) -> str:
        """Name of the backend implementation (e.g., 'transformers', 'mock')."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device running the model ('cpu', 'cuda', etc.)."""
        pass

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context window length in tokens."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a complete response for the given prompt or LDM handoff.
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream generated tokens incrementally.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        pass
