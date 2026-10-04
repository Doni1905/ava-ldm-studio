"""
src/llm/__init__.py
===================
AVA Local LLM Interface Package.

Provides a modular, decoupled interface between the LDM semantic handoff
and local language models.
"""

from .base import BaseLLM, LDMHandoff, LLMResponse
from .local_engine import MemoryTracker, MockLLMEngine, TransformersLLMEngine
from .model_loader import ModelLoader, load_llm
from .prompt_builder import PromptBuilder

__all__ = [
    "BaseLLM",
    "LDMHandoff",
    "LLMResponse",
    "MockLLMEngine",
    "TransformersLLMEngine",
    "MemoryTracker",
    "ModelLoader",
    "load_llm",
    "PromptBuilder",
]
