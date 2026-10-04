"""
src/llm/local_engine.py
=======================
Local LLM engine implementations for AVA LDM Studio.

Contains:
1. MemoryTracker: Measures CPU RAM (via tracemalloc) and CUDA VRAM.
2. MockLLMEngine: High-speed, zero-download engine for deterministic tests and lightweight CPU environments.
3. TransformersLLMEngine: Full Hugging Face AutoModelForCausalLM engine with streaming and hardware acceleration.
"""

from __future__ import annotations

import logging
import threading
import time
import tracemalloc
from typing import Any, Dict, Iterator, List, Optional, Union

import torch

from .base import BaseLLM, LDMHandoff, LLMResponse
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Helper to monitor peak memory during model inference."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._cuda_active = "cuda" in device and torch.cuda.is_available()

    def __enter__(self) -> "MemoryTracker":
        tracemalloc.start()
        tracemalloc.reset_peak()
        if self._cuda_active:
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def get_peak_mb(self) -> float:
        """Returns peak allocated memory in Megabytes (MB)."""
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_mb = peak_bytes / (1024 * 1024)

        if self._cuda_active:
            cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return max(cpu_mb, cuda_mb)

        return cpu_mb


class MockLLMEngine(BaseLLM):
    """
    Mock Local LLM engine.

    Provides deterministic, semantic responses tailored to AVA intents and entities.
    Requires zero model downloads, enabling ultra-fast testing and decoupled development.
    """

    def __init__(
        self,
        model_name: str = "mock-ava-v1",
        context_length: int = 2048,
        prompt_builder: Optional[PromptBuilder] = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._context_length = context_length
        self.prompt_builder = prompt_builder or PromptBuilder()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def context_length(self) -> int:
        return self._context_length

    def count_tokens(self, text: str) -> int:
        """Approximates token count by word and punctuation boundaries."""
        return max(1, len(text.split()))

    def generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        t0 = time.perf_counter()

        with MemoryTracker("cpu") as mem:
            response_text = self._synthesize_response(prompt)
            tokens_generated = self.count_tokens(response_text)

        latency_ms = (time.perf_counter() - t0) * 1000
        peak_mb = mem.get_peak_mb()

        return LLMResponse(
            text=response_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            memory_peak_mb=peak_mb,
            model_name=self.model_name,
            backend=self.backend,
            metadata={
                "intent": getattr(prompt, "intent", None) if isinstance(prompt, LDMHandoff) else None,
                "temperature": temperature or 0.0,
            },
        )

    def stream_generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        full_text = self._synthesize_response(prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk

    def _synthesize_response(self, prompt: Union[str, LDMHandoff]) -> str:
        if isinstance(prompt, LDMHandoff):
            intent = prompt.intent
            entities = prompt.entities
            norm_text = prompt.normalized_text

            if intent == "CREATE_REMINDER":
                task = entities.get("task", "your task")
                date = entities.get("date", "soon")
                time_str = entities.get("time", "")
                when = f"{date} at {time_str}".strip() if time_str else date
                return f"I have scheduled a reminder to {task} for {when}."

            if intent == "MAKE_CALL":
                person = entities.get("person", "the contact")
                return f"Placing a phone call to {person}."

            if intent == "SEND_MESSAGE":
                person = entities.get("person", "the recipient")
                msg = entities.get("message_content", norm_text)
                return f"Drafting message to {person}: '{msg}'."

            if intent == "OPEN_APP":
                app = entities.get("app", "the requested app")
                return f"Opening {app}."

            if intent == "DEVICE_SETTING":
                setting = entities.get("device_setting", "setting")
                return f"Adjusting {setting}."

            if norm_text:
                return f"Understood: '{norm_text}'. I am assisting you with this request."

            return "Understood. How can I assist you further?"

        # String prompt
        return f"Response to: {prompt.strip()}"


class TransformersLLMEngine(BaseLLM):
    """
    Hugging Face Transformers LLM Engine.

    Supports causal language models (e.g. Qwen, TinyLlama, SmolLM, Mistral).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: str = "cpu",
        context_length: int = 2048,
        default_gen_cfg: Optional[Dict[str, Any]] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._model_name = model_name
        self._device = device
        self._context_length = context_length
        self.default_gen_cfg = default_gen_cfg or {}
        self.prompt_builder = prompt_builder or PromptBuilder()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return "transformers"

    @property
    def device(self) -> str:
        return self._device

    @property
    def context_length(self) -> int:
        return self._context_length

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    def _prepare_prompt_text(self, prompt: Union[str, LDMHandoff]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                messages = self.prompt_builder.build_messages(prompt)
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.debug(f"Chat template application failed ({e}), falling back to text prompt")

        return self.prompt_builder.build_text(prompt)

    def generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt_text = self._prepare_prompt_text(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self._device)
        input_len = inputs.input_ids.shape[1]

        # Merge generation kwargs
        gen_kwargs = {**self.default_gen_cfg, **kwargs}
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        # Normalize sampling vs greedy
        temp = gen_kwargs.get("temperature", 0.7)
        if temp <= 0.0 or not gen_kwargs.get("do_sample", True):
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)

        t0 = time.perf_counter()

        with MemoryTracker(self._device) as mem:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        latency_ms = (time.perf_counter() - t0) * 1000
        peak_mb = mem.get_peak_mb()

        # Extract only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_generated = len(new_tokens)

        return LLMResponse(
            text=output_text,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            memory_peak_mb=peak_mb,
            model_name=self.model_name,
            backend=self.backend,
            metadata={"prompt_tokens": input_len, "total_tokens": input_len + tokens_generated},
        )

    def stream_generate(
        self,
        prompt: Union[str, LDMHandoff],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        prompt_text = self._prepare_prompt_text(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self._device)

        gen_kwargs = {**self.default_gen_cfg, **kwargs}
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        temp = gen_kwargs.get("temperature", 0.7)
        if temp <= 0.0 or not gen_kwargs.get("do_sample", True):
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        # Run model.generate in a separate thread so streamer can yield tokens as generated
        thread = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs, "pad_token_id": self.tokenizer.eos_token_id},
        )
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()
