"""
src/llm/model_loader.py
=======================
Model loading and instantiation factory for the AVA Local LLM module.

Supports:
- Configurable models via YAML
- Model path or Hugging Face ID
- Configurable CPU/CUDA hardware target
- Replaceable backend (Mock / Transformers)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

from .base import BaseLLM
from .local_engine import MockLLMEngine, TransformersLLMEngine
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/llm.yaml")


class ModelLoader:
    """
    Factory for instantiating the configured local LLM backend.
    """

    @staticmethod
    def load(
        config_path_or_dict: Optional[Union[str, Path, Dict[str, Any]]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> BaseLLM:
        """
        Load and return a configured BaseLLM engine.

        Parameters
        ----------
        config_path_or_dict : Path, str, or dict, optional
            Path to YAML configuration or pre-parsed dict.
        project_root : Path, optional
            Root project directory for resolving relative paths.
        """
        cfg = ModelLoader._resolve_config(config_path_or_dict, project_root)

        model_cfg = cfg.get("model", {})
        gen_cfg = cfg.get("generation", {})
        prompt_cfg = cfg.get("prompt", {})

        # 1. Resolve prompt builder
        prompt_builder = PromptBuilder(
            system_prompt=prompt_cfg.get("system_prompt"),
            format_type=prompt_cfg.get("format", "chat"),
            include_metadata=prompt_cfg.get("include_metadata", True),
        )

        backend = model_cfg.get("backend", "mock").lower()
        model_name_or_path = model_cfg.get("path") or model_cfg.get("name", "mock")
        context_len = model_cfg.get("max_context_length", 2048)

        # 2. Return Mock if requested
        if backend == "mock":
            logger.info("Instantiating MockLLMEngine (backend: mock)...")
            return MockLLMEngine(
                model_name=model_name_or_path if model_name_or_path != "mock" else "mock-ava-v1",
                context_length=context_len,
                prompt_builder=prompt_builder,
            )

        # 3. Transformers backend
        if backend == "transformers":
            return ModelLoader._load_transformers(
                model_name_or_path=model_name_or_path,
                model_cfg=model_cfg,
                gen_cfg=gen_cfg,
                prompt_builder=prompt_builder,
                context_len=context_len,
            )

        raise ValueError(
            f"Unsupported LLM backend: '{backend}'. Available backends: ['mock', 'transformers']"
        )

    @staticmethod
    def _load_transformers(
        model_name_or_path: str,
        model_cfg: Dict[str, Any],
        gen_cfg: Dict[str, Any],
        prompt_builder: PromptBuilder,
        context_len: int,
    ) -> TransformersLLMEngine:
        """Load Hugging Face model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_req = model_cfg.get("device", "auto").lower()
        if device_req == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_req

        dtype_str = model_cfg.get("torch_dtype", "float32").lower()
        if dtype_str == "float16" and "cuda" in device:
            torch_dtype = torch.float16
        elif dtype_str == "bfloat16" and "cuda" in device:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        logger.info(
            f"Loading Transformers model '{model_name_or_path}' on {device} (dtype={torch_dtype})..."
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
        )
        model.to(device)
        model.eval()

        # Extract generation kwargs
        default_gen_cfg = {
            "max_new_tokens": gen_cfg.get("max_new_tokens", 128),
            "temperature": gen_cfg.get("temperature", 0.7),
            "top_p": gen_cfg.get("top_p", 0.9),
            "top_k": gen_cfg.get("top_k", 50),
            "repetition_penalty": gen_cfg.get("repetition_penalty", 1.1),
            "do_sample": gen_cfg.get("do_sample", True),
        }

        return TransformersLLMEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            device=device,
            context_length=context_len,
            default_gen_cfg=default_gen_cfg,
            prompt_builder=prompt_builder,
        )

    @staticmethod
    def _resolve_config(
        config_path_or_dict: Optional[Union[str, Path, Dict[str, Any]]],
        project_root: Optional[Union[str, Path]],
    ) -> Dict[str, Any]:
        if isinstance(config_path_or_dict, dict):
            return config_path_or_dict

        root = Path(project_root) if project_root else Path.cwd()
        cfg_path = Path(config_path_or_dict) if config_path_or_dict else root / _DEFAULT_CONFIG_PATH

        if not cfg_path.exists():
            # Check relative to this file
            alt_path = Path(__file__).resolve().parent.parent.parent / "configs" / "llm.yaml"
            if alt_path.exists():
                cfg_path = alt_path

        if not cfg_path.exists():
            logger.warning(f"Config file not found at {cfg_path}, using default empty config.")
            return {}

        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def load_llm(
    config_path_or_dict: Optional[Union[str, Path, Dict[str, Any]]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> BaseLLM:
    """Convenience helper to load an LLM engine."""
    return ModelLoader.load(config_path_or_dict=config_path_or_dict, project_root=project_root)
