# Local Large Language Model (LLM) Interface

This document specifies the Local LLM Interface in AVA, detailing the modular adapter architecture, prompt construction, memory profiling, and model interchangeability.

---

## 1. Design Principles & Separation of Concerns

The LLM module sits at the downstream end of the LDM pipeline:

```
[ LDM Structured Output ]
            │
            ▼
┌────────────────────────┐
│  Prompt Builder Module │  (Formats normalized text, intent, and slots)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   BaseLLMEngine Class  │  (Pluggable Abstract Interface)
└───────────┬────────────┘
            │
    ┌───────┴────────┬────────────────┐
    ▼                ▼                ▼
[ LlamaCpp ]  [ Transformers ]   [ MockEngine ]
  (GGUF)        (PyTorch)       (Testing/CI)
```

### Critical Architectural Decisions:
1. **Normalized Input Only**: The Local LLM **never** receives raw dialect speech or colloquial slang directly. Instead, it receives canonical, disambiguated English semantics produced by the LDM.
2. **Backend Agnostic**: The system does not lock into a single model. Any quantized model running via `llama-cpp-python` or `transformers` can be swapped via configuration.
3. **Hardware Flexibility**: Operates entirely offline on local CPU (using AVX2 / ARM NEON) or CUDA GPUs.

---

## 2. Configuration & Parameter Specification (`configs/llm.yaml`)

The LLM is fully configured via external YAML without requiring code changes:

```yaml
llm:
  backend: "mock"                 # Options: 'llama_cpp', 'huggingface', 'mock'
  model_path: "models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  device: "cpu"                   # Options: 'cpu', 'cuda'
  n_threads: 4                    # Number of CPU threads
  n_gpu_layers: 0                 # Layers to offload to GPU (0 for CPU)
  context_length: 2048            # Maximum context window
  max_new_tokens: 128             # Maximum tokens to generate
  temperature: 0.1                # Low temperature for deterministic output
  top_p: 0.95
  streaming: false                # Token streaming support
```

---

## 3. Engine Adapter Implementations (`src/llm/`)

- **`BaseLLMEngine` (`src/llm/base.py`)**: Abstract base class enforcing contracts:
  - `generate(prompt: str) -> LLMResponse`
  - `generate_stream(prompt: str) -> Iterator[str]`
  - `get_metrics() -> Dict[str, float]` (measures generation latency, tokens per second, memory consumption).
- **`LlamaCppEngine` (`src/llm/local_engine.py`)**: High-performance backend utilizing quantized GGUF weights (Q4_K_M, Q5_K_M) via `llama-cpp-python`. Enables 10–25 tokens/second inference on commodity quad-core CPUs.
- **`HuggingFaceEngine` (`src/llm/local_engine.py`)**: PyTorch-based pipeline using Hugging Face AutoModelForCausalLM with 4-bit/8-bit bitsandbytes quantization.
- **`MockLLMEngine` (`src/llm/local_engine.py`)**: Zero-dependency mock engine producing validated responses for continuous testing and automated benchmarks.

---

## 4. Prompt Engineering for Structured Handoff

The `PromptBuilder` (`src/llm/prompt_builder.py`) constructs focused, minimal prompts designed to elicit deterministic assistant actions:

```
[SYSTEM PROMPT]
You are AVA, an accurate, concise voice assistant. 
The user's query has been normalized from regional speech into standard English.
Respond helpfully, concisely, and execute the user's intent.

[SEMANTIC INPUT]
Normalized Request: "Remind me to submit my assignment tomorrow."
Detected Intent: CREATE_REMINDER
Entities: {"action": "submit assignment", "time": "tomorrow"}

[ASSISTANT ACTION PLAN]
```

By providing both the normalized natural language and the extracted semantic slots, the local LLM can produce action plans with zero reasoning overhead or linguistic confusion.
