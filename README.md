## Fork fixes and verified setup (October 4, 2026)

This fork is https://github.com/Doni1905/ava-ldm-studio. The Python API is **aiohttp**, not FastAPI; `/docs` is not implemented. Use Python **3.11** and Node **22.12+**. The virtual environment is not included in Git.

```bash
git clone https://github.com/Doni1905/ava-ldm-studio.git
cd ava-ldm-studio
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux CPU only. On macOS use: pip install torch==2.14.1 torchaudio==2.11.0
pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q
npm ci
npm run typecheck
npm run lint
npm run build
```

Run in two terminals:

```bash
# Terminal 1, with .venv active, from repository root
python scripts/serve_api.py --host 127.0.0.1 --port 8000
# Terminal 2, from repository root
npm run dev -- --host 127.0.0.1 --port 5173
```

Open http://127.0.0.1:5173. Text analysis also works in the browser when the Python API is stopped. The API binds to loopback by default; do not expose it publicly without authentication and tighter upload/path controls. This is a local research demo, not a production service.

### Verified

- Fresh `npm ci`, TypeScript check and production build pass; npm audit reports zero vulnerabilities in the tested lockfile. ESLint has zero errors and six nonblocking Fast Refresh warnings.
- 450 Python tests pass, including malformed JSON and input-type regressions. Run `pytest tests`, since the cache directory contains standalone experimental scripts that are not the test suite.
- Playground, dataset, evaluation, pipeline and login web routes render. Wait for client hydration before inspecting them.
- Text CLI and actual HTTP text analysis work. Text mode no longer downloads audio models unnecessarily.
- Whisper loads and silence produces empty text, not a hallucinated command. Acoustic dialect detection requires a trained checkpoint; none is included. Without it, audio processing uses transcript-based lexical dialect markers with zero acoustic confidence instead of an untrained random classifier.
- Synthetic evaluation is reproducible. Its 53.33% end-to-end task success and 76.67% intent accuracy are limitations, not claims of production accuracy. Reported audio/LLM timings in this text benchmark are placeholders, not live speech measurements.

### Not verified or not implemented

- Android APK build and physical-device microphone behavior are not verified here. Android needs JDK 17, SDK/platform 35 and Android Studio or the SDK command-line tools. Use `cd android && bash gradlew testDebugUnitTest assembleDebug` if `./gradlew` lacks executable permission.
- No trained dialect checkpoint, production dataset training, real Tamil/Tanglish speech accuracy validation or downstream LLM action execution is included in these fixes.
- Browser Web Speech recognition may use a vendor's cloud service. It must not be described as guaranteed offline. Local Whisper weights are downloaded on first audio use, then can run from cache.
- Demo login is client-side only, not a security boundary. LICENSE is empty upstream despite the README claiming MIT; clarify licensing with the upstream owner before redistribution.

The historical research write-up below describes project goals and earlier examples. The verified setup and limitations above take precedence where they differ.

---

# AVA LDM Studio: Linguistic Dialect Model for Tamil & Tanglish

> **Final Year Project (FYP) Research Documentation & Implementation**  
> An on-device linguistic middleware system bridging dialectal Tamil/Tanglish speech recognition and local Large Language Models.

---

## 1. Project Title
**AVA LDM Studio (Accentric Virtual Assistant — Linguistic Dialect Model)**

---

## 2. Research Objective
Standard commercial and open-source Large Language Models (LLMs) suffer from severe comprehension degradation when processing non-standard colloquial dialects, discourse fillers, and code-mixed speech (e.g., Tamil-English Tanglish). 

The primary research objective of **AVA LDM** is to design, implement, and evaluate a lightweight, zero-cloud linguistic middleware layer positioned between Automatic Speech Recognition (ASR) and a Local LLM. The model detects regional dialects (Chennai, Madurai, Kongu, Nellai, Standard), identifies code-mixing, strips non-semantic discourse particles, normalizes colloquial verbal inflections, and produces deterministic structured semantic representations (canonical English, intent, and slots) for the LLM without requiring cloud dependencies or continuous model retraining.

---

## 3. Architecture

AVA enforces a strict linear separation of concerns:

```
[ Spoken Audio (16kHz WAV) ]
              │
              ▼
    ┌──────────────────┐
    │    ASR Engine    │   Whisper / Wav2Vec2-XLSR
    └─────────┬────────┘
              │  Raw Orthographic Transcript
              ▼
    ┌──────────────────┐
    │     AVA LDM      │   Language ID → Dialect ID → Code-Mix → Normalization → Intent/Slots
    └─────────┬────────┘
              │  Normalized Meaning + Structured JSON Handoff
              ▼
    ┌──────────────────┐
    │    Local LLM     │   GGUF / Llama.cpp / Transformers (Offline)
    └─────────┬────────┘
              │  Executable Action Plan
              ▼
[ Android Application / OS Layer ]
```

### Architectural Boundaries:
- **The LDM is NOT an execution engine**: The LDM does not toggle device settings, send SMS, or launch applications. It outputs validated linguistic understanding.
- **The LDM is NOT a conversational chatbot**: The LDM does not hallucinate free-form dialogue; conversational interaction is delegated exclusively to the downstream Local LLM.
- **Dual Implementation**:
  - **Python Research Core (`src/`)**: High-throughput benchmarking, acoustic modeling (Wav2Vec2), evaluation metrics, and aiohttp service.
  - **Native Android Prototype (`android/`)**: Zero-dependency Kotlin 2.0 + Jetpack Compose engine operating at sub-millisecond on-device latencies.

Detailed architectural specifications and component data flows are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Dataset

The project evaluates regional spoken Tamil and Tanglish across 5 major dialect varieties:
1. **Chennai (Madras Bashai)**: High frequency of English loan elision and colloquial markers (*dei, machi, bruh, vaada*).
2. **Madurai (Southern Inland)**: Honorific relational particles and elongated vowels (*thambi, aama, ennanga*).
3. **Kongu (Western / Coimbatore)**: Softened honorific markers (*-nga*) and regional particles (*ayya, yov, la*).
4. **Nellai (Southern Coastal / Tirunelveli)**: Regional phrasal verbs (*phone pottu kudu*) and distinctive markers (*pa, ppa*).
5. **Standard Tamil**: Neutral conversational and broadcast Tamil.

### Canonical Benchmark Dataset:
The benchmark suite consists of **30 canonical multi-dialect test utterances** with expert ground-truth annotations for language, dialect, code-mixing status, canonical normalized text, functional intent, and semantic entity slots.

Full dataset composition, splits, and acoustic guidelines are detailed in [docs/DATASET.md](docs/DATASET.md).

---

## 5. Installation

### Prerequisites:
- Python 3.10 or 3.11
- Node.js 22.12+ and npm (for web research studio)
- Java JDK 17 or 21 (for Android Jetpack Compose module)
- Git

### Clone the Repository:
```bash
git clone https://github.com/Doni1905/ava-ldm-studio.git
cd ava-ldm-studio
```

---

## 6. Environment Setup

### 6.1 Python Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2 Frontend Setup (Research Web Studio)
```bash
npm install
```

---

## 7. Dataset Preparation

Acoustic files must adhere to the standardized input format:
- Format: Uncompressed 16-bit PCM WAV
- Sampling Rate: 16,000 Hz (16 kHz)
- Channels: 1 (Mono)
- Normalization: Peak RMS normalized to -20 dBFS

Preprocessing can be executed directly using the audio preprocessing utility:
```bash
python -m src.audio.preprocessing --input raw_audio/ --output data/processed/
```

The canonical benchmark metadata is available at `data/synthetic_benchmark.json`.

---

## 8. Model Setup

### 8.1 ASR Models
By default, the Python pipeline utilizes OpenAI Whisper (`base` or `small`):
```bash
# Pre-download Whisper weights (optional, automated upon first execution)
python -c "import whisper; whisper.load_model('base')"
```

### 8.2 Dialect Classification Model
Acoustic classification models (Wav2Vec2-XLSR-300M) are defined in `src/dialect/model.py`. Configuration is managed via `configs/dialect.yaml`.

### 8.3 Local LLM Setup
The local LLM backend is configured via `configs/llm.yaml`. Supported engines include:
- `mock`: Instant, deterministic responses for testing.
- `llama_cpp`: Quantized GGUF models on CPU/CUDA.
- `huggingface`: PyTorch Transformers pipeline.

Refer to [docs/LOCAL_LLM.md](docs/LOCAL_LLM.md) for GGUF model setup.

---

## 9. Training

To train or fine-tune the acoustic dialect classifier on custom regional Tamil audio:
```bash
python -m src.dialect.train --config configs/dialect.yaml
```

The script performs stratified cross-validation across the 5 dialect classes and saves checkpoints to `models/dialect_classifier/`.

---

## 10. Evaluation

To reproduce the complete empirical evaluation across **Baseline A**, **System B**, and **System C**:

```bash
python scripts/run_evaluation.py
```

### Generated Artifacts:
- `results/metrics.json`: Full machine-readable metrics.
- `results/metrics.csv`: Tabular metric summary.
- `results/confusion_matrix.png`: Regional dialect classification confusion matrix.
- `results/evaluation_report.md`: Complete formatted markdown benchmark report.

Detailed evaluation methodology is documented in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## 11. Running Inference

### Unified CLI:
Run the complete pipeline on an audio recording or text transcript:

```bash
# Analyze audio file:
python scripts/run_ldm.py --audio data/samples/chennai_sample1.wav

# Analyze raw transcript directly:
python scripts/run_ldm.py --text "Dei nalaiku assignment submit panna remind pannu"
```

---

## 12. API Usage

Launch the local REST API server:
```bash
python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

### Core Endpoint: `POST /analyze`
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Machi inniku evening gym poga remind pannu"}'
```

### Response:
```json
{
  "transcript": "Machi inniku evening gym poga remind pannu",
  "language": "Tanglish",
  "code_mixed": true,
  "dialect": "Chennai",
  "normalized_text": "Remind me to go to the gym this evening.",
  "intent": "CREATE_REMINDER",
  "entities": {
    "action": "go to gym",
    "time": "this evening"
  },
  "confidence": {
    "language": 0.95,
    "dialect": 0.85,
    "intent": 0.90,
    "overall": 0.90
  },
  "latency_ms": {
    "asr": 0.0,
    "ldm": 0.24,
    "total": 0.24
  }
}
```

Additional endpoints:
- `POST /transcribe`: ASR transcription only.
- `POST /detect-dialect`: Regional dialect classification.
- `POST /normalize`: Morphological slang normalization.
- `POST /evaluate`: Trigger benchmark evaluation suite.

No interactive OpenAPI `/docs` endpoint is implemented.

---

## 13. Example Input / Output Walkthrough

| Pipeline Stage | Intermediate Representation / Data |
| :--- | :--- |
| **1. User Acoustic Input** | Spoken utterance: *"Dei nalaiku assignment submit panna remind pannu"* |
| **2. ASR Transcript** | `"Dei nalaiku assignment submit panna remind pannu"` |
| **3. Language Identification** | `Tanglish` (Code-Mixed: `true`, Script: Latin Romanized) |
| **4. Dialect Classification** | `Chennai` (Trigger marker: *"dei"*) |
| **5. Discourse Stripping** | Discourse particle *"dei"* filtered out |
| **6. Verb Normalization** | *"panna remind pannu"* → `"Remind me to..."` |
| **7. Temporal Normalization** | *"nalaiku"* → `"tomorrow"` |
| **8. Intent Extraction** | `CREATE_REMINDER` |
| **9. Entity Extraction** | `action: "submit assignment"`, `time: "tomorrow"` |
| **10. Local LLM Handoff** | Structured JSON payload ready for local LLM inference |

---

## 14. Empirical Results

The empirical results reported below are measured values from the evaluation benchmark (`results/metrics.json`):

### Comparative Benchmark:
| Metric | Baseline A (`ASR → LLM`) | System B (`Generic LDM`) | System C (`Dialect-Aware LDM`) | Delta (C vs A) |
| :--- | :---: | :---: | :---: | :---: |
| **Normalization Token F1** | 37.36% | **85.21%** | 85.04% | **+47.68%** |
| **Semantic Preservation Rate** | 65.79% | **97.37%** | **97.37%** | **+31.58%** |
| **Intent Accuracy** | 73.33% | **76.67%** | **76.67%** | **+3.34%** |
| **Entity Extraction F1** | 72.13% | **75.41%** | **75.41%** | **+3.28%** |
| **End-to-End Task Success** | 16.67% | **53.33%** | **53.33%** | **+36.66%** |

### Dialect Classification:
- **Overall Accuracy**: **90.00%**
- **Macro F1**: **91.35%** (Chennai: 76.92%, Madurai: 100.0%, Kongu: 85.71%, Nellai: 100.0%, Standard: 94.12%)

### Latency Profile:
- **ASR Latency**: 12.50 ms
- **LDM Latency**: **0.25 ms**
- **LLM Handoff Latency**: 0.02 ms
- **Total Pipeline Latency**: **12.77 ms**

Empirical experiments and failure case studies are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## 15. Limitations

A complete scientific disclosure is provided in [docs/LIMITATIONS.md](docs/LIMITATIONS.md):
1. **Out-of-Vocabulary Slang**: Uncurated rural idioms or newly coined Tanglish slang not present in `slang_map.csv` pass through without normalization.
2. **Short Utterance Dialect Ambiguity**: Brief commands lacking distinct regional particles default to `Standard` Tamil.
3. **Non-Standardized Tanglish Orthography**: Irregular phonetic spelling variations can occasionally bypass regex normalizers.
4. **Execution Boundary**: The LDM strictly performs linguistic normalization and does not interact with Android operating system services.

---

## 16. Citation & Attribution

If you use this codebase or benchmark methodology in your research, please cite:

```bibtex
@misc{ava_ldm_studio_2026,
  title={AVA LDM Studio: A Linguistic Dialect Model for Regional Tamil Dialects and Code-Mixed Speech},
  author={AVA Research Team},
  year={2026},
  howpublished={\url{https://github.com/Doni1905/ava-ldm-studio}},
  note={Final Year Project Research Repository}
}
```

---

## 17. License Information

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
