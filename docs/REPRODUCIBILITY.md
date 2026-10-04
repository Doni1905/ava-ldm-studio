# Reproducibility Guide & Benchmark Replication

This guide provides step-by-step instructions to replicate all empirical evaluations, run the LDM pipelines, launch the local REST API, and build both the research web studio and native Android prototype.

---

## 1. System Requirements & Environment

### Hardware Specifications:
- **CPU**: Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5/7, or Apple Silicon (M1/M2/M3)
- **RAM**: Minimum 8 GB (16 GB recommended for full local LLM inference)
- **Disk Space**: ~2 GB for code and synthetic datasets (~8 GB if downloading 7B GGUF weights)
- **OS**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 13+

### Software Prerequisites:
- Python 3.10 or 3.11
- Node.js 18+ and npm
- Java JDK 17 or 21 (for Android module build)
- Git

---

## 2. Python Environment Setup

Clone the repository and initialize the Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/example/ava-ldm-studio.git
cd ava-ldm-studio

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Replicating the Benchmark Evaluation

To execute the complete empirical evaluation across Baseline A, System B, and System C, run:

```bash
python scripts/run_evaluation.py
```

### What this script does:
1. Loads the 30 canonical test benchmark utterances from `data/synthetic_benchmark.json`.
2. Evaluates ASR transcription metrics (WER, CER).
3. Evaluates dialect classification accuracy and computes the per-class confusion matrix.
4. Evaluates linguistic normalization (Token F1, BLEU-1, Semantic Preservation).
5. Compares task success rate across Baseline A, System B, and System C.
6. Measures component and end-to-end latencies.
7. Generates reproducible artifacts in `results/`:
   - `results/metrics.json` (machine-readable metrics)
   - `results/metrics.csv` (tabular metrics)
   - `results/confusion_matrix.png` (dialect classification heatmap)
   - `results/evaluation_report.md` (detailed markdown report)

---

## 4. Running the Unified LDM CLI

Run the full LDM pipeline on individual audio files or raw transcripts:

```bash
# Analyze an audio file:
python scripts/run_ldm.py --audio data/samples/chennai_sample1.wav

# Or analyze a text transcript directly:
python scripts/run_ldm.py --text "Dei nalaiku assignment submit panna remind pannu"
```

### Output:
```json
{
  "transcript": "Dei nalaiku assignment submit panna remind pannu",
  "language": "Tanglish",
  "code_mixed": true,
  "dialect": "Chennai",
  "normalized_text": "Remind me to submit my assignment tomorrow.",
  "intent": "CREATE_REMINDER",
  "entities": {
    "action": "submit assignment",
    "time": "tomorrow"
  },
  "confidence": {
    "language": 0.95,
    "dialect": 0.85,
    "intent": 0.90,
    "overall": 0.90
  },
  "latency_ms": {
    "asr": 12.5,
    "ldm": 0.25,
    "total": 12.75
  }
}
```

---

## 5. Serving the Local REST API

Start the high-performance FastAPI integration layer:

```bash
python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

Verify the endpoint with curl:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Machi inniku evening gym poga remind pannu"}'
```

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 6. Building the Web Research Studio

```bash
# Install frontend dependencies
npm install

# Start local research studio
npm run dev

# Or compile production bundle
npm run build
```

---

## 7. Building & Testing the Native Android Module

```bash
# On Windows:
.\android\gradlew.bat -p android testDebugUnitTest

# On Linux / macOS:
./android/gradlew -p android testDebugUnitTest
```

All 9 canonical unit tests in `LdmProcessorTest.kt` will execute, verifying dialect parsing, normalization, JSON formatting, and the evaluation engine.
