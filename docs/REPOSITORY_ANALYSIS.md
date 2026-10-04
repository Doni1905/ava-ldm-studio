# AVA LDM Studio - Repository Analysis

## Overview
This repository currently contains a **web-based prototype** built with React, Vite, and TypeScript. While the `README.md` explicitly requests a Native Android App (Kotlin + Jetpack Compose) and warns against creating a web UI, the existing codebase is a Lovable-generated web application that mocks the intended behavior of the Linguistic Dialect Model (LDM).

## Existing Architecture
- **Framework:** React SPA (Single Page Application) using Vite and TanStack Router.
- **Styling:** Tailwind CSS with Radix UI components.
- **Environment:** Node.js/Bun based environment.
- **State:** No backend; all logic executes on the client side in the browser. 
- **Missing Architecture:** There is no Android/Kotlin architecture, no Python environment, and no ML framework integration.

## Existing Modules
The core logic resides in `src/lib/ldm/` and `src/hooks/`:
- `processor.ts`: A mock, rule-based LDM processor written in TypeScript. It uses regular expressions and keyword matching to identify dialects (e.g., Chennai, Madurai), code-mixing, and intents, and to normalize Tanglish to English.
- `dataset.ts`: A hardcoded synthetic dataset containing 30 sample utterances.
- `evaluation.ts`: A simple evaluation script that computes token-level F1 scores for normalization and checks intent/dialect accuracy against the synthetic dataset.
- `use-speech.ts`: A React hook implementing Automatic Speech Recognition (ASR) via the browser's native `SpeechRecognition` (Web Speech API).
- **UI Routes (`src/routes/`):** Contains the playground (`index.tsx`), dataset viewer (`dataset.tsx`), evaluation dashboard (`evaluation.tsx`), and pipeline visualization (`pipeline.tsx`).

## Data Flow
1. **Input:** User provides text or voice input via the browser UI.
2. **ASR:** If voice is used, `use-speech.ts` captures audio and transcripts it using the browser's Web Speech API.
3. **Processing:** The transcript is passed to `ruleBasedLdm.analyzeUtterance()` (in `processor.ts`).
4. **Output:** The processor returns a JSON object (`LdmAnalysis`) containing `normalizedText`, `language`, `dialect`, `codeMix`, `intent`, and `entities`.
5. **UI Rendering:** The React UI displays the parsed output and formatting suitable for an "LLM Handoff".

## Model Flow
- **Actual Models:** **None.** There are no actual Machine Learning models, weight files, or model pipelines in the repository.
- **Mock Model Flow:** The flow mimics an ML pipeline using a sequence of synchronous string operations: Tokenization -> Regex Phrase Replacement -> Intent Matching -> Entity Extraction -> Template-based Normalization.

## Dependencies
- **Current (Frontend):** `@tanstack/react-router`, `react`, `tailwindcss`, `lucide-react`, `radix-ui`, `vite`. 
- **Missing (Android):** Kotlin, Jetpack Compose, Android SDK.
- **Missing (ML/Python):** `requirements.txt`, PyTorch/TensorFlow, Transformers, Python scripts.

## Current Status
- **Implemented:** A complete, interactive web-based UI prototype and a robust rule-based TS mock of the LDM logic.
- **Partially Implemented:** Synthetic dataset and basic evaluation logic (but in TS, not Python).
- **Not Implemented (per README):** Android native application, Kotlin `LdmProcessor` interface, Python training environment, actual ML-based LDM, and testing suites.

## Missing Components
1. **Android Project Structure:** `build.gradle`, `AndroidManifest.xml`, Kotlin source directories.
2. **Native ASR:** Android `SpeechRecognizer` implementation.
3. **Python Environment:** `requirements.txt`, `setup.py`, or `pyproject.toml`.
4. **ML Infrastructure:** Model training scripts, data loaders, Jupyter notebooks, and PyTorch/TF models.
5. **Testing:** Unit tests (`*.test.ts` or Kotlin tests).

## Recommended Implementation Order
To align the repository with the goals outlined in the `README.md`, the following implementation sequence is recommended:

1. **Initialize Android Project:** Create a new Android Studio project in the repository root (or a dedicated `android/` subfolder) using Kotlin and Jetpack Compose.
2. **Port UI to Jetpack Compose:** Recreate the minimal, professional dark theme UI (Playground, Analysis, LLM Handoff) using Android native components.
3. **Port LDM Logic to Kotlin:** Translate the rule-based logic from `src/lib/ldm/processor.ts` into a Kotlin `LdmProcessor.analyzeUtterance(input, userProfile)` interface.
4. **Integrate Native ASR:** Replace the browser Web Speech API with Android's native Speech-to-Text capabilities.
5. **Establish ML Workspace:** Create a `ml/` directory with Python configuration (`requirements.txt`), training scripts, and Jupyter notebooks to begin transitioning the LDM from a rule-based system to an actual trained model.
6. **Deprecate Web Prototype:** Once the Android application is functional, phase out the React/Vite codebase.
