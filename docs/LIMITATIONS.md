# Research Limitations & Ethical Considerations

This document transparently outlines the known limitations, boundary assumptions, and areas for future research in the AVA Linguistic Dialect Model (LDM).

---

## 1. Scientific & Technical Limitations

### 1.1 Finite Lexicon & Out-of-Vocabulary (OOV) Slang
- **Current State**: The LDM normalizer relies on curated lexicons and morphological mapping rules (`data/slang_map.csv`, `data/expressions.csv`).
- **Limitation**: Regional youth slang, internet vernacular, and hyper-local district idioms not present in the curated mappings will not be normalized. While unmapped standard words pass through safely, rare colloquialisms may fail normalization.
- **Future Direction**: Integrating contextual token embeddings or continuous semantic vector search for OOV slang candidates.

### 1.2 Dialect Ambiguity in Short Utterances
- **Current State**: Dialect classification achieves **90.0%** overall accuracy on canonical benchmark utterances.
- **Limitation**: When users issue concise, non-dialectal commands (e.g., *"Amma ku call pannu"* or *"Alarm vai"*), no overt regional markers (*dei, la, pa, ayya*) exist. In these instances, the model defaults to `Standard`, as observed in the 2 misclassified Chennai samples in the confusion matrix.
- **Future Direction**: Incorporating long-term user session history to reinforce dialect priors across multiple turns.

### 1.3 Non-Standardized Tanglish Orthography
- **Current State**: Tanglish has no official spelling convention. A single word can be transcribed variously (e.g., *naalaikku*, *nalaiku*, *nalaiki*, *naalaiku*).
- **Limitation**: While regular expressions handle primary orthographic variants, highly irregular phonetic spellings can bypass normalization rules.
- **Future Direction**: Training a dedicated character-level Seq2Seq phonetic normalizer for romanized South Asian languages.

### 1.4 Acoustic Noise Sensitivity in Dialect Classification
- **Current State**: Acoustic classification was evaluated on clean 16 kHz audio normalized to -20 dBFS.
- **Limitation**: In high-noise environments (e.g., roadside traffic, outdoor market noise below 10 dB SNR), acoustic dialect confidence degrades, requiring fallback to lexical parsing.

### 1.5 Local LLM Quantization & Memory Footprint
- **Current State**: The LDM itself is ultra-lightweight (< 1 MB RAM, 0.25 ms latency).
- **Limitation**: Downstream execution of a 7B parameter Local LLM (e.g., Mistral-7B Q4_K_M) requires ~4.5 GB of RAM. On low-tier mobile devices with under 4 GB RAM, full on-device LLM hosting remains constrained without smaller specialized 1B–2B parameter models.

---

## 2. Dataset Scope & Prototype Scale

- **Scale of Evaluation**: The canonical evaluation suite comprises 30 human-verified utterances across 5 regional varieties. While this provides statistically consistent comparative baselines (Baseline A vs B vs C) for a Final Year Project (FYP), commercial production requires expanding to thousands of hours across all 38 districts of Tamil Nadu.
- **Demographic Balance**: Current sample recordings reflect predominantly young adult speakers (aged 18–35); elderly and child speakers with different acoustic timbres remain underrepresented.

---

## 3. Strict Boundary Assumptions

1. **No Direct Execution**: The LDM is explicitly designed **not** to interact with Android system services, databases, or third-party APIs. It is strictly an intermediate language understanding model.
2. **No General Conversational Chat**: The LDM does not answer general knowledge trivia or engage in open-ended chitchat; conversational capabilities are delegated exclusively to the downstream Local LLM.

---

## 4. Ethical Considerations & Privacy

- **Privacy Preserved**: AVA operates entirely locally (zero cloud dependency), ensuring that regional voice data, personal contacts, and daily schedules are never uploaded to remote servers.
- **Fair Linguistic Representation**: By explicitly modeling non-standard regional varieties (Madurai, Kongu, Nellai, Chennai), AVA aims to counteract linguistic bias in standard voice assistants that favor formal broadcast language.
