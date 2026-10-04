# AVA Dialect & Code-Mixed Speech Dataset

This document details the corpus composition, dialect distribution, annotation taxonomy, and benchmarking splits used for research and evaluation in AVA LDM Studio.

---

## 1. Corpus Overview

The AVA dataset is designed to address the severe underrepresentation of colloquial regional Tamil and Tamil-English code-mixing (Tanglish) in standard NLP and speech benchmarks. Most existing Tamil corpora (e.g., Common Voice Tamil) feature formal, standard script read speech, which fails when deployed on real-world colloquial conversational speech.

### Key Target Dialect Regions:
1. **Chennai (Madras Bashai)**: High frequency of English loanwords, distinctive slang (*machi, dei, galaata, bakkara, dhool*), rapid elision.
2. **Madurai (Southern Inland)**: Honorific and relational markers (*thambi, aama, ennanga*), elongated vowels, specific verb morphosyntax.
3. **Kongu (Western Tamil Nadu — Coimbatore/Erode/Salem)**: Distinctive honorific suffix (*-nga*), regional discourse particles (*ayya, yov, la*), phonological softening.
4. **Nellai (Tirunelveli/Thoothukudi — Southern Coastal)**: Unique markers (*pa, ppa, le*), verb construction *phone pottu kudu* (call someone), lexical archaisms.
5. **Standard Tamil**: Formal/semi-formal spoken Tamil as used in official media, navigation prompts, and educational contexts.

---

## 2. Dataset Splits and Partitioning

| Split | Description | Purpose | Target Proportion |
| :--- | :--- | :--- | :---: |
| **Train Set** | Acoustic dialect recordings + augmented Tanglish utterances | Training Wav2Vec2 dialect classifier and tuning lexical maps | 70% |
| **Validation Set** | Balanced speaker recordings across all 5 dialect regions | Hyperparameter tuning and checkpoint selection | 15% |
| **Test Set** | Held-out speaker recordings from all target districts | Acoustic dialect classification benchmarking | 15% |
| **Canonical Benchmark** | 30 rigorously human-verified multi-dialect test utterances | End-to-end linguistic normalization & task success evaluation | Fixed suite |

---

## 3. Canonical Benchmark Dataset (30 Utterances)

The 30-sample canonical benchmark suite serves as the empirical evaluation benchmark for the LDM, comparing Baseline A, System B, and System C.

### Representative Sample Set:
| ID | Raw Utterance | Dialect | Language | Code-Mixed | Ground Truth Normalized Text | Intent |
| :-: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | `Dei nalaiku assignment submit panna remind pannu` | Chennai | Tanglish | Yes | *Remind me to submit my assignment tomorrow.* | `CREATE_REMINDER` |
| **2** | `Machi inniku evening gym poga remind pannu` | Chennai | Tanglish | Yes | *Remind me to go to the gym this evening.* | `CREATE_REMINDER` |
| **3** | `Naalaikku kaalaila 6 maniku alarm vai` | Standard | Tamil (rom) | Yes | *Set an alarm for 6 in the morning tomorrow.* | `SET_ALARM` |
| **4** | `Amma ku call pannu` | Standard | Tanglish | Yes | *Call mother.* | `MAKE_CALL` |
| **5** | `Thambi ku oru message anuppu naan late ah varen nu` | Madurai | Tanglish | Yes | *Send a message to my brother that I will come late.* | `SEND_MESSAGE` |
| **6** | `Semma song ondru podu` | Chennai | Tamil (rom) | No | *Play a good song.* | `PLAY_MUSIC` |
| **7** | `Inniku weather eppadi iruku` | Standard | Tanglish | Yes | *How is the weather today?* | `CHECK_WEATHER` |
| **8** | `WhatsApp open pannu da` | Chennai | Tanglish | Yes | *Open WhatsApp.* | `OPEN_APP` |
| **9** | `Coimbatore ku vazhi kaatu` | Kongu | Tamil (rom) | No | *Show me directions to Coimbatore.* | `NAVIGATE` |
| **10** | `Volume konjam kammi pannu` | Standard | Tanglish | Yes | *Reduce the volume a little.* | `DEVICE_SETTING` |
| **14** | `Appa ku phone pottu kudu` | Nellai | Tamil (rom) | No | *Call father.* | `MAKE_CALL` |
| **15** | `Ilayaraja paatu podu la` | Kongu | Tamil (rom) | No | *Play Ilayaraja song.* | `PLAY_MUSIC` |
| **18** | `Office ku route sollu bruh` | Chennai | Tanglish | Yes | *Show me the route to the office.* | `NAVIGATE` |
| **26** | `Ayya nalaiku medicine saapida remind pannunga` | Kongu | Tanglish | Yes | *Remind me to take medicine tomorrow.* | `CREATE_REMINDER` |
| **29** | `Friend ku call pottu kudu pa` | Nellai | Tanglish | Yes | *Call my friend.* | `MAKE_CALL` |

*Complete dataset items and metadata are stored in `data/synthetic_benchmark.json` and `ai.ava.ldm.data.SyntheticDataset`.*

---

## 4. Acoustic Specifications

For all acoustic data:
- **Audio Encoding**: Uncompressed Linear PCM WAV
- **Sampling Rate**: 16,000 Hz (16 kHz)
- **Channels**: 1 (Mono)
- **Bit Depth**: 16-bit signed integer
- **Normalization**: Peak RMS normalized to -20 dBFS
- **Silence Trimming**: Leading and trailing silence trimmed below -40 dBFS threshold using `src/audio/preprocessing.py`.

---

## 5. Annotation Schema & Guidelines

Each utterance is annotated using strict linguistic criteria:
1. **`transcript`**: Verbatim orthographic or romanized representation of the acoustic input.
2. **`language`**: Categorized as `Tamil` (native Tamil script), `Tanglish` (romanized Tamil mixed with English tokens), `Tamil (romanised)` (romanized Tamil without English vocabulary), or `English`.
3. **`code_mixed`**: Boolean flag set to `true` if utterance contains tokens from multiple languages, defined via token distribution and lexical switching points.
4. **`dialect`**: Regional classification among `Chennai`, `Madurai`, `Kongu`, `Nellai`, or `Standard`.
5. **`normalized_text`**: Canonical English semantic translation stripped of slang discourse fillers while retaining exact action verbs and parameters.
6. **`intent`**: Primary functional category (e.g. `CREATE_REMINDER`, `SET_ALARM`).
7. **`entities`**: Key-value slot pairs (e.g., `time`, `target`, `action`, `location`).
