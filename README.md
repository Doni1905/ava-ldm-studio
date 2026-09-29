# AVA LDM Studio

Build the LDM module as a NATIVE ANDROID APP for our project:

AVA – Accentric Virtual Assistant.

Use Kotlin + Jetpack Compose and integrate it into the existing Android UI/template. Do NOT create a website, web app, or browser UI.

LDM PURPOSE:

The Linguistic Dialect Model sits between ASR and the local LLM. It understands Tamil/Tanglish, regional dialect, slang, informal expressions and code-mixing, then normalizes the user's intended meaning for the LLM.

Create one clean Android LDM screen with:

1. LDM Playground

- Text input

- Analyze button

- 4–5 sample utterances

Example:

"Dei nalaiku assignment submit panna remind pannu"

→ "Remind me to submit my assignment tomorrow."

2. Analysis

Show:

Language | Dialect | Code-Mix | Style

Normalized Text | Intent | Entities | Confidence

3. Pipeline

ASR/Input → Language Detection → Dialect → Code-Mix → Normalization → Intent → LLM Handoff

4. LLM Handoff

Display structured JSON:

normalized_text, language, dialect, code_mix, intent, entities, confidence

with a Copy button.

5. Synthetic Dataset

Use only ~30 sample local utterances covering Tamil, Tanglish, English, slang and code-mixing with 8–10 intents.

6. Mini Evaluation

Show prototype metrics:

Normalization Accuracy, Intent Accuracy, Dialect/Code-Mix Detection, Processing Time.

Clearly label them "Synthetic Prototype Evaluation".

IMPLEMENTATION:

Use a lightweight local rule-based/mock LDM. No cloud API, no large model, no real training yet.

Create a replaceable Kotlin interface:

LdmProcessor.analyzeUtterance(input, userProfile)

The LDM must only normalize linguistic input; it must NOT execute Android actions or behave as a chatbot.

Final flow:

Android Voice/ASR → LDM → Normalized Meaning → Local LLM → Agent → Android Actions.

Keep the UI minimal, professional, dark blue/black/white, and consistent with the existing AVA Android design.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/f5b1d038-ac6b-48a2-b1d1-be701f65c67d).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
