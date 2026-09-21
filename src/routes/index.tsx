import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { DATASET, INTENT_LABELS, SAMPLE_UTTERANCES } from "@/lib/ldm/dataset";
import { ruleBasedLdm } from "@/lib/ldm/processor";
import { runEvaluation } from "@/lib/ldm/evaluation";
import { toHandoff, type LdmAnalysis } from "@/lib/ldm/types";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AVA — Linguistic Dialect Model (LDM)" },
      {
        name: "description",
        content:
          "AVA's Linguistic Dialect Model: understands Tamil, Tanglish, dialect and slang, then normalizes meaning for the local LLM.",
      },
      { property: "og:title", content: "AVA — Linguistic Dialect Model (LDM)" },
      {
        property: "og:description",
        content:
          "Analyze Tamil/Tanglish utterances, detect dialect and code-mixing, and hand normalized meaning to the local LLM.",
      },
    ],
  }),
  component: LdmScreen,
});

const PIPELINE = [
  "ASR / Input",
  "Language Detection",
  "Dialect",
  "Code-Mix",
  "Normalization",
  "Intent",
  "LLM Handoff",
];

function Section({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-border bg-card p-4">
      <div className="mb-3">
        <h2 className="text-sm font-semibold tracking-wide text-foreground uppercase">{title}</h2>
        {caption && <p className="mt-1 text-xs text-muted-foreground">{caption}</p>}
      </div>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-secondary px-3 py-2">
      <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-0.5 text-sm leading-snug text-foreground">{value}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-secondary px-3 py-3 text-center">
      <p className="text-lg font-semibold text-foreground">{value}</p>
      <p className="mt-1 text-[10px] leading-tight tracking-wide text-muted-foreground uppercase">
        {label}
      </p>
    </div>
  );
}

function LdmScreen() {
  const [input, setInput] = useState(SAMPLE_UTTERANCES[0]);
  const [analysis, setAnalysis] = useState<LdmAnalysis | null>(null);
  const [copied, setCopied] = useState(false);
  const [showDataset, setShowDataset] = useState(false);

  const evaluation = useMemo(() => runEvaluation(), []);

  const analyze = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setAnalysis(ruleBasedLdm.analyzeUtterance(t, { region: "Tamil Nadu", preferredLanguage: "ta" }));
    setCopied(false);
  };

  const handoff = analysis ? JSON.stringify(toHandoff(analysis), null, 2) : "";
  const activeStage = analysis ? PIPELINE.length - 1 : 0;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(handoff);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="min-h-screen bg-app-shell">
      <div className="mx-auto min-h-screen w-full max-w-[430px] bg-background pb-10">
        {/* App bar */}
        <header className="sticky top-0 z-10 border-b border-border bg-background/95 px-4 py-3 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-sm font-bold text-primary-foreground">
              AVA
            </div>
            <div>
              <h1 className="text-sm font-semibold text-foreground">Linguistic Dialect Model</h1>
              <p className="text-[11px] text-muted-foreground">
                Accentric Virtual Assistant · on-device module
              </p>
            </div>
          </div>
        </header>

        <main className="space-y-4 px-4 pt-4">
          {/* 1. Playground */}
          <Section
            title="LDM Playground"
            caption="Speak or type Tamil, Tanglish or English. The LDM only normalizes meaning — it never performs actions."
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={3}
              placeholder="Dei nalaiku assignment submit panna remind pannu"
              className="w-full resize-none rounded-xl border border-input bg-secondary px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/40 focus:outline-none"
            />
            <button
              onClick={() => analyze(input)}
              className="mt-3 w-full rounded-xl bg-primary py-3 text-sm font-semibold text-primary-foreground transition-opacity active:opacity-80"
            >
              Analyze
            </button>

            <p className="mt-4 mb-2 text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
              Sample utterances
            </p>
            <div className="space-y-2">
              {SAMPLE_UTTERANCES.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setInput(s);
                    analyze(s);
                  }}
                  className="w-full rounded-xl border border-border bg-secondary px-3 py-2 text-left text-xs leading-snug text-secondary-foreground transition-colors active:bg-accent"
                >
                  {s}
                </button>
              ))}
            </div>
          </Section>

          {/* 2. Analysis */}
          <Section title="Analysis" caption="Linguistic surface features and resolved meaning.">
            {analysis ? (
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Language" value={analysis.language} />
                  <Field label="Dialect" value={analysis.dialect} />
                  <Field label="Code-Mix" value={analysis.codeMix} />
                  <Field label="Style" value={analysis.style} />
                </div>
                <Field label="Normalized Text" value={analysis.normalizedText} />
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Intent" value={INTENT_LABELS[analysis.intent]} />
                  <Field
                    label="Confidence"
                    value={`${Math.round(analysis.confidence * 100)}%`}
                  />
                </div>
                <div className="rounded-xl bg-secondary px-3 py-2">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
                    Entities
                  </p>
                  {analysis.entities.length ? (
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {analysis.entities.map((e) => (
                        <span
                          key={`${e.type}:${e.value}`}
                          className="rounded-lg border border-border px-2 py-1 text-[11px] text-foreground"
                        >
                          <span className="text-muted-foreground">{e.type}:</span> {e.value}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-1 text-sm text-muted-foreground">None detected</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Run Analyze to see language, dialect, code-mix and normalized meaning.
              </p>
            )}
          </Section>

          {/* 3. Pipeline */}
          <Section title="Pipeline" caption="Where the LDM sits between ASR and the local LLM.">
            <ol className="space-y-1.5">
              {PIPELINE.map((stage, i) => {
                const done = analysis !== null && i <= activeStage;
                return (
                  <li key={stage} className="flex items-center gap-2.5">
                    <span
                      className={`flex size-6 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold ${
                        done
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border bg-secondary text-muted-foreground"
                      }`}
                    >
                      {i + 1}
                    </span>
                    <span
                      className={`text-xs ${done ? "text-foreground" : "text-muted-foreground"}`}
                    >
                      {stage}
                    </span>
                  </li>
                );
              })}
            </ol>
          </Section>

          {/* 4. LLM Handoff */}
          <Section title="LLM Handoff" caption="Structured payload passed to the local LLM.">
            <pre className="max-h-64 overflow-auto rounded-xl bg-secondary p-3 text-[11px] leading-relaxed text-foreground">
              {handoff || "// Analyze an utterance to generate the handoff payload"}
            </pre>
            <button
              onClick={copy}
              disabled={!handoff}
              className="mt-3 w-full rounded-xl border border-border bg-secondary py-2.5 text-xs font-semibold text-foreground transition-colors active:bg-accent disabled:opacity-50"
            >
              {copied ? "Copied" : "Copy JSON"}
            </button>
          </Section>

          {/* 5. Dataset */}
          <Section
            title="Synthetic Dataset"
            caption={`${DATASET.length} local utterances · ${evaluation.intentCount} intents · Tamil, Tanglish, English, slang, code-mixing`}
          >
            <button
              onClick={() => setShowDataset((v) => !v)}
              className="w-full rounded-xl border border-border bg-secondary py-2.5 text-xs font-semibold text-foreground active:bg-accent"
            >
              {showDataset ? "Hide samples" : "View samples"}
            </button>
            {showDataset && (
              <ul className="mt-3 space-y-2">
                {DATASET.map((d) => (
                  <li key={d.id} className="rounded-xl border border-border px-3 py-2">
                    <button
                      onClick={() => {
                        setInput(d.input);
                        analyze(d.input);
                      }}
                      className="w-full text-left"
                    >
                      <p className="text-xs text-foreground">{d.input}</p>
                      <p className="mt-1 text-[11px] text-muted-foreground">→ {d.normalized}</p>
                      <p className="mt-1 text-[10px] tracking-wide text-muted-foreground uppercase">
                        {d.language} · {d.dialect} · {INTENT_LABELS[d.intent]}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Section>

          {/* 6. Evaluation */}
          <Section
            title="Synthetic Prototype Evaluation"
            caption="Indicative only — measured on the synthetic dataset with the rule-based LDM. Not a trained-model benchmark."
          >
            <div className="grid grid-cols-2 gap-2">
              <Metric
                label="Normalization Accuracy"
                value={`${Math.round(evaluation.normalizationAccuracy * 100)}%`}
              />
              <Metric
                label="Intent Accuracy"
                value={`${Math.round(evaluation.intentAccuracy * 100)}%`}
              />
              <Metric
                label="Dialect / Code-Mix Detection"
                value={`${Math.round(evaluation.dialectCodeMixAccuracy * 100)}%`}
              />
              <Metric
                label="Avg Processing Time"
                value={`${evaluation.avgProcessingMs.toFixed(2)} ms`}
              />
            </div>
          </Section>

          <p className="px-1 text-center text-[10px] leading-relaxed text-muted-foreground">
            Flow: Voice / ASR → LDM → Normalized meaning → Local LLM → Agent → Device actions.
            <br />
            The LDM normalizes language only; it does not execute actions or chat.
          </p>
        </main>
      </div>
    </div>
  );
}
