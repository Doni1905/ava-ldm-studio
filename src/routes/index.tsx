import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { SAMPLE_UTTERANCES, INTENT_LABELS } from "@/lib/ldm/dataset";
import { ruleBasedLdm } from "@/lib/ldm/processor";
import { toHandoff, type LdmAnalysis } from "@/lib/ldm/types";
import { AppShell } from "@/components/app-shell";
import { useSpeech } from "@/hooks/use-speech";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AVA Assistant — Linguistic Dialect Model" },
      {
        name: "description",
        content:
          "Speak or type Tamil, Tanglish or English. AVA's Linguistic Dialect Model normalizes meaning for the local LLM.",
      },
      { property: "og:title", content: "AVA Assistant — Linguistic Dialect Model" },
      {
        property: "og:description",
        content:
          "Voice-first LDM playground: dialect, slang and code-mix detection with normalized meaning and intent.",
      },
    ],
  }),
  component: AssistantScreen,
});

interface Turn {
  id: number;
  input: string;
  analysis: LdmAnalysis;
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-background/60 px-2.5 py-1.5">
      <p className="text-[9px] font-medium tracking-wider text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-0.5 text-[12px] leading-snug text-foreground">{value}</p>
    </div>
  );
}

function AssistantScreen() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [copied, setCopied] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const speech = useSpeech("ta-IN");

  const analyze = (text: string) => {
    const t = text.trim();
    if (!t) return;
    const analysis = ruleBasedLdm.analyzeUtterance(t, {
      region: "Tamil Nadu",
      preferredLanguage: "ta",
    });
    setTurns((p) => [...p, { id: Date.now(), input: t, analysis }]);
    setInput("");
  };

  // When voice recording ends with a transcript, analyze it automatically.
  useEffect(() => {
    if (!speech.listening && speech.transcript) {
      const t = speech.transcript;
      speech.setTranscript("");
      analyze(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speech.listening, speech.transcript]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length]);

  const copy = async (turn: Turn) => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(toHandoff(turn.analysis), null, 2));
      setCopied(turn.id);
      setTimeout(() => setCopied(null), 1600);
    } catch {
      setCopied(null);
    }
  };

  return (
    <AppShell title="AVA Assistant" subtitle="LDM · on-device normalization">
      {turns.length === 0 && (
        <div className="pt-6 pb-4">
          <h2 className="bg-gradient-to-r from-primary via-chart-2 to-chart-5 bg-clip-text text-2xl leading-snug font-semibold text-transparent">
            Vanakkam.
            <br />
            Sollunga, naan purinjukiren.
          </h2>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            Tap the mic and speak, or pick a demo utterance. The LDM only normalizes language — it
            never performs actions.
          </p>

          <p className="mt-6 mb-2 text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
            Demo utterances
          </p>
          <div className="space-y-2">
            {SAMPLE_UTTERANCES.map((s) => (
              <button
                key={s}
                onClick={() => analyze(s)}
                className="w-full rounded-2xl border border-border bg-card px-3.5 py-3 text-left text-xs leading-snug text-foreground active:bg-accent"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-4">
        {turns.map((turn) => (
          <div key={turn.id} className="space-y-2">
            <div className="flex justify-end">
              <p className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2.5 text-xs leading-snug text-primary-foreground">
                {turn.input}
              </p>
            </div>
            <div className="rounded-2xl rounded-bl-md border border-border bg-card p-3">
              <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
                Normalized meaning
              </p>
              <p className="mt-1 text-sm leading-snug text-foreground">
                {turn.analysis.normalizedText}
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Chip label="Language" value={turn.analysis.language} />
                <Chip label="Dialect" value={turn.analysis.dialect} />
                <Chip label="Code-Mix" value={turn.analysis.codeMix} />
                <Chip label="Style" value={turn.analysis.style} />
                <Chip
                  label="Intent"
                  value={INTENT_LABELS[turn.analysis.intent] ?? turn.analysis.intent}
                />
                <Chip
                  label="Confidence"
                  value={`${Math.round(turn.analysis.confidence * 100)}%`}
                />
              </div>
              {turn.analysis.entities.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {turn.analysis.entities.map((e) => (
                    <span
                      key={`${e.type}:${e.value}`}
                      className="rounded-lg border border-border px-2 py-1 text-[11px] text-foreground"
                    >
                      <span className="text-muted-foreground">{e.type}:</span> {e.value}
                    </span>
                  ))}
                </div>
              )}
              <details className="mt-3">
                <summary className="cursor-pointer text-[11px] text-muted-foreground">
                  LLM handoff JSON · {turn.analysis.processingMs.toFixed(2)} ms
                </summary>
                <pre className="mt-2 max-h-56 overflow-auto rounded-xl bg-secondary p-3 text-[11px] leading-relaxed text-foreground">
                  {JSON.stringify(toHandoff(turn.analysis), null, 2)}
                </pre>
                <button
                  onClick={() => copy(turn)}
                  className="mt-2 w-full rounded-xl border border-border bg-secondary py-2 text-[11px] font-semibold text-foreground active:bg-accent"
                >
                  {copied === turn.id ? "Copied" : "Copy JSON"}
                </button>
              </details>
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {/* Composer */}
      <div className="fixed bottom-16 left-1/2 z-20 w-full max-w-[430px] -translate-x-1/2 px-4 pb-2">
        {(speech.listening || speech.error) && (
          <p className="mb-2 text-center text-[11px] text-muted-foreground">
            {speech.listening
              ? speech.transcript || "Listening… speak now"
              : speech.error}
          </p>
        )}
        <div className="flex items-end gap-2 rounded-3xl border border-border bg-card p-2 shadow-lg">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                analyze(input);
              }
            }}
            rows={1}
            placeholder="Ask AVA in Tamil, Tanglish or English…"
            className="max-h-24 flex-1 resize-none bg-transparent px-2.5 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <button
            onClick={() => (speech.listening ? speech.stop() : speech.start())}
            aria-label={speech.listening ? "Stop recording" : "Start recording"}
            className={`flex size-11 shrink-0 items-center justify-center rounded-full transition-all ${
              speech.listening
                ? "animate-pulse bg-destructive text-destructive-foreground"
                : "bg-gradient-to-br from-primary to-chart-5 text-primary-foreground"
            }`}
          >
            <svg viewBox="0 0 24 24" fill="none" className="size-5">
              <path
                d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"
                fill="currentColor"
              />
              <path
                d="M5 11a7 7 0 0 0 14 0M12 18v3"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <button
            onClick={() => analyze(input)}
            disabled={!input.trim()}
            className="flex size-11 shrink-0 items-center justify-center rounded-full bg-secondary text-foreground disabled:opacity-40"
            aria-label="Analyze"
          >
            <svg viewBox="0 0 24 24" fill="none" className="size-5">
              <path
                d="M5 12h13m0 0-5-5m5 5-5 5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </AppShell>
  );
}
