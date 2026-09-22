import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { AppShell } from "@/components/app-shell";
import { runEvaluation } from "@/lib/ldm/evaluation";
import { DATASET } from "@/lib/ldm/dataset";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Metrics — AVA Linguistic Dialect Model" },
      {
        name: "description",
        content:
          "Indicative prototype metrics for AVA's rule-based LDM on the synthetic Tamil/Tanglish dataset.",
      },
      { property: "og:title", content: "Metrics — AVA Linguistic Dialect Model" },
      {
        property: "og:description",
        content:
          "Normalization, intent and dialect/code-mix accuracy plus processing time for the LDM prototype.",
      },
    ],
  }),
  component: EvaluationPage,
});

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-3">
      <div className="flex items-baseline justify-between">
        <p className="text-xs text-foreground">{label}</p>
        <p className="text-sm font-semibold text-foreground">{Math.round(value * 100)}%</p>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-gradient-to-r from-primary to-chart-5"
          style={{ width: `${Math.min(100, Math.round(value * 100))}%` }}
        />
      </div>
    </div>
  );
}

function EvaluationPage() {
  const e = useMemo(() => runEvaluation(), []);

  return (
    <AppShell title="Prototype Metrics" subtitle="Rule-based LDM · synthetic dataset">
      <div className="space-y-2">
        <Bar label="Normalization accuracy" value={e.normalizationAccuracy} />
        <Bar label="Intent accuracy" value={e.intentAccuracy} />
        <Bar label="Dialect / code-mix detection" value={e.dialectCodeMixAccuracy} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        {[
          { label: "Samples", value: String(DATASET.length) },
          { label: "Intents", value: String(e.intentCount) },
          { label: "Avg ms", value: e.avgProcessingMs.toFixed(2) },
        ].map((m) => (
          <div
            key={m.label}
            className="rounded-2xl border border-border bg-card px-2 py-3 text-center"
          >
            <p className="text-base font-semibold text-foreground">{m.value}</p>
            <p className="mt-1 text-[10px] tracking-wide text-muted-foreground uppercase">
              {m.label}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground">
        Indicative only — measured on the synthetic dataset with the rule-based LDM. Not a
        trained-model benchmark.
      </p>
    </AppShell>
  );
}
