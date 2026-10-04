import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { runEvaluation, type EvalResult } from "@/lib/ldm/evaluation";

export const Route = createFileRoute("/evaluation")({ component: EvaluationDashboard });
export function EvaluationDashboard() {
  const [metrics, setMetrics] = useState<EvalResult | null>(null);
  return (
    <AppShell
      title="Synthetic text evaluation"
      subtitle="Browser rule engine, not an audio or trained-model benchmark"
    >
      <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5">
        <h2 className="text-lg font-semibold">
          Run the current engine on 30 hand-written examples
        </h2>
        <p className="text-sm text-slate-400">
          These examples also informed the rules. This is an in-sample smoke test, not held-out
          accuracy. ASR WER, acoustic dialect F1, LLM performance and RAM are not measured here.
        </p>
        <button
          className="rounded-xl bg-sky-600 px-4 py-3 text-sm"
          onClick={() => setMetrics(runEvaluation())}
        >
          Re-evaluate Benchmark
        </button>
        {metrics ? (
          <div className="grid grid-cols-2 gap-4">
            {[
              ["Normalization token F1", `${(metrics.normalizationAccuracy * 100).toFixed(1)}%`],
              ["Intent exact match", `${(metrics.intentAccuracy * 100).toFixed(1)}%`],
              ["Code-mix agreement", `${(metrics.dialectCodeMixAccuracy * 100).toFixed(1)}%`],
              ["Mean browser processing time", `${metrics.avgProcessingMs.toFixed(2)} ms`],
            ].map(([name, value]) => (
              <div key={name} className="rounded-xl bg-slate-950 p-4">
                <p className="text-xs text-slate-400">{name}</p>
                <p className="mt-2 text-xl text-sky-300">{value}</p>
              </div>
            ))}
            <p className="col-span-2 text-sm">
              {metrics.sampleCount} samples evaluated using the current browser engine. Timing
              varies by device.
            </p>
          </div>
        ) : (
          <p role="status" className="text-sm text-slate-400">
            Not run yet. No live metrics are available.
          </p>
        )}
      </section>
    </AppShell>
  );
}
