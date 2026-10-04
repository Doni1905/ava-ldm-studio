import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { DATASET, INTENT_LABELS } from "@/lib/ldm/dataset";
import {
  Database,
  Users,
  MapPin,
  PieChart,
  Search,
  Filter,
  Layers,
  FileSpreadsheet,
  CheckCircle,
} from "lucide-react";

export const Route = createFileRoute("/dataset")({
  head: () => ({
    meta: [
      { title: "Dataset Explorer — IndicVoices-R Tamil & AVA LDM Studio" },
      {
        name: "description",
        content:
          "Academic dataset specifications for IndicVoices-R Tamil and AVA Benchmark: sample counts, speaker clusters, district stratification, and train/val/test splits.",
      },
    ],
  }),
  component: DatasetExplorer,
});

export function DatasetExplorer() {
  const [query, setQuery] = useState("");
  const [selectedDialect, setSelectedDialect] = useState<string>("all");
  const [selectedIntent, setSelectedIntent] = useState<string>("all");

  const dialects = useMemo(() => ["all", ...new Set(DATASET.map((d) => d.dialect))], []);
  const intents = useMemo(() => ["all", ...new Set(DATASET.map((d) => d.intent))], []);

  const filteredItems = useMemo(() => {
    return DATASET.filter((d) => {
      const matchDialect = selectedDialect === "all" || d.dialect === selectedDialect;
      const matchIntent = selectedIntent === "all" || d.intent === selectedIntent;
      const matchQuery =
        !query.trim() ||
        `${d.input} ${d.normalized} ${d.dialect} ${d.language}`
          .toLowerCase()
          .includes(query.trim().toLowerCase());
      return matchDialect && matchIntent && matchQuery;
    });
  }, [query, selectedDialect, selectedIntent]);

  return (
    <AppShell
      title="Synthetic dataset explorer"
      subtitle="Bundled Tamil, Tanglish and English examples"
    >
      <div className="space-y-6">
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold">30 hand-written text examples</h2>
          <p className="mt-2 text-sm text-slate-400">
            Only the synthetic AVA examples below are bundled. No IndicVoices audio, speakers,
            district metadata or train/test split is included in this app. The rules were developed
            using these examples; this is not an independent test corpus.
          </p>
        </section>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
                Canonical Evaluation Utterances ({filteredItems.length} of {DATASET.length})
              </h3>
              <p className="mt-0.5 text-[11px] text-slate-500">
                Hand-written text pairs used for token F1 and intent agreement benchmarks:
              </p>
            </div>

            {/* Filter controls */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 size-3.5 text-slate-500" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Filter utterances..."
                  className="rounded-lg border border-slate-800 bg-slate-950 py-1.5 pl-8 pr-3 text-xs text-slate-200 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
                />
              </div>

              <select
                value={selectedDialect}
                onChange={(e) => setSelectedDialect(e.target.value)}
                className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-300 focus:border-sky-500 focus:outline-none"
              >
                {dialects.map((d) => (
                  <option key={d} value={d}>
                    {d === "all" ? "All Dialects" : d}
                  </option>
                ))}
              </select>

              <select
                value={selectedIntent}
                onChange={(e) => setSelectedIntent(e.target.value)}
                className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-300 focus:border-sky-500 focus:outline-none"
              >
                {intents.map((i) => (
                  <option key={i} value={i}>
                    {i === "all" ? "All Intents" : (INTENT_LABELS[i] ?? i)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Table */}
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-[760px] w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] text-slate-400 uppercase">
                  <th className="py-2.5 font-medium">ID</th>
                  <th className="py-2.5 font-medium">Raw Input Utterance</th>
                  <th className="py-2.5 font-medium">Normalized English Target</th>
                  <th className="py-2.5 font-medium text-center">Dialect</th>
                  <th className="py-2.5 font-medium text-center">Intent</th>
                  <th className="py-2.5 font-medium text-center">Code-Mix</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {!filteredItems.length && (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-slate-400">
                      No matching examples. Clear your filters to see the dataset.
                    </td>
                  </tr>
                )}
                {filteredItems.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 font-mono text-slate-500">#{item.id}</td>
                    <td className="py-2.5 font-mono text-slate-300">{item.input}</td>
                    <td className="py-2.5 font-medium text-sky-300">{item.normalized}</td>
                    <td className="py-2.5 text-center">
                      <span className="rounded border border-slate-800 bg-slate-950 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
                        {item.dialect}
                      </span>
                    </td>
                    <td className="py-2.5 text-center">
                      <span className="rounded border border-emerald-900/40 bg-emerald-950/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
                        {item.intent.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 text-center">
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                          item.codeMix
                            ? "bg-amber-950/30 text-amber-400"
                            : "bg-slate-800/40 text-slate-400"
                        }`}
                      >
                        {item.codeMix ? "Yes" : "No"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
