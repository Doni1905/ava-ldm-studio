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
          .includes(query.toLowerCase());
      return matchDialect && matchIntent && matchQuery;
    });
  }, [query, selectedDialect, selectedIntent]);

  return (
    <AppShell
      title="Dataset & Corpus Specifications"
      subtitle="SPRINGLab IndicVoices-R Tamil & AVA Linguistic Benchmark"
    >
      <div className="space-y-6">
        {/* ========================================================================= */}
        {/* DATASET PROVENANCE & SPECS                                                */}
        {/* ========================================================================= */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <Database className="size-5 text-sky-400" />
                <h2 className="text-base font-semibold text-white">
                  IndicVoices-R Tamil & AVA LDM Benchmark
                </h2>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                Sociolinguistically stratified Tamil speech corpus from SPRINGLab (IIT Madras / AI4Bharat)
                paired with AVA's canonical dialect-normalization evaluation suite.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-slate-800 bg-slate-950 px-2.5 py-1 font-mono text-xs text-slate-300">
                Tamil (ta-IN) · 16 kHz
              </span>
            </div>
          </div>

          {/* 5 Core Required Metrics Grid */}
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {/* 1. Dataset Name */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center gap-1.5 text-slate-500">
                <FileSpreadsheet className="size-3.5" />
                <span className="text-[10px] font-medium tracking-wider uppercase">Dataset Name</span>
              </div>
              <p className="mt-1 text-sm font-bold text-white">IndicVoices-R</p>
              <p className="mt-0.5 text-[10px] text-slate-500">Tamil ASR & Dialect Subset</p>
            </div>

            {/* 2. Number of Samples */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center gap-1.5 text-slate-500">
                <Layers className="size-3.5" />
                <span className="text-[10px] font-medium tracking-wider uppercase">Total Samples</span>
              </div>
              <p className="mt-1 text-sm font-bold font-mono text-sky-400">5,030</p>
              <p className="mt-0.5 text-[10px] text-slate-500">5,000 audio + 30 test pairs</p>
            </div>

            {/* 3. Speakers */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center gap-1.5 text-slate-500">
                <Users className="size-3.5" />
                <span className="text-[10px] font-medium tracking-wider uppercase">Speakers</span>
              </div>
              <p className="mt-1 text-sm font-bold font-mono text-emerald-400">50+ Clusters</p>
              <p className="mt-0.5 text-[10px] text-slate-500">Speaker-disjoint partitions</p>
            </div>

            {/* 4. Districts */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center gap-1.5 text-slate-500">
                <MapPin className="size-3.5" />
                <span className="text-[10px] font-medium tracking-wider uppercase">Districts</span>
              </div>
              <p className="mt-1 text-sm font-bold font-mono text-amber-400">10 Districts</p>
              <p className="mt-0.5 text-[10px] text-slate-500">Chennai, Madurai, Kongu, etc.</p>
            </div>

            {/* 5. Splits */}
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center gap-1.5 text-slate-500">
                <PieChart className="size-3.5" />
                <span className="text-[10px] font-medium tracking-wider uppercase">Data Splits</span>
              </div>
              <p className="mt-1 text-sm font-bold font-mono text-indigo-400">70 / 15 / 15 %</p>
              <p className="mt-0.5 text-[10px] text-slate-500">3,500 / 750 / 750</p>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* DATASET SPLIT & DISTRICT MATRIX DETAILS                                   */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Partition Splits Table */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm backdrop-blur">
            <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
              Train / Validation / Test Partitions
            </h3>
            <p className="mt-1 text-[11px] text-slate-500">
              Stratified split ensuring zero speaker leakage across partitions:
            </p>

            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[10px] text-slate-400 uppercase">
                    <th className="py-2 font-medium">Split Name</th>
                    <th className="py-2 font-medium text-center">Clips Count</th>
                    <th className="py-2 font-medium text-center">Ratio</th>
                    <th className="py-2 font-medium text-center">Est. Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr>
                    <td className="py-2.5 font-medium text-emerald-400">Train Split</td>
                    <td className="py-2.5 text-center font-mono text-slate-200">3,500</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">70.0%</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">~4.8 hrs</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-medium text-sky-400">Validation Split</td>
                    <td className="py-2.5 text-center font-mono text-slate-200">750</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">15.0%</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">~1.0 hr</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-medium text-indigo-400">Test Split</td>
                    <td className="py-2.5 text-center font-mono text-slate-200">750</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">15.0%</td>
                    <td className="py-2.5 text-center font-mono text-slate-400">~1.0 hr</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Regional Dialect Coverage Table */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm backdrop-blur">
            <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
              District to Regional Dialect Mapping
            </h3>
            <p className="mt-1 text-[11px] text-slate-500">
              Corpus districts mapped to 5 sociolinguistic dialect zones:
            </p>

            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[10px] text-slate-400 uppercase">
                    <th className="py-2 font-medium">Dialect Region</th>
                    <th className="py-2 font-medium">Districts Covered</th>
                    <th className="py-2 font-medium text-center">Primary Characteristics</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr>
                    <td className="py-2.5 font-semibold text-sky-400">Chennai</td>
                    <td className="py-2.5 text-slate-300">Chennai, Kanchipuram, Tiruvallur</td>
                    <td className="py-2.5 text-center text-slate-400 text-[11px]">Madras Bashai, Tanglish slang</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-semibold text-sky-400">Kongu</td>
                    <td className="py-2.5 text-slate-300">Coimbatore, Erode, Salem, Tiruppur</td>
                    <td className="py-2.5 text-center text-slate-400 text-[11px]">Honorific 'ayya', 'yov', 'la'</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-semibold text-sky-400">Madurai</td>
                    <td className="py-2.5 text-slate-300">Madurai, Dindigul, Theni</td>
                    <td className="py-2.5 text-center text-slate-400 text-[11px]">Southern intonation, 'ennanga'</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-semibold text-sky-400">Nellai</td>
                    <td className="py-2.5 text-slate-300">Tirunelveli, Thoothukudi, Kanyakumari</td>
                    <td className="py-2.5 text-center text-slate-400 text-[11px]">Tirunelveli particles 'pa', 'ppa'</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 font-semibold text-sky-400">Standard</td>
                    <td className="py-2.5 text-slate-300">Broadcast / Romanised Tamil</td>
                    <td className="py-2.5 text-center text-slate-400 text-[11px]">Standard literary spoken Tamil</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* INTERACTIVE BENCHMARK UTTERANCE BROWSER                                   */}
        {/* ========================================================================= */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
                Canonical Evaluation Utterances ({filteredItems.length} of {DATASET.length})
              </h3>
              <p className="mt-0.5 text-[11px] text-slate-500">
                Ground-truth annotated speech pairs used for token F1 and intent accuracy benchmarks:
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
            <table className="w-full text-left text-xs">
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
                          item.codeMix ? "bg-amber-950/30 text-amber-400" : "bg-slate-800/40 text-slate-400"
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
