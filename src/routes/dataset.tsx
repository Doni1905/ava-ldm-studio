import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { DATASET, INTENT_LABELS } from "@/lib/ldm/dataset";

export const Route = createFileRoute("/dataset")({
  head: () => ({
    meta: [
      { title: "Dataset — AVA Linguistic Dialect Model" },
      {
        name: "description",
        content:
          "Synthetic Tamil, Tanglish and English utterance dataset with dialect, code-mix and intent labels.",
      },
      { property: "og:title", content: "Dataset — AVA Linguistic Dialect Model" },
      {
        property: "og:description",
        content: "Browse and filter the synthetic local-language dataset used by AVA's LDM.",
      },
    ],
  }),
  component: DatasetPage,
});

function DatasetPage() {
  const [query, setQuery] = useState("");
  const [intent, setIntent] = useState("all");

  const intents = useMemo(() => [...new Set(DATASET.map((d) => d.intent))], []);
  const rows = DATASET.filter(
    (d) =>
      (intent === "all" || d.intent === intent) &&
      (query.trim() === "" ||
        `${d.input} ${d.normalized} ${d.language} ${d.dialect}`
          .toLowerCase()
          .includes(query.toLowerCase())),
  );

  return (
    <AppShell title="Synthetic Dataset" subtitle={`${DATASET.length} labelled utterances`}>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search utterances, dialect…"
        maxLength={80}
        className="w-full rounded-2xl border border-input bg-secondary px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/40 focus:outline-none"
      />
      <div className="mt-3 flex flex-wrap gap-1.5">
        {["all", ...intents].map((i) => (
          <button
            key={i}
            onClick={() => setIntent(i)}
            className={`rounded-full border px-3 py-1.5 text-[11px] font-medium ${
              intent === i
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-secondary text-muted-foreground"
            }`}
          >
            {i === "all" ? "All" : (INTENT_LABELS[i] ?? i)}
          </button>
        ))}
      </div>

      <ul className="mt-4 space-y-2">
        {rows.map((d) => (
          <li key={d.id} className="rounded-2xl border border-border bg-card p-3">
            <p className="text-xs text-foreground">{d.input}</p>
            <p className="mt-1 text-[11px] text-muted-foreground">→ {d.normalized}</p>
            <p className="mt-1.5 text-[10px] tracking-wide text-muted-foreground uppercase">
              {d.language} · {d.dialect} · {INTENT_LABELS[d.intent] ?? d.intent}
              {d.codeMix ? " · code-mixed" : ""}
            </p>
          </li>
        ))}
        {rows.length === 0 && <li className="text-xs text-muted-foreground">No matches.</li>}
      </ul>
    </AppShell>
  );
}
