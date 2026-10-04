import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";

export const Route = createFileRoute("/pipeline")({
  head: () => ({
    meta: [
      { title: "Pipeline — AVA Linguistic Dialect Model" },
      {
        name: "description",
        content:
          "How AVA's LDM sits between Android voice input and the local LLM: detection, normalization, intent, handoff.",
      },
      { property: "og:title", content: "Pipeline — AVA Linguistic Dialect Model" },
      {
        property: "og:description",
        content: "Voice → LDM → normalized meaning → local LLM → agent → device actions.",
      },
    ],
  }),
  component: PipelinePage,
});

const STAGES = [
  {
    name: "Voice / ASR",
    note: "Desktop only: microphone uses browser speech services or a separately running Python API. APK accepts text only.",
  },
  { name: "Language Detection", note: "Identifies Tamil, romanised Tamil, English or mixed." },
  { name: "Dialect", note: "Chennai, Madurai, Kongu, Nellai, standard Tamil." },
  { name: "Code-Mix", note: "Measures Tamil-English mixing and slang density." },
  { name: "Normalization", note: "Rewrites slang and informal phrasing into clean meaning." },
  { name: "Intent + Entities", note: "Resolves intent and extracts time, person, app, place." },
  {
    name: "LLM Handoff",
    note: "Structured JSON is prepared for copying. No LLM is bundled or called.",
  },
  {
    name: "Agent / Actions",
    note: "Future integration only. This app performs no device actions.",
  },
];

function PipelinePage() {
  return (
    <AppShell title="LDM Pipeline" subtitle="Between ASR and the local LLM">
      <ol className="space-y-2">
        {STAGES.map((s, i) => (
          <li key={s.name} className="flex gap-3 rounded-2xl border border-border bg-card p-3">
            <span
              className={`flex size-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                i === STAGES.length - 1
                  ? "border border-border bg-secondary text-muted-foreground"
                  : "bg-gradient-to-br from-primary to-chart-5 text-primary-foreground"
              }`}
            >
              {i + 1}
            </span>
            <div>
              <p className="text-sm font-medium text-foreground">{s.name}</p>
              <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">{s.note}</p>
            </div>
          </li>
        ))}
      </ol>
      <p className="mt-4 rounded-2xl border border-border bg-card p-3 text-[11px] leading-relaxed text-muted-foreground">
        The LDM normalizes language only. It never executes actions and never behaves as a chatbot.
      </p>
    </AppShell>
  );
}
