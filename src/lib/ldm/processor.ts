import type { Intent, LdmAnalysis, LdmEntity, LdmProcessor, UserProfile } from "./types";

/* ------------------------------------------------------------------ */
/* Lexicons                                                            */
/* ------------------------------------------------------------------ */

/** Multi-word Tanglish patterns, longest first. */
const PHRASES: [RegExp, string][] = [
  [/\bremind\s+pannunga\b/g, "remind me"],
  [/\bremind\s+pannu\b/g, "remind me"],
  [/\bnyabagam\s+padutthu\b/g, "remind me"],
  [/\bcall\s+pottu\s+kudu\b/g, "call"],
  [/\bphone\s+pottu\s+kudu\b/g, "call"],
  [/\bcall\s+pannu\b/g, "call"],
  [/\bmessage\s+anuppu\b/g, "send message to"],
  [/\bmessage\s+podu\b/g, "send message to"],
  [/\balarm\s+vai(kka|kku)?\b/g, "set alarm"],
  [/\balarm\s+set\s+pannu\b/g, "set alarm"],
  [/\bvazhi\s+kaatu\b/g, "directions to"],
  [/\broute\s+sollu\b/g, "route to"],
  [/\bsearch\s+pannu\b/g, "search"],
  [/\bopen\s+pannu\b/g, "open"],
  [/\boff\s+pannu\b/g, "turn off"],
  [/\bon\s+pannu\b/g, "turn on"],
  [/\bkammi\s+pannu\b/g, "reduce"],
  [/\bkuraikka?\b/g, "reduce"],
  [/\bkoothu\b/g, "increase"],
  [/\bethu?tthu\b/g, "increase"],
  [/\beppadi\s+iruku\b/g, "how is"],
  [/\beppadi\s+iruka\b/g, "how are you"],
  [/\bkaalaila\b/g, "in the morning"],
  [/\bmaniku\b/g, "o'clock"],
  [/\bgym\s+poga\b/g, "go to the gym"],
  [/\bsubmit\s+panna\b/g, "submit"],
  [/\bsaapida\b/g, "take"],
  [/\bmazhai\s+varuma\b/g, "will it rain"],
  [/\benna\s+panra\b/g, "what are you doing"],
  [/\blate\s+ah\s+varen\b/g, "I will come late"],
];

/** Single-token gloss. Empty string = discourse filler to drop. */
const WORDS: Record<string, string> = {
  dei: "",
  da: "",
  la: "",
  pa: "",
  ppa: "",
  machi: "",
  bruh: "",
  ayya: "",
  thala: "",
  nu: "",
  oru: "a",
  konjam: "some",
  sikkiram: "quickly",
  semma: "good",
  nalla: "good",
  nalaiku: "tomorrow",
  naalaikku: "tomorrow",
  naalaiku: "tomorrow",
  inniku: "today",
  innaiku: "today",
  ipo: "now",
  ippo: "now",
  raathiri: "tonight",
  amma: "mother",
  appa: "father",
  thambi: "my brother",
  akka: "my sister",
  anna: "",
  paatu: "song",
  padal: "song",
  song: "song",
  podu: "play",
  vanakkam: "hello",
  sollu: "tell me",
  sollunga: "tell me",
  evlo: "how much",
  enna: "what",
  naan: "I",
  ava: "AVA",
  ku: "",
  kku: "",
  ondru: "a",
  onnu: "a",
  vendum: "",
  please: "please",
};

const TIME_WORDS = ["tomorrow", "today", "tonight", "now", "morning", "evening", "night"];
const APPS = ["whatsapp", "camera", "youtube", "instagram", "gallery", "spotify", "maps", "gmail"];
const PLACES = ["chennai", "coimbatore", "bangalore", "madurai", "office", "home", "college"];
const DEVICE_TARGETS = ["volume", "brightness", "wifi", "bluetooth", "torch", "flashlight"];
const PEOPLE = ["mother", "father", "my brother", "my sister", "sister", "friend", "wife", "husband"];

const DIALECT_MARKERS: [string[], string][] = [
  [["dei", "machi", "bruh", "scene", "semma", "vaada"], "Chennai (Madras Bashai)"],
  [["thambi", "aama", "ennanga"], "Madurai"],
  [["la", "ayya", "yov"], "Kongu (Coimbatore)"],
  [["pa", "ppa", "ille"], "Nellai"],
];

const TAMIL_SCRIPT = /[\u0B80-\u0BFF]/;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}:\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

const ENGLISH_HINTS = new Set([
  "remind",
  "me",
  "to",
  "the",
  "set",
  "alarm",
  "call",
  "message",
  "send",
  "play",
  "some",
  "open",
  "weather",
  "today",
  "tomorrow",
  "morning",
  "evening",
  "volume",
  "brightness",
  "wifi",
  "off",
  "on",
  "search",
  "route",
  "traffic",
  "meeting",
  "assignment",
  "submit",
  "gym",
  "music",
  "song",
  "songs",
  "please",
  "medicine",
  "bill",
  "electricity",
  "pay",
  "population",
  "sister",
  "friend",
  "phone",
  "camera",
  "am",
  "pm",
  "an",
  "a",
  "for",
  "is",
  "how",
  "tamil",
  "directions",
  "show",
  "turn",
  ...APPS,
  ...PLACES,
]);

function isLocalToken(t: string): boolean {
  if (ENGLISH_HINTS.has(t)) return false;
  if (/^\d/.test(t)) return false;
  return true;
}

function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function cleanup(words: string[]): string {
  return words.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

/* ------------------------------------------------------------------ */
/* Rule-based LDM                                                      */
/* ------------------------------------------------------------------ */

function detectIntent(gloss: string): { intent: Intent; strength: number } {
  const g = ` ${gloss} `;
  const rules: [Intent, RegExp, number][] = [
    ["set_reminder", /\bremind\b/, 0.95],
    ["set_alarm", /\b(set )?alarm\b/, 0.93],
    ["send_message", /\bsend message\b|\bmessage\b/, 0.88],
    ["make_call", /\bcall\b/, 0.92],
    ["play_music", /\bplay\b|\bsong\b|\bmusic\b/, 0.9],
    ["check_weather", /\bweather\b|\brain\b/, 0.9],
    ["navigate", /\bdirections\b|\broute\b/, 0.9],
    ["open_app", /\bopen\b/, 0.89],
    ["device_control", /\b(volume|brightness|wifi|bluetooth|torch)\b/, 0.91],
    ["search_info", /\bsearch\b|\bhow much\b|\btell me\b|\btraffic\b/, 0.82],
    ["smalltalk", /\bhello\b|\bhow are you\b|\bwhat are you doing\b/, 0.85],
  ];
  for (const [intent, re, strength] of rules) {
    if (re.test(g)) return { intent, strength };
  }
  return { intent: "unknown", strength: 0.4 };
}

function extractEntities(gloss: string, raw: string): LdmEntity[] {
  const out: LdmEntity[] = [];
  const g = gloss.toLowerCase();
  const timeOfDay = TIME_WORDS.filter((w) => g.includes(w));
  const clock = g.match(/\b\d{1,2}(:\d{2})?\s?(am|pm)?\b/);
  if (clock) out.push({ type: "time", value: clock[0].trim() });
  for (const t of timeOfDay) out.push({ type: "date_time", value: t });
  for (const p of PEOPLE) if (g.includes(p)) out.push({ type: "person", value: p });
  for (const a of APPS) if (g.includes(a)) out.push({ type: "app", value: titleCase(a) });
  for (const p of PLACES) if (g.includes(p)) out.push({ type: "location", value: titleCase(p) });
  for (const d of DEVICE_TARGETS) if (g.includes(d)) out.push({ type: "device_setting", value: d });
  if (/\bassignment\b/.test(g)) out.push({ type: "task", value: "assignment" });
  if (/\bmedicine\b/.test(g)) out.push({ type: "task", value: "medicine" });
  if (/\bmeeting\b/.test(g)) out.push({ type: "task", value: "meeting" });
  if (/ilayaraja|rahman|anirudh/i.test(raw))
    out.push({ type: "artist", value: titleCase(raw.match(/ilayaraja|rahman|anirudh/i)![0]) });
  const seen = new Set<string>();
  return out.filter((e) => {
    const k = `${e.type}:${e.value}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

function buildNormalized(intent: Intent, gloss: string, entities: LdmEntity[]): string {
  const words = gloss.split(" ").filter(Boolean);
  const time = entities.filter((e) => e.type === "time" || e.type === "date_time").map((e) => e.value);
  const timePhrase = time.length ? ` ${time.reverse().join(" ")}` : "";
  const person = entities.find((e) => e.type === "person")?.value;
  const app = entities.find((e) => e.type === "app")?.value;
  const place = entities.find((e) => e.type === "location")?.value;
  const setting = entities.find((e) => e.type === "device_setting")?.value;

  const without = (drop: string[]) =>
    cleanup(words.filter((w) => !drop.includes(w) && !TIME_WORDS.includes(w)));

  switch (intent) {
    case "set_reminder": {
      const task = without(["remind", "me", "to", "in", "the", "o'clock", "a", "some"]);
      return `Remind me to ${task || "do this"}${timePhrase}.`;
    }
    case "set_alarm": {
      const t = cleanup(words.filter((w) => /^\d/.test(w) || ["am", "pm"].includes(w)));
      const day = words.filter((w) => TIME_WORDS.includes(w)).join(" ");
      return `Set an alarm for ${[t, day].filter(Boolean).join(" ") || "the morning"}.`;
    }
    case "make_call":
      return `Call ${person ?? without(["call", "to", "a"]) || "the contact"}.`;
    case "send_message": {
      const body = without(["send", "message", "to", "a"]);
      const target = person ? ` to ${person}` : "";
      return `Send a message${target}${body && !person?.includes(body) ? ` saying ${body}` : ""}.`;
    }
    case "play_music": {
      const what = without(["play", "podu"]) || "music";
      return `Play ${what}.`;
    }
    case "check_weather":
      return `${titleCase(without([]))}?`;
    case "open_app":
      return `Open ${app ?? without(["open", "the", "a"]) || "the app"}.`;
    case "navigate":
      return `Show me directions to ${place ?? without(["directions", "route", "to"]) || "the destination"}.`;
    case "device_control": {
      const verb = words.find((w) => ["reduce", "increase", "turn off", "turn on"].includes(w));
      const compound = /turn off/.test(gloss) ? "Turn off" : /turn on/.test(gloss) ? "Turn on" : null;
      const action = compound ?? titleCase(verb ?? "adjust");
      return `${action} the ${setting ?? "setting"}${/some/.test(gloss) && !compound ? " a little" : ""}.`;
    }
    case "search_info":
      return `${titleCase(without([]))}.`;
    case "smalltalk":
      return `${titleCase(without([]))}?`;
    default:
      return `${titleCase(cleanup(words))}.`;
  }
}

export const ruleBasedLdm: LdmProcessor = {
  analyzeUtterance(input: string, userProfile?: UserProfile): LdmAnalysis {
    const start =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    const raw = input.trim();
    const trace: string[] = [];

    const tokens = tokenize(raw);
    const localCount = tokens.filter(isLocalToken).length;
    const englishCount = tokens.length - localCount;
    const hasTamilScript = TAMIL_SCRIPT.test(raw);
    const localRatio = tokens.length ? localCount / tokens.length : 0;

    let language: string;
    if (hasTamilScript) language = "Tamil (Tamil script)";
    else if (localRatio > 0.75) language = "Tamil (romanised)";
    else if (localRatio > 0.15) language = "Tanglish";
    else language = "English";
    trace.push(`language: ${language} (${Math.round(localRatio * 100)}% local tokens)`);

    let dialect = language === "English" ? "Indian English" : "Standard Tamil";
    for (const [markers, name] of DIALECT_MARKERS) {
      if (tokens.some((t) => markers.includes(t))) {
        dialect = name;
        break;
      }
    }
    if (dialect === "Standard Tamil" && userProfile?.region) {
      trace.push(`profile region hint: ${userProfile.region}`);
    }
    trace.push(`dialect: ${dialect}`);

    const mixRatio = tokens.length ? englishCount / tokens.length : 0;
    const codeMix =
      localCount > 0 && englishCount > 0
        ? `Intra-sentential Tamil–English (${Math.round(mixRatio * 100)}% English)`
        : "None (monolingual)";
    trace.push(`code-mix: ${codeMix}`);

    const slangHit = tokens.some((t) =>
      ["dei", "da", "machi", "bruh", "semma", "la", "pa", "ayya", "thala"].includes(t),
    );
    const polite = /please|pannunga|sollunga|kudunga/i.test(raw);
    const style = slangHit
      ? "Informal / slang"
      : polite
        ? "Polite / respectful"
        : "Neutral";
    trace.push(`style: ${style}`);

    let gloss = ` ${tokens.join(" ")} `;
    for (const [re, rep] of PHRASES) gloss = gloss.replace(re, ` ${rep} `);
    gloss = cleanup(
      gloss
        .split(/\s+/)
        .map((t) => (t in WORDS ? WORDS[t] : t))
        .filter(Boolean),
    );
    trace.push(`gloss: ${gloss}`);

    const { intent, strength } = detectIntent(gloss);
    const entities = extractEntities(gloss, raw);
    const normalizedText = buildNormalized(intent, gloss, entities);
    trace.push(`intent: ${intent} · entities: ${entities.length}`);

    const unknownTokens = gloss.split(" ").filter((t) => isLocalToken(t)).length;
    const coverage = gloss ? 1 - unknownTokens / Math.max(gloss.split(" ").length, 1) : 0;
    const confidence = Math.max(
      0.35,
      Math.min(0.98, strength * 0.7 + coverage * 0.3 + (entities.length ? 0.03 : 0)),
    );

    const end = typeof performance !== "undefined" ? performance.now() : Date.now();

    return {
      rawText: raw,
      language,
      dialect,
      codeMix,
      style,
      normalizedText,
      intent,
      entities,
      confidence,
      processingMs: Math.max(0.1, end - start),
      trace,
    };
  },
};
