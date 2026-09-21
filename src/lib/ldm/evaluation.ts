import { DATASET } from "./dataset";
import { ruleBasedLdm } from "./processor";

function tokens(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .split(/\s+/)
    .filter(Boolean);
}

/** Token-level F1 between predicted and reference normalisation. */
function f1(pred: string, ref: string): number {
  const p = tokens(pred);
  const r = tokens(ref);
  if (!p.length || !r.length) return 0;
  const pool = [...r];
  let hit = 0;
  for (const t of p) {
    const i = pool.indexOf(t);
    if (i >= 0) {
      hit++;
      pool.splice(i, 1);
    }
  }
  const precision = hit / p.length;
  const recall = hit / r.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

export interface EvalResult {
  normalizationAccuracy: number;
  intentAccuracy: number;
  dialectCodeMixAccuracy: number;
  avgProcessingMs: number;
  sampleCount: number;
  intentCount: number;
}

export function runEvaluation(): EvalResult {
  let normSum = 0;
  let intentHits = 0;
  let mixHits = 0;
  let timeSum = 0;

  for (const item of DATASET) {
    const a = ruleBasedLdm.analyzeUtterance(item.input);
    normSum += f1(a.normalizedText, item.normalized);
    if (a.intent === item.intent) intentHits++;
    const predictedMix = !a.codeMix.startsWith("None");
    if (predictedMix === item.codeMix) mixHits++;
    timeSum += a.processingMs;
  }

  const n = DATASET.length;
  return {
    normalizationAccuracy: normSum / n,
    intentAccuracy: intentHits / n,
    dialectCodeMixAccuracy: mixHits / n,
    avgProcessingMs: timeSum / n,
    sampleCount: n,
    intentCount: new Set(DATASET.map((d) => d.intent)).size,
  };
}
