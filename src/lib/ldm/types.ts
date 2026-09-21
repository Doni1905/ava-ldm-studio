export type Intent =
  | "set_reminder"
  | "set_alarm"
  | "make_call"
  | "send_message"
  | "play_music"
  | "check_weather"
  | "open_app"
  | "navigate"
  | "device_control"
  | "search_info"
  | "smalltalk"
  | "unknown";

export interface LdmEntity {
  type: string;
  value: string;
}

export interface LdmAnalysis {
  rawText: string;
  language: string;
  dialect: string;
  codeMix: string;
  style: string;
  normalizedText: string;
  intent: Intent;
  entities: LdmEntity[];
  confidence: number;
  processingMs: number;
  trace: string[];
}

export interface UserProfile {
  name?: string;
  region?: string;
  preferredLanguage?: string;
}

/**
 * Replaceable LDM contract. Swap the rule-based implementation for a
 * trained model later without touching the UI.
 */
export interface LdmProcessor {
  analyzeUtterance(input: string, userProfile?: UserProfile): LdmAnalysis;
}

export interface LdmHandoff {
  normalized_text: string;
  language: string;
  dialect: string;
  code_mix: string;
  intent: Intent;
  entities: { type: string; value: string }[];
  confidence: number;
}

export function toHandoff(a: LdmAnalysis): LdmHandoff {
  return {
    normalized_text: a.normalizedText,
    language: a.language,
    dialect: a.dialect,
    code_mix: a.codeMix,
    intent: a.intent,
    entities: a.entities.map((e) => ({ type: e.type, value: e.value })),
    confidence: Number(a.confidence.toFixed(2)),
  };
}
