import { useCallback, useEffect, useRef, useState } from "react";

export interface SpeechResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}
export interface SpeechErrorEvent {
  error?: string;
}
export type SpeechConstructor = new () => Rec;
export type SpeechWindow = Window & {
  SpeechRecognition?: SpeechConstructor;
  webkitSpeechRecognition?: SpeechConstructor;
  webkitAudioContext?: typeof AudioContext;
};
export type Rec = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: SpeechResultEvent) => void) | null;
  onerror: ((e: SpeechErrorEvent) => void) | null;
  onend: (() => void) | null;
};

function getCtor(): SpeechConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as SpeechWindow;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** Browser speech-to-text may use vendor cloud services; not an offline ASR guarantee. */
export function useSpeech(lang = "ta-IN") {
  const recRef = useRef<Rec | null>(null);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    setSupported(!!getCtor());
  }, []);

  const start = useCallback(() => {
    const Ctor = getCtor();
    if (!Ctor) {
      setSupported(false);
      setError("Voice input is not supported in this browser. Try Chrome.");
      return;
    }
    setError(null);
    setTranscript("");
    const rec: Rec = new Ctor();
    rec.lang = lang;
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (e: SpeechResultEvent) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i]?.[0]?.transcript ?? "";
      setTranscript(text.trim());
    };
    rec.onerror = (e: SpeechErrorEvent) => {
      setError(
        e?.error === "not-allowed"
          ? "Microphone permission denied."
          : `Voice error: ${e?.error ?? "unknown"}`,
      );
      setListening(false);
    };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    rec.start();
    setListening(true);
  }, [lang]);

  const stop = useCallback(() => {
    recRef.current?.stop();
    setListening(false);
  }, []);

  return { listening, transcript, error, supported, start, stop, setTranscript };
}
