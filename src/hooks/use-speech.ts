import { useCallback, useEffect, useRef, useState } from "react";

type Rec = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: any) => void) | null;
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
};

function getCtor(): any {
  if (typeof window === "undefined") return null;
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** Live browser speech-to-text for the LDM playground (on-device, no cloud). */
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
    rec.onresult = (e: any) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      setTranscript(text.trim());
    };
    rec.onerror = (e: any) => {
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
