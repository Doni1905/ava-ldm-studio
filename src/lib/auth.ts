import { useEffect, useState } from "react";

export const DEMO_USERS = [
  { email: "demo@ava.ai", password: "ava1234", name: "Demo User" },
  { email: "harish@ava.ai", password: "ldm2026", name: "Harish" },
];

const KEY = "ava.session";

export interface Session {
  email: string;
  name: string;
}

export function signIn(email: string, password: string): Session | null {
  const u = DEMO_USERS.find(
    (d) => d.email.toLowerCase() === email.trim().toLowerCase() && d.password === password,
  );
  if (!u) return null;
  const session = { email: u.email, name: u.name };
  localStorage.setItem(KEY, JSON.stringify(session));
  return session;
}

export function signOut() {
  localStorage.removeItem(KEY);
}

export function readSession(): Session | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

/** Demo-only client session. Ready = hydrated check finished. */
export function useSession() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setSession(readSession());
    setReady(true);
  }, []);

  return { session, ready, setSession };
}
