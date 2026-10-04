import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { DEMO_USERS, signIn } from "@/lib/auth";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — AVA Accentric Virtual Assistant" },
      {
        name: "description",
        content: "Sign in to the AVA demo to test the Linguistic Dialect Model with voice input.",
      },
      { property: "og:title", content: "Sign in — AVA Accentric Virtual Assistant" },
      {
        property: "og:description",
        content: "Demo sign-in for AVA's Tamil/Tanglish Linguistic Dialect Model prototype.",
      },
    ],
  }),
  component: LoginScreen,
});

function LoginScreen() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const s = signIn(email, password);
    if (!s) return setError("Invalid credentials. Use the demo login below.");
    navigate({ to: "/" });
  };

  const fill = (i: number) => {
    const user = DEMO_USERS[i];
    if (!user) return;
    setEmail(user.email);
    setPassword(user.password);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-app-shell">
      <div className="mx-auto flex min-h-screen w-full max-w-[430px] flex-col justify-center bg-background px-6 py-10">
        <div className="mb-8 text-center">
          <div className="mx-auto flex size-16 items-center justify-center rounded-3xl bg-gradient-to-br from-primary to-chart-5 text-lg font-bold text-primary-foreground">
            AVA
          </div>
          <h1 className="mt-5 bg-gradient-to-r from-primary via-chart-2 to-chart-5 bg-clip-text text-2xl font-semibold text-transparent">
            Accentric Virtual Assistant
          </h1>
          <p className="mt-2 text-xs text-muted-foreground">
            Linguistic Dialect Model · Tamil · Tanglish · dialect · slang
          </p>
        </div>

        <form onSubmit={submit} className="space-y-3">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            maxLength={120}
            className="w-full rounded-2xl border border-input bg-secondary px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/40 focus:outline-none"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            maxLength={64}
            className="w-full rounded-2xl border border-input bg-secondary px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:ring-2 focus:ring-ring/40 focus:outline-none"
          />
          {error && <p className="text-xs text-destructive">{error}</p>}
          <button
            type="submit"
            className="w-full rounded-2xl bg-gradient-to-r from-primary to-chart-5 py-3.5 text-sm font-semibold text-primary-foreground active:opacity-85"
          >
            Sign in
          </button>
        </form>

        <div className="mt-8 rounded-2xl border border-border bg-card p-4">
          <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">
            Sample credentials
          </p>
          <div className="mt-2 space-y-2">
            {DEMO_USERS.map((u, i) => (
              <button
                key={u.email}
                onClick={() => fill(i)}
                className="w-full rounded-xl bg-secondary px-3 py-2 text-left active:bg-accent"
              >
                <p className="text-xs text-foreground">{u.email}</p>
                <p className="text-[11px] text-muted-foreground">password: {u.password}</p>
              </button>
            ))}
          </div>
          <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
            Demo login only — no account or server is used.
          </p>
        </div>
      </div>
    </div>
  );
      }
