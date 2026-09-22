import { Link, useNavigate } from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import { signOut, useSession } from "@/lib/auth";

const NAV = [
  { to: "/", label: "Assistant" },
  { to: "/pipeline", label: "Pipeline" },
  { to: "/dataset", label: "Dataset" },
  { to: "/evaluation", label: "Metrics" },
] as const;

export function AppShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  const { session, ready } = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (ready && !session) navigate({ to: "/login" });
  }, [ready, session, navigate]);

  if (!ready || !session) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-app-shell">
        <div className="size-10 animate-pulse rounded-2xl bg-primary/40" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-app-shell">
      <div className="mx-auto flex min-h-screen w-full max-w-[430px] flex-col bg-background">
        <header className="sticky top-0 z-20 border-b border-border bg-background/90 px-4 py-3 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-chart-5 text-[11px] font-bold text-primary-foreground">
              AVA
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-semibold text-foreground">{title}</h1>
              <p className="truncate text-[11px] text-muted-foreground">
                {subtitle ?? `Signed in as ${session.name}`}
              </p>
            </div>
            <button
              onClick={() => {
                signOut();
                navigate({ to: "/login" });
              }}
              className="rounded-full border border-border px-3 py-1.5 text-[11px] font-medium text-muted-foreground active:bg-accent"
            >
              Sign out
            </button>
          </div>
        </header>

        <main className="flex-1 px-4 pt-4 pb-28">{children}</main>

        <nav className="fixed bottom-0 z-20 w-full max-w-[430px] border-t border-border bg-background/95 px-2 py-2 backdrop-blur-xl">
          <div className="grid grid-cols-4 gap-1">
            {NAV.map((n) => (
              <Link
                key={n.to}
                to={n.to}
                className="rounded-xl px-2 py-2 text-center text-[11px] font-medium text-muted-foreground transition-colors"
                activeProps={{ className: "bg-secondary text-foreground" }}
                activeOptions={{ exact: n.to === "/" }}
              >
                {n.label}
              </Link>
            ))}
          </div>
        </nav>
      </div>
    </div>
  );
}
