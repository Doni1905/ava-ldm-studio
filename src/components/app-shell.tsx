import { Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";
import { signOut, useSession } from "@/lib/auth";
import { Activity, Database, BarChart3, GitFork, Mic } from "lucide-react";

const NAV = [
  { to: "/", label: "Playground", icon: Mic },
  { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { to: "/dataset", label: "Dataset", icon: Database },
  { to: "/pipeline", label: "Pipeline", icon: GitFork },
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
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    // Check if Python API server is running on port 8000
    fetch("http://127.0.0.1:8000/health", { mode: "cors" })
      .then((res) => (res.ok ? setApiOnline(true) : setApiOnline(false)))
      .catch(() => setApiOnline(false));
  }, []);

  if (!ready || !session) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950">
        <div className="size-10 animate-pulse rounded-2xl bg-sky-500/40" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 antialiased selection:bg-sky-500/20">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 sm:px-6 lg:px-8">
        {/* Academic Header */}
        <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/85 py-3.5 backdrop-blur-xl">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-xs font-bold text-white shadow-md shadow-sky-500/20">
                AVA
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base font-semibold tracking-tight text-white">{title}</h1>
                  <span className="hidden rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-400 sm:inline-block">
                    FYP Research Demo
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  {subtitle ?? "Accentric Virtual Assistant — Linguistic Dialect Model"}
                </p>
              </div>
            </div>

            {/* Navigation links & status */}
            <div className="flex items-center gap-3">
              <nav className="hidden items-center gap-1 sm:flex">
                {NAV.map((n) => {
                  const Icon = n.icon;
                  return (
                    <Link
                      key={n.to}
                      to={n.to}
                      className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors hover:bg-slate-900 hover:text-white"
                      activeProps={{
                        className: "bg-slate-900 text-sky-400 border border-slate-800",
                      }}
                      activeOptions={{ exact: n.to === "/" }}
                    >
                      <Icon className="size-3.5" />
                      {n.label}
                    </Link>
                  );
                })}
              </nav>

              {/* API Status Pill */}
              <div className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-2.5 py-1 text-[11px]">
                <span
                  className={`size-2 rounded-full ${
                    apiOnline === true
                      ? "bg-emerald-500 shadow-sm shadow-emerald-500/50"
                      : apiOnline === false
                        ? "bg-amber-500"
                        : "bg-slate-500"
                  }`}
                />
                <span className="text-slate-300">
                  {apiOnline === true
                    ? "API: 8000"
                    : apiOnline === false
                      ? "Local Engine"
                      : "Connecting..."}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 py-6">{children}</main>

        {/* Mobile Navigation */}
        <nav className="fixed bottom-0 left-0 z-30 w-full border-t border-slate-800 bg-slate-950/95 px-2 py-2 backdrop-blur-xl sm:hidden">
          <div className="grid grid-cols-4 gap-1">
            {NAV.map((n) => {
              const Icon = n.icon;
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  className="flex flex-col items-center gap-1 rounded-lg px-2 py-1.5 text-center text-[10px] font-medium text-slate-400 transition-colors"
                  activeProps={{ className: "bg-slate-900 text-sky-400" }}
                  activeOptions={{ exact: n.to === "/" }}
                >
                  <Icon className="size-4" />
                  {n.label}
                </Link>
              );
            })}
          </div>
        </nav>
      </div>
    </div>
  );
                      }
