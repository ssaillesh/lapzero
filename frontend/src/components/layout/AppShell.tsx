import type { ReactNode } from "react";

import { api } from "../../api/client";
import { useBackendHealth } from "../../hooks/useBackendHealth";
import { formatClockTime } from "../../lib/formatters";
import type { Section } from "../../lib/lastVisited";
import { lastHrefFor, sectionOf } from "../../lib/lastVisited";
import type { Route } from "../../lib/router";
import { Link } from "../common/Link";
import { BackendStatus } from "./BackendStatus";

const NAV: readonly { section: Section; label: string }[] = [
  { section: "leaderboard", label: "Leaderboard" },
  { section: "tracks", label: "Track Explorer" },
  { section: "governance", label: "Governance" },
];

function Wordmark() {
  return (
    <Link
      href="/leaderboard"
      className="flex shrink-0 items-center gap-2.5 rounded-[var(--radius-ctl)]"
      aria-label="LapZero home"
    >
      <span aria-hidden="true" className="h-4 w-[7px] -skew-x-[18deg] bg-accent" />
      <span className="text-[15px] font-semibold tracking-[-0.03em] text-ink">
        Lap<span className="text-muted">Zero</span>
      </span>
    </Link>
  );
}

function NavLinks({ active, className }: { active: Section | null; className: string }) {
  return (
    <ul className={className}>
      {NAV.map(({ section, label }) => {
        const current = section === active;
        return (
          <li key={section} className="shrink-0">
            <Link
              href={lastHrefFor(section)}
              aria-current={current ? "page" : undefined}
              className={`relative flex h-full items-center px-3 text-[13px] font-medium transition-colors duration-150 ${
                current ? "text-ink" : "text-muted hover:text-ink"
              }`}
            >
              {label}
              {current && (
                <span aria-hidden="true" className="absolute inset-x-3 bottom-0 h-0.5 bg-accent" />
              )}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function entityOf(route: Route): string | null {
  if (route.name === "track") {
    return route.lapId === null
      ? `Track ${route.trackId}`
      : `Track ${route.trackId} / lap ${route.lapId}`;
  }
  if (route.name === "leaderboard" && route.params.trackId !== null)
    return `Track ${route.params.trackId}`;
  return null;
}

function StatusBar({ route }: { route: Route }) {
  const { snapshot } = useBackendHealth();
  const entity = entityOf(route);
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-x-6 gap-y-1 px-4 py-2.5 font-mono text-[11px] text-faint sm:px-6">
        <span className="truncate">API {api.baseUrl}</span>
        <span>Contract {snapshot.contractVersion ?? "unknown"}</span>
        <span>
          {snapshot.lastChecked === null
            ? "Not checked yet"
            : `Checked ${formatClockTime(snapshot.lastChecked)}`}
        </span>
        {entity !== null && <span className="min-w-0 truncate sm:ml-auto">{entity}</span>}
      </div>
    </footer>
  );
}

/** Persistent frame: brand, navigation, connection status, the view, and a status bar. */
export function AppShell({ route, children }: { route: Route; children: ReactNode }) {
  const active = sectionOf(route);
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only z-30 bg-accent px-3 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>
      {/* z-index scale: header 20, skip link 30. Nothing else stacks. */}
      <header className="sticky top-0 z-20 border-b border-line bg-ground">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-6 px-4 sm:px-6">
          <Wordmark />
          <nav aria-label="Primary" className="hidden h-full md:block">
            <NavLinks active={active} className="flex h-full items-stretch" />
          </nav>
          <div className="ml-auto">
            <BackendStatus />
          </div>
        </div>
        <nav aria-label="Primary" className="border-t border-line md:hidden">
          <NavLinks active={active} className="flex h-11 items-stretch overflow-x-auto px-1" />
        </nav>
      </header>
      <main
        id="main"
        tabIndex={-1}
        className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-6 outline-none sm:px-6 lg:py-8"
      >
        {children}
      </main>
      <StatusBar route={route} />
    </div>
  );
}
