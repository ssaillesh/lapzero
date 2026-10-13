import { describeFailure } from "../../api/errors";
import { useBackendHealth } from "../../hooks/useBackendHealth";
import { formatClockTime } from "../../lib/formatters";
import type { HealthStatus } from "../../lib/health";

const LABELS: Record<HealthStatus, string> = {
  checking: "Checking API",
  connected: "API connected",
  unavailable: "API unreachable",
  error: "API error",
};

const DOT: Record<HealthStatus, string> = {
  checking: "bg-faint motion-safe:animate-pulse-dot",
  connected: "bg-ok",
  unavailable: "bg-bad",
  error: "bg-warn",
};

/**
 * Connection indicator. The dot is the one status light on the page; it reflects a real
 * /api/health result and is never green before a check succeeds. Clicking re-checks.
 */
export function BackendStatus() {
  const { snapshot, recheck } = useBackendHealth();
  const { status, lastChecked, failure, inFlight } = snapshot;
  const detail =
    failure !== null
      ? describeFailure(failure).message
      : lastChecked !== null
        ? `Last checked ${formatClockTime(lastChecked)}`
        : "First check in progress";
  return (
    <button
      type="button"
      onClick={recheck}
      aria-busy={inFlight}
      title={`${detail} Click to check again.`}
      className="press group flex h-8 items-center gap-2 rounded-[var(--radius-ctl)] border border-line px-2.5 text-xs text-muted hover:border-line-strong hover:text-ink aria-busy:cursor-progress"
    >
      <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${DOT[status]}`} />
      <span role="status" className="whitespace-nowrap">
        {LABELS[status]}
      </span>
      <span className="sr-only">. {detail}</span>
    </button>
  );
}
