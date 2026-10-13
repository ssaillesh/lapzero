import type { ReactNode } from "react";

import { ApiError, describeFailure } from "../../api/errors";
import { api } from "../../api/client";

/** A placeholder block shaped like the content it stands for. */
export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden="true" className={`skeleton motion-safe:animate-shimmer ${className}`} />;
}

/** Rows of skeleton lines for tables and lists. */
export function SkeletonRows({ rows, className = "" }: { rows: number; className?: string }) {
  return (
    <div role="status" aria-label="Loading" className={`grid gap-2 ${className}`}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-9" />
      ))}
    </div>
  );
}

interface EmptyStateProps {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}

/** A successful response with nothing to show, and how to get something to show. */
export function EmptyState({ title, children, action }: EmptyStateProps) {
  return (
    <div className="arrive border border-dashed border-line-strong px-6 py-10 rounded-[var(--radius-ctl)]">
      <p className="text-[15px] font-medium text-ink">{title}</p>
      {children !== undefined && <div className="mt-2 max-w-[62ch] text-muted">{children}</div>}
      {action !== undefined && <div className="mt-5">{action}</div>}
    </div>
  );
}

interface ErrorStateProps {
  error: ApiError | Error;
  onRetry?: () => void;
  /** Override the heading, e.g. "Lap not found" for a 404 on a replay. */
  title?: string;
  /** Extra explanation under the message. */
  children?: ReactNode;
}

/** A failed request: what failed in plain words, and a retry when retrying can help. */
export function ErrorState({ error, onRetry, title, children }: ErrorStateProps) {
  const described =
    error instanceof ApiError
      ? describeFailure(error.failure)
      : {
          title: "Something went wrong",
          message: "The dashboard hit an unexpected error. Reload the page to try again.",
        };
  const unreachable =
    error instanceof ApiError &&
    (error.failure.kind === "network" || error.failure.kind === "timeout");
  return (
    <div role="alert" className="arrive border-l-2 border-bad bg-panel px-5 py-4">
      <p className="font-medium text-ink">{title ?? described.title}</p>
      <p className="mt-1 max-w-[70ch] text-muted">{described.message}</p>
      {unreachable && <p className="mt-2 font-mono text-xs text-faint">API base: {api.baseUrl}</p>}
      {children !== undefined && <div className="mt-2 max-w-[70ch] text-muted">{children}</div>}
      {onRetry !== undefined && (
        <button type="button" onClick={onRetry} className="press mt-4 btn-secondary">
          Try again
        </button>
      )}
    </div>
  );
}

/** A quiet note about a capability the API does not provide. */
export function Notice({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "warn";
}) {
  const border = tone === "warn" ? "border-warn" : "border-line-strong";
  return <div className={`border-l-2 ${border} py-1 pl-3 text-[13px] text-muted`}>{children}</div>;
}
