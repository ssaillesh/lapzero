import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import type { ChampionResponse } from "../../api/types";
import { formatDateTime, majorVersion } from "../../lib/formatters";

/** Copy a value to the clipboard, with a short confirmation in place of the label. */
function CopyButton({ value, label }: { value: string; label: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  const copy = () => {
    navigator.clipboard.writeText(value).then(
      () => {
        setState("copied");
      },
      () => {
        setState("failed");
      },
    );
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setState("idle");
    }, 1600);
  };
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={`Copy ${label}`}
      className="press shrink-0 rounded-[2px] border border-line px-1.5 py-px font-mono text-[10.5px] text-muted hover:border-line-strong hover:text-ink"
    >
      <span aria-live="polite">
        {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}
      </span>
    </button>
  );
}

interface CompatibilityProps {
  runContract: string | null;
  apiContract: string | null;
}

/** Informational: does the run's logged contract share the API's major version? */
export function ContractCompatibility({ runContract, apiContract }: CompatibilityProps) {
  if (runContract === null)
    return <span className="text-muted">The training run logged no contract version.</span>;
  if (apiContract === null)
    return <span className="text-muted">Waiting for the API&apos;s contract version.</span>;
  const run = majorVersion(runContract);
  const live = majorVersion(apiContract);
  if (run === null || live === null)
    return <span className="text-warn">A version is not semantic: cannot compare.</span>;
  if (run === live) return <span className="text-ok">Same major version ({run}) as the API</span>;
  return (
    <span className="text-warn">
      Major {run} differs from the API&apos;s {live}: the API rejects laps of another major, and the
      gate refuses to compare across majors.
    </span>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-baseline gap-3 py-2.5">
      <dt className="label">{label}</dt>
      <dd className="min-w-0 text-[13px] text-ink">{children}</dd>
    </div>
  );
}

/** Who the champion is: registry identity, the run behind it, and its contract version. */
export function ChampionMetadata({
  champion,
  apiContract,
}: {
  champion: ChampionResponse;
  apiContract: string | null;
}) {
  return (
    <section
      aria-labelledby="champion-heading"
      className="arrive grid overflow-hidden rounded-[var(--radius-ctl)] border border-line lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]"
    >
      <div className="relative flex flex-col justify-between gap-8 border-b border-line bg-panel p-6 lg:border-b-0 lg:border-r">
        <span aria-hidden="true" className="absolute inset-y-0 left-0 w-0.5 bg-accent" />
        <div>
          <p className="label">Champion alias</p>
          <h2 id="champion-heading" className="mt-2 font-mono text-[15px] text-muted">
            {champion.model_name}
          </h2>
        </div>
        <div>
          <p className="font-mono text-[64px] font-medium leading-none tracking-[-0.04em] text-accent sm:text-[80px]">
            v{champion.version}
          </p>
          <p className="mt-3 text-[13px] text-muted">
            Registered{" "}
            <time dateTime={champion.created_at}>{formatDateTime(champion.created_at)}</time>
          </p>
        </div>
      </div>
      <dl className="divide-y divide-line px-6 py-2">
        <Field label="Registered model">
          <span className="font-mono">{champion.model_name}</span>
        </Field>
        <Field label="Version">
          <span className="font-mono">{champion.version}</span>
        </Field>
        <Field label="MLflow run">
          {champion.run_id === null ? (
            <span className="text-muted">Registered without a run</span>
          ) : (
            <span className="flex min-w-0 items-center gap-2">
              <span className="truncate font-mono">{champion.run_id}</span>
              <CopyButton value={champion.run_id} label="MLflow run ID" />
            </span>
          )}
        </Field>
        <Field label="Run contract">
          <span className="font-mono">{champion.contract_version ?? "not logged"}</span>
        </Field>
        <Field label="API contract">
          <span className="font-mono">{apiContract ?? "unknown"}</span>
        </Field>
        <Field label="Compatibility">
          <ContractCompatibility
            runContract={champion.contract_version}
            apiContract={apiContract}
          />
        </Field>
      </dl>
    </section>
  );
}
