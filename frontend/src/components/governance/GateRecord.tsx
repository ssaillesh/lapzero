import type { ReactNode } from "react";

import type { ChampionResponse } from "../../api/types";
import { GATE_TAGS } from "../../api/types";

/**
 * The release gate's evidence as recorded on the champion's version tags (lapzero/registry.py).
 * The dashboard only reports what the registry holds; it does not run or re-evaluate the gate.
 */
export function GateRecord({ champion }: { champion: ChampionResponse }) {
  const tags = champion.tags;
  const result = tags[GATE_TAGS.result];
  const promoted = result === "promote";

  return (
    <section aria-labelledby="gate-heading" className="grid gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 id="gate-heading" className="text-[15px] font-semibold text-ink">
          Release gate record
        </h2>
        <p className="font-mono text-xs text-faint">
          From version tags written by scripts/eval_gate.py --promote
        </p>
      </div>

      <div className="grid gap-px overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-line lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="bg-panel p-5">
          <p className="label">Verdict</p>
          {result === undefined ? (
            <>
              <p className="mt-2 text-[15px] font-medium text-ink">
                No gate record on this version
              </p>
              <p className="mt-1.5 max-w-[60ch] text-[13px] text-muted">
                v{champion.version} has no <code className="font-mono">{GATE_TAGS.result}</code>{" "}
                tag, so the registry holds no release-gate evidence for it. The first registered
                version of a model, and versions promoted before the gate existed, hold the alias
                without one.
              </p>
            </>
          ) : (
            <>
              <p className={`mt-2 text-[15px] font-medium ${promoted ? "text-ok" : "text-warn"}`}>
                {promoted ? "Promoted by the release gate" : `Gate result "${result}"`}
              </p>
              <dl className="mt-4 grid gap-2.5 text-[13px]">
                <Evidence label="Evaluated on">{tags[GATE_TAGS.track]}</Evidence>
                <Evidence label="Replaced">
                  {tags[GATE_TAGS.replacedVersion] === undefined
                    ? undefined
                    : `v${tags[GATE_TAGS.replacedVersion]}`}
                </Evidence>
                <Evidence label="Champion sweep" mono>
                  {tags[GATE_TAGS.championSweep]}
                </Evidence>
                <Evidence label="Challenger sweep" mono>
                  {tags[GATE_TAGS.challengerSweep]}
                </Evidence>
              </dl>
            </>
          )}
        </div>
        <div className="bg-panel p-5">
          <p className="label">Promotion rule</p>
          <ul className="mt-2 grid gap-2.5 text-[13px] text-muted">
            <li>
              <span className="text-ink">Robustness.</span> Both policies are swept on the same
              track over tyre friction μ 0.50 to 1.05 in 0.05 steps, 10 jittered-start episodes per
              level. The challenger must complete at least 98 % of all episodes with μ from 0.60 to
              1.05.
            </li>
            <li>
              <span className="text-ink">Pace.</span> Its mean lap time at nominal grip (μ 1.05) may
              be at most 2.0 % slower than the champion&apos;s.
            </li>
            <li>
              <span className="text-ink">Contract.</span> Sweeps with different contract major
              versions are never compared.
            </li>
          </ul>
          <p className="mt-3 font-mono text-[11px] text-faint">Source: lapzero/eval/gate.py</p>
        </div>
      </div>
    </section>
  );
}

function Evidence({
  label,
  mono = false,
  children,
}: {
  label: string;
  mono?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-baseline gap-3">
      <dt className="label">{label}</dt>
      <dd
        className={`min-w-0 truncate ${mono ? "font-mono text-[12px]" : ""} ${children === undefined ? "text-faint" : "text-ink"}`}
      >
        {children ?? "not recorded"}
      </dd>
    </div>
  );
}
