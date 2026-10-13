import type { ApiError } from "../api/errors";
import { isHttpStatus } from "../api/errors";
import type { ChampionResponse } from "../api/types";
import { GATE_TAGS } from "../api/types";
import { PageHeader } from "../components/common/Controls";
import { EmptyState, ErrorState, Notice, Skeleton } from "../components/common/States";
import { ChampionMetadata } from "../components/governance/ChampionMetadata";
import { GateRecord } from "../components/governance/GateRecord";
import { TrackRecord } from "../components/governance/TrackRecord";
import { useChampion, useTrackList } from "../hooks/useApi";
import { useBackendHealth } from "../hooks/useBackendHealth";

const GATE_TAG_KEYS = new Set<string>(Object.values(GATE_TAGS));

/** Which policy holds the champion alias, the evidence recorded with it, and how it ranks. */
export function ChampionView() {
  const champion = useChampion();
  const { snapshot } = useBackendHealth();

  return (
    <div className="grid gap-8">
      <PageHeader
        title="Model governance"
        actions={
          <button
            type="button"
            className="press btn-secondary"
            onClick={champion.reload}
            disabled={champion.status === "loading"}
          >
            Refresh
          </button>
        }
      >
        The policy the MLflow registry&apos;s champion alias points at, and the evidence recorded
        with it. This page only reads the registry; promotion happens in{" "}
        <code className="font-mono text-ink">scripts/eval_gate.py</code>.
      </PageHeader>

      {champion.status === "loading" || champion.status === "idle" ? (
        <div className="grid gap-4">
          <Skeleton className="h-[260px]" />
          <Skeleton className="h-[180px]" />
        </div>
      ) : champion.status === "error" ? (
        <ChampionUnavailable error={champion.error} onRetry={champion.reload} />
      ) : (
        <ChampionDetails champion={champion.data} apiContract={snapshot.contractVersion} />
      )}

      <section aria-labelledby="unavailable-heading" className="grid gap-2">
        <h2 id="unavailable-heading" className="text-[15px] font-semibold text-ink">
          Not served by the API
        </h2>
        <Notice>
          Training parameters and metrics, friction sweep reports (completion rates and nominal-grip
          lap times), the challenger alias and earlier versions live in MLflow and under{" "}
          <code className="font-mono text-ink">data/eval/friction_sweep/</code>. The API exposes
          only <code className="font-mono text-ink">GET /api/models/champion</code>, so this page
          shows nothing it would have to guess. Look the run up in MLflow by its run ID.
        </Notice>
      </section>
    </div>
  );
}

function ChampionUnavailable({ error, onRetry }: { error: ApiError | Error; onRetry: () => void }) {
  if (isHttpStatus(error, 404)) {
    return (
      <EmptyState title="No champion registered">
        The registered model has no <code className="font-mono text-ink">champion</code> alias yet,
        or is not registered at all. The first version a training run registers becomes champion (
        <code className="font-mono text-ink">scripts/train_bc.py</code>,{" "}
        <code className="font-mono text-ink">scripts/train_ppo.py</code>
        ); later versions replace it only through the release gate.
      </EmptyState>
    );
  }
  if (isHttpStatus(error, 503)) {
    return (
      <ErrorState error={error} onRetry={onRetry} title="Model registry unavailable">
        The API is up but could not read the MLflow registry at its MLFLOW_TRACKING_URI.
      </ErrorState>
    );
  }
  return <ErrorState error={error} onRetry={onRetry} />;
}

function ChampionDetails({
  champion,
  apiContract,
}: {
  champion: ChampionResponse;
  apiContract: string | null;
}) {
  const tracks = useTrackList();
  const otherTags = Object.entries(champion.tags)
    .filter(([key]) => !GATE_TAG_KEYS.has(key))
    .sort(([a], [b]) => a.localeCompare(b));

  return (
    <>
      <ChampionMetadata champion={champion} apiContract={apiContract} />
      <GateRecord champion={champion} />
      {tracks.status === "success" && tracks.data.length > 0 ? (
        <TrackRecord tracks={tracks.data} champion={champion} />
      ) : tracks.status === "error" ? (
        <ErrorState error={tracks.error} onRetry={tracks.reload} title="Could not load tracks" />
      ) : tracks.status === "success" ? null : (
        <Skeleton className="h-40" />
      )}
      <section aria-labelledby="tags-heading" className="grid gap-3">
        <h2 id="tags-heading" className="text-[15px] font-semibold text-ink">
          Other version tags
        </h2>
        {otherTags.length === 0 ? (
          <p className="text-[13px] text-muted">None besides the gate record.</p>
        ) : (
          <dl className="grid gap-px overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-line sm:grid-cols-2 xl:grid-cols-3">
            {otherTags.map(([key, value]) => (
              <div key={key} className="min-w-0 bg-panel px-4 py-2.5">
                <dt className="font-mono text-[11px] text-faint">{key}</dt>
                <dd className="mt-0.5 truncate font-mono text-[13px] text-ink" title={value}>
                  {value}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>
    </>
  );
}
