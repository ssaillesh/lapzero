import { useEffect } from "react";

import { isHttpStatus } from "../api/errors";
import type {
  ChampionResponse,
  LeaderboardEntry,
  Replay,
  TrackDetail,
  TrackSummary,
} from "../api/types";
import { LEADERBOARD_MAX_LIMIT } from "../api/types";
import { PageHeader, SelectField } from "../components/common/Controls";
import { Link } from "../components/common/Link";
import {
  EmptyState,
  ErrorState,
  Notice,
  Skeleton,
  SkeletonRows,
} from "../components/common/States";
import { ReplayCanvas } from "../components/replay/ReplayCanvas";
import { MapFrame, ReplayWorkspace } from "../components/replay/ReplayWorkspace";
import { useChampion, useLeaderboard, useReplay, useTrack, useTrackList } from "../hooks/useApi";
import type { Resource } from "../hooks/useResource";
import { isChampion } from "../lib/champion";
import { formatDateTime, formatLapTime, formatLength } from "../lib/formatters";
import { hrefFor, navigate } from "../lib/router";
import { isHiddenTrack } from "../lib/tracks";

/** /tracks: open the first track, or explain why there is none. */
export function TrackIndexView() {
  const tracks = useTrackList();
  const first = tracks.status === "success" ? tracks.data[0]?.track_id : undefined;
  useEffect(() => {
    if (first !== undefined)
      navigate(hrefFor({ name: "track", trackId: first, lapId: null }), { replace: true });
  }, [first]);

  return (
    <div className="grid gap-6">
      <PageHeader title="Track Explorer">
        Replay any ranked lap on its track, with synchronized telemetry.
      </PageHeader>
      {tracks.status === "error" ? (
        <ErrorState error={tracks.error} onRetry={tracks.reload} />
      ) : tracks.status === "success" && tracks.data.length === 0 ? (
        <EmptyState title="No tracks yet">
          Tracks appear once a lap is uploaded to{" "}
          <code className="font-mono text-ink">POST /api/laps</code>.
        </EmptyState>
      ) : (
        <Skeleton className="h-[420px]" />
      )}
    </div>
  );
}

/** /tracks/:trackId and /tracks/:trackId/replays/:lapId. */
export function TrackViewerView({ trackId, lapId }: { trackId: string; lapId: string | null }) {
  // A hidden track is never shown, even from a direct URL: open the first visible track instead.
  const hidden = isHiddenTrack(trackId);
  useEffect(() => {
    if (hidden) navigate(hrefFor({ name: "tracks" }), { replace: true });
  }, [hidden]);

  const tracks = useTrackList();
  const track = useTrack(hidden ? null : trackId);
  const laps = useLeaderboard(hidden ? null : trackId, null, LEADERBOARD_MAX_LIMIT);
  const replay = useReplay(hidden ? null : lapId);
  const champion = useChampion();
  const championData = champion.status === "success" ? champion.data : null;

  // A track without a chosen lap opens its fastest lap, paused at the start.
  const fastestLap = laps.status === "success" ? laps.data.entries[0]?.lap_id : undefined;
  useEffect(() => {
    if (lapId === null && fastestLap !== undefined) {
      navigate(hrefFor({ name: "track", trackId, lapId: fastestLap }), {
        replace: true,
        keepScroll: true,
      });
    }
  }, [trackId, lapId, fastestLap]);

  // A replay from another track belongs under that track's URL.
  const replayTrack = replay.status === "success" ? replay.data.track_id : undefined;
  useEffect(() => {
    if (lapId !== null && replayTrack !== undefined && replayTrack !== trackId) {
      navigate(hrefFor({ name: "track", trackId: replayTrack, lapId }), {
        replace: true,
        keepScroll: true,
      });
    }
  }, [trackId, lapId, replayTrack]);

  if (hidden) return null;

  if (
    track.status === "error" &&
    (isHttpStatus(track.error, 404) || isHttpStatus(track.error, 422))
  ) {
    return (
      <div className="grid gap-6">
        <PageHeader title="Track not found" />
        <EmptyState
          title={`The API has no track "${trackId}"`}
          action={
            <Link href="/tracks" className="press btn-secondary">
              Open the first track
            </Link>
          }
        >
          Tracks exist once a lap has been uploaded on them.
        </EmptyState>
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[16.5rem_minmax(0,1fr)] lg:gap-8">
      <aside className="grid content-start gap-6 lg:sticky lg:top-20 lg:max-h-[calc(100dvh-7rem)] lg:overflow-y-auto lg:pr-1">
        <TrackList tracks={tracks} activeId={trackId} />
        <LapList trackId={trackId} laps={laps} activeLapId={lapId} champion={championData} />
      </aside>

      <div className="grid min-w-0 content-start gap-5">
        <TrackHeader track={track} />
        {track.status === "error" ? (
          <ErrorState error={track.error} onRetry={track.reload} />
        ) : (
          <ReplayArea
            trackId={trackId}
            lapId={lapId}
            track={track.status === "success" ? track.data : null}
            replay={replay}
            laps={laps}
            champion={championData}
          />
        )}
      </div>
    </div>
  );
}

function TrackHeader({ track }: { track: Resource<TrackDetail> }) {
  if (track.status !== "success") {
    return (
      <div className="grid gap-2">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-4 w-80" />
      </div>
    );
  }
  const { name, track_id: id, length_m: length, created_at: created } = track.data;
  return (
    <PageHeader title={name}>
      <span className="font-mono text-[13px]">
        {name !== id && <>{id}, </>}
        {formatLength(length)} centreline, first lap uploaded {formatDateTime(created)}
      </span>
    </PageHeader>
  );
}

interface ReplayAreaProps {
  trackId: string;
  lapId: string | null;
  track: TrackDetail | null;
  replay: Resource<Replay>;
  laps: Resource<{ entries: LeaderboardEntry[] }>;
  champion: ChampionResponse | null;
}

function ReplayArea({ trackId, lapId, track, replay, laps, champion }: ReplayAreaProps) {
  const geometry = track?.geometry ?? null;
  const trackOnly = (
    <MapFrame>
      {track === null ? (
        <Skeleton className="absolute inset-0 rounded-none" />
      ) : geometry === null ? (
        <div className="absolute inset-0 grid place-items-center p-6 text-center text-muted">
          No centreline on the server for this track.
        </div>
      ) : (
        <ReplayCanvas
          geometry={geometry}
          replay={null}
          clock={null}
          description={`Map of ${track.name}`}
        />
      )}
    </MapFrame>
  );

  if (lapId === null) {
    if (laps.status === "success" && laps.data.entries.length === 0) {
      return (
        <>
          {trackOnly}
          <EmptyState title="No completed laps on this track">
            Only completed laps are listed. A lap that did not finish still opens by its URL,{" "}
            <code className="font-mono text-ink">/tracks/{trackId}/replays/&lt;lap_id&gt;</code>.
          </EmptyState>
        </>
      );
    }
    if (laps.status === "error") {
      return (
        <>
          {trackOnly}
          <ErrorState error={laps.error} onRetry={laps.reload} title="Could not list laps" />
        </>
      );
    }
    return trackOnly;
  }

  if (replay.status === "error") {
    const missing = isHttpStatus(replay.error, 404) || isHttpStatus(replay.error, 422);
    return (
      <>
        {trackOnly}
        <ErrorState
          error={replay.error}
          {...(missing ? { title: "Lap not found" } : { onRetry: replay.reload })}
        >
          {missing ? (
            <>
              No uploaded lap has the ID <code className="font-mono text-ink">{lapId}</code>. Pick a
              lap from the list.
            </>
          ) : undefined}
        </ErrorState>
      </>
    );
  }

  if (replay.status !== "success" || track === null || replay.data.track_id !== trackId) {
    return (
      <>
        <LapSummarySkeleton />
        {trackOnly}
        <Skeleton className="h-9" />
        <Skeleton className="h-16" />
      </>
    );
  }

  const entry =
    laps.status === "success"
      ? (laps.data.entries.find((candidate) => candidate.lap_id === lapId) ?? null)
      : null;
  return (
    <>
      <LapSummary
        replay={replay.data}
        entry={entry}
        champion={entry !== null && isChampion(entry, champion)}
      />
      <ReplayWorkspace
        key={replay.data.lap_id}
        replay={replay.data}
        geometry={geometry}
        trackName={track.name}
      />
    </>
  );
}

function LapSummarySkeleton() {
  return <Skeleton className="h-[58px]" />;
}

function LapSummary({
  replay,
  entry,
  champion,
}: {
  replay: Replay;
  entry: LeaderboardEntry | null;
  champion: boolean;
}) {
  const finished = replay.lap_time_s !== null;
  return (
    <dl className="arrive grid grid-cols-2 gap-x-6 gap-y-3 border-y border-line py-3 sm:flex sm:flex-wrap sm:items-end">
      <div className="min-w-0">
        <dt className="label">Driver</dt>
        <dd className="mt-0.5 flex items-center gap-2 text-[15px] font-medium text-ink">
          <span className="truncate">{replay.driver_label}</span>
          {champion && (
            <span className="rounded-[2px] bg-accent px-1.5 py-px font-mono text-[10px] font-semibold uppercase tracking-[0.06em] text-accent-ink">
              Champion
            </span>
          )}
        </dd>
      </div>
      <div>
        <dt className="label">Lap time</dt>
        <dd className="mt-0.5 font-mono text-[15px] text-ink">
          {finished ? formatLapTime(replay.lap_time_s ?? 0) : "n/a"}
        </dd>
      </div>
      <div>
        <dt className="label">Result</dt>
        <dd className={`mt-0.5 text-[13px] font-medium ${finished ? "text-ok" : "text-warn"}`}>
          {finished ? "Completed" : "Did not finish"}
        </dd>
      </div>
      {entry !== null && (
        <div>
          <dt className="label">Position</dt>
          <dd className="mt-0.5 font-mono text-[15px] text-ink">P{entry.rank}</dd>
        </div>
      )}
      <div className="col-span-2 min-w-0 sm:ml-auto sm:text-right">
        <dt className="label">Lap ID</dt>
        <dd className="mt-0.5 truncate font-mono text-xs text-muted">{replay.lap_id}</dd>
      </div>
    </dl>
  );
}

function TrackList({ tracks, activeId }: { tracks: Resource<TrackSummary[]>; activeId: string }) {
  return (
    <nav aria-label="Tracks" className="grid gap-2">
      <h2 className="label hidden lg:block">Tracks</h2>
      {tracks.status === "error" ? (
        <ErrorState error={tracks.error} onRetry={tracks.reload} />
      ) : tracks.status !== "success" ? (
        <SkeletonRows rows={3} />
      ) : (
        <>
          <div className="lg:hidden">
            <SelectField
              label="Track"
              value={activeId}
              onChange={(trackId) => {
                navigate(hrefFor({ name: "track", trackId, lapId: null }));
              }}
            >
              {!tracks.data.some((t) => t.track_id === activeId) && (
                <option value={activeId}>{activeId}</option>
              )}
              {tracks.data.map((t) => (
                <option key={t.track_id} value={t.track_id}>
                  {t.name}
                </option>
              ))}
            </SelectField>
          </div>
          <ul className="hidden gap-px lg:grid">
            {tracks.data.map((t) => {
              const active = t.track_id === activeId;
              return (
                <li key={t.track_id}>
                  <Link
                    href={hrefFor({ name: "track", trackId: t.track_id, lapId: null })}
                    aria-current={active ? "page" : undefined}
                    className={`relative flex items-baseline justify-between gap-3 rounded-[var(--radius-ctl)] px-3 py-2 transition-colors duration-150 ${
                      active ? "bg-raised text-ink" : "text-muted hover:bg-panel hover:text-ink"
                    }`}
                  >
                    {active && (
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-1.5 left-0 w-0.5 bg-accent"
                      />
                    )}
                    <span className="truncate font-medium">{t.name}</span>
                    <span className="shrink-0 font-mono text-[11px] text-faint">
                      {formatLength(t.length_m)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </nav>
  );
}

interface LapListProps {
  trackId: string;
  laps: Resource<{ entries: LeaderboardEntry[] }>;
  activeLapId: string | null;
  champion: ChampionResponse | null;
}

function LapList({ trackId, laps, activeLapId, champion }: LapListProps) {
  return (
    <nav aria-label="Laps" className="grid gap-2">
      <div className="hidden items-baseline justify-between lg:flex">
        <h2 className="label">Ranked laps</h2>
        {laps.status === "success" && (
          <span className="font-mono text-[11px] text-faint">{laps.data.entries.length}</span>
        )}
      </div>
      {laps.status === "error" ? (
        <Notice tone="warn">The lap list could not be loaded.</Notice>
      ) : laps.status !== "success" ? (
        <SkeletonRows rows={5} />
      ) : laps.data.entries.length === 0 ? (
        <p className="text-[13px] text-muted">No completed laps yet.</p>
      ) : (
        <>
          <div className="lg:hidden">
            <SelectField
              label="Lap"
              value={activeLapId ?? ""}
              onChange={(lapId) => {
                navigate(hrefFor({ name: "track", trackId, lapId }), { keepScroll: true });
              }}
            >
              {activeLapId !== null && !laps.data.entries.some((e) => e.lap_id === activeLapId) && (
                <option value={activeLapId}>Lap {activeLapId} (opened by link)</option>
              )}
              {laps.data.entries.map((e) => (
                <option key={e.lap_id} value={e.lap_id}>
                  P{e.rank} {e.driver_label}, {formatLapTime(e.lap_time_s)}
                </option>
              ))}
            </SelectField>
          </div>
          <ol className="hidden gap-px lg:grid">
            {laps.data.entries.map((entry) => {
              const active = entry.lap_id === activeLapId;
              const isChamp = isChampion(entry, champion);
              return (
                <li key={entry.lap_id}>
                  <Link
                    href={hrefFor({ name: "track", trackId, lapId: entry.lap_id })}
                    navigation={{ keepScroll: true }}
                    aria-current={active ? "true" : undefined}
                    className={`relative grid grid-cols-[2.25rem_minmax(0,1fr)_auto] items-baseline gap-2 rounded-[var(--radius-ctl)] px-3 py-2 text-[13px] transition-colors duration-150 ${
                      active ? "bg-raised text-ink" : "text-muted hover:bg-panel hover:text-ink"
                    }`}
                  >
                    {active && (
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-1.5 left-0 w-0.5 bg-accent"
                      />
                    )}
                    <span className="font-mono text-[11px] text-faint">P{entry.rank}</span>
                    <span className="flex min-w-0 items-center gap-1.5">
                      <span className="truncate">{entry.driver_label}</span>
                      {isChamp && (
                        <>
                          <span
                            aria-hidden="true"
                            className="h-1.5 w-1.5 shrink-0 rounded-[1px] bg-accent"
                            title="Champion"
                          />
                          <span className="sr-only">, champion</span>
                        </>
                      )}
                    </span>
                    <span className="font-mono text-[12px] text-ink">
                      {formatLapTime(entry.lap_time_s)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ol>
          {activeLapId !== null && !laps.data.entries.some((e) => e.lap_id === activeLapId) && (
            <p className="hidden text-xs text-faint lg:block">
              The open lap is not ranked (it did not finish, or is not this driver&apos;s best).
            </p>
          )}
        </>
      )}
    </nav>
  );
}
