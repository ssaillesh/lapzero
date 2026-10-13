import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { isHttpStatus } from "../api/errors";
import type { ChampionResponse, DriverType, LeaderboardResponse, TrackSummary } from "../api/types";
import {
  PageHeader,
  SearchField,
  SegmentedControl,
  SelectField,
} from "../components/common/Controls";
import { Link } from "../components/common/Link";
import { EmptyState, ErrorState, Skeleton, SkeletonRows } from "../components/common/States";
import { LeaderboardTable } from "../components/leaderboard/LeaderboardTable";
import { useChampion, useLeaderboard, useTrackList } from "../hooks/useApi";
import type { Resource } from "../hooks/useResource";
import { isChampion } from "../lib/champion";
import { formatGap, formatLapTime, formatLength } from "../lib/formatters";
import {
  DEFAULT_SORT,
  fastestTime,
  filterEntries,
  nextSort,
  sortEntries,
} from "../lib/leaderboard";
import type { SortState } from "../lib/leaderboard";
import type { LeaderboardParams } from "../lib/router";
import { LEADERBOARD_LIMITS, hrefFor, navigate } from "../lib/router";

type DriverFilter = "all" | DriverType;

const DRIVER_OPTIONS: readonly { value: DriverFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "human", label: "Human" },
  { value: "model", label: "Model" },
];

export function LeaderboardView({ params }: { params: LeaderboardParams }) {
  const tracks = useTrackList();
  const champion = useChampion();

  const go = (next: Partial<LeaderboardParams>, replace = false) => {
    navigate(hrefFor({ name: "leaderboard", params: { ...params, ...next } }), {
      replace,
      keepScroll: true,
    });
  };

  // No track in the URL: show the first track, and say so in the URL so a refresh keeps it.
  const firstTrack = tracks.status === "success" ? tracks.data[0]?.track_id : undefined;
  useEffect(() => {
    if (params.trackId === null && firstTrack !== undefined) {
      navigate(hrefFor({ name: "leaderboard", params: { ...params, trackId: firstTrack } }), {
        replace: true,
        keepScroll: true,
      });
    }
  }, [params, firstTrack]);

  return (
    <div className="grid gap-6">
      <PageHeader title="Leaderboard">
        Each driver&apos;s fastest completed lap on a track. Every model version ranks as its own
        driver; crashed, timed-out and stuck laps never rank.
      </PageHeader>

      {tracks.status === "loading" || tracks.status === "idle" ? (
        <div className="grid gap-4">
          <Skeleton className="h-16" />
          <SkeletonRows rows={6} />
        </div>
      ) : tracks.status === "error" ? (
        <ErrorState error={tracks.error} onRetry={tracks.reload} />
      ) : tracks.data.length === 0 ? (
        <EmptyState title="No tracks yet">
          Tracks appear once a lap is uploaded to{" "}
          <code className="font-mono text-ink">POST /api/laps</code>. Record one with{" "}
          <code className="font-mono text-ink">scripts/drive.py</code> and upload its telemetry
          JSON.
        </EmptyState>
      ) : (
        <TrackBoard tracks={tracks.data} params={params} champion={champion} onChange={go} />
      )}
    </div>
  );
}

interface TrackBoardProps {
  tracks: TrackSummary[];
  params: LeaderboardParams;
  champion: Resource<ChampionResponse>;
  onChange: (next: Partial<LeaderboardParams>) => void;
}

function TrackBoard({ tracks, params, champion, onChange }: TrackBoardProps) {
  const track = tracks.find((candidate) => candidate.track_id === params.trackId) ?? null;
  const board = useLeaderboard(track?.track_id ?? null, params.driverType, params.limit);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [query, setQuery] = useState("");
  const championData = champion.status === "success" ? champion.data : null;

  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-[minmax(12rem,16rem)_auto_8rem_minmax(12rem,1fr)_auto] lg:items-end">
        <SelectField
          label="Track"
          value={track?.track_id ?? ""}
          onChange={(trackId) => {
            onChange({ trackId });
          }}
        >
          {track === null && <option value="">Choose a track</option>}
          {tracks.map((option) => (
            <option key={option.track_id} value={option.track_id}>
              {option.name}
            </option>
          ))}
        </SelectField>
        <SegmentedControl
          label="Drivers"
          value={params.driverType ?? "all"}
          options={DRIVER_OPTIONS}
          onChange={(value) => {
            onChange({ driverType: value === "all" ? null : value });
          }}
        />
        <SelectField
          label="Show"
          value={String(params.limit)}
          onChange={(value) => {
            onChange({ limit: Number(value) });
          }}
        >
          {LEADERBOARD_LIMITS.map((limit) => (
            <option key={limit} value={limit}>
              Top {limit}
            </option>
          ))}
        </SelectField>
        <SearchField
          label="Filter shown rows"
          value={query}
          onChange={setQuery}
          placeholder="Driver, version or lap ID"
        />
        <button
          type="button"
          onClick={() => {
            board.reload();
            champion.reload();
          }}
          disabled={board.status === "loading"}
          className="press btn-secondary justify-self-start"
        >
          Refresh
        </button>
      </div>

      {track === null ? (
        <EmptyState title={`No track "${params.trackId ?? ""}"`}>
          The API has no laps on this track. Choose one of the {tracks.length} tracks above.
        </EmptyState>
      ) : board.status === "loading" || board.status === "idle" ? (
        <div className="grid gap-4">
          <Skeleton className="h-20" />
          <SkeletonRows rows={Math.min(params.limit, 8)} />
        </div>
      ) : board.status === "error" ? (
        <ErrorState
          error={board.error}
          onRetry={board.reload}
          {...(isHttpStatus(board.error, 404) ? { title: "Track not found" } : {})}
        />
      ) : (
        <BoardBody
          track={track}
          board={board.data}
          champion={champion}
          championData={championData}
          sort={sort}
          onSort={(key) => {
            setSort((current) => nextSort(current, key));
          }}
          query={query}
          onClearQuery={() => {
            setQuery("");
          }}
          limit={params.limit}
        />
      )}
    </>
  );
}

interface BoardBodyProps {
  track: TrackSummary;
  board: LeaderboardResponse;
  champion: Resource<ChampionResponse>;
  championData: ChampionResponse | null;
  sort: SortState;
  onSort: (key: SortState["key"]) => void;
  query: string;
  onClearQuery: () => void;
  limit: number;
}

function BoardBody({
  track,
  board,
  champion,
  championData,
  sort,
  onSort,
  query,
  onClearQuery,
  limit,
}: BoardBodyProps) {
  const fastest = fastestTime(board.entries);
  const rows = useMemo(
    () => sortEntries(filterEntries(board.entries, query), sort),
    [board.entries, query, sort],
  );
  const leader = board.entries.find((entry) => entry.rank === 1) ?? null;
  const championEntry = board.entries.find((entry) => isChampion(entry, championData)) ?? null;
  const scope =
    board.driver_type === null
      ? "drivers"
      : board.driver_type === "human"
        ? "human drivers"
        : "model versions";

  if (board.entries.length === 0) {
    return (
      <EmptyState
        title={`No completed laps on ${track.name}${board.driver_type === null ? "" : ` by ${scope}`}`}
      >
        Only completed laps rank. Laps that crashed, timed out or got stuck are stored but never
        appear here.
      </EmptyState>
    );
  }

  return (
    <div className="arrive grid gap-4">
      <dl className="grid grid-cols-1 gap-px overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-line sm:grid-cols-3">
        <Stat label="Fastest lap">
          <span className="font-mono text-[22px] font-medium tracking-[-0.01em] text-ink">
            {leader === null ? "n/a" : formatLapTime(leader.lap_time_s)}
          </span>
          <span className="truncate text-muted">{leader?.driver_label ?? ""}</span>
        </Stat>
        <Stat label="Ranked">
          <span className="font-mono text-[22px] font-medium tracking-[-0.01em] text-ink">
            {board.entries.length}
          </span>
          <span className="text-muted">
            {scope}
            {board.entries.length >= limit ? `, top ${limit} shown` : ""} on{" "}
            {formatLength(track.length_m)}
          </span>
        </Stat>
        <Stat label="Champion">
          <ChampionSummary
            champion={champion}
            championData={championData}
            entry={championEntry}
            fastest={fastest}
            limit={limit}
            humansOnly={board.driver_type === "human"}
          />
        </Stat>
      </dl>

      {rows.length === 0 ? (
        <EmptyState
          title={`Nothing matches "${query.trim()}"`}
          action={
            <button type="button" className="press btn-secondary" onClick={onClearQuery}>
              Clear filter
            </button>
          }
        >
          The filter searches the {board.entries.length} rows the API returned, by driver, model
          version and lap ID.
        </EmptyState>
      ) : (
        <LeaderboardTable
          trackId={track.track_id}
          entries={rows}
          fastest={fastest}
          champion={championData}
          sort={sort}
          onSort={onSort}
        />
      )}
      <p className="text-xs text-faint">
        Sorting and filtering apply to the {board.entries.length} rows returned by the API. Lap
        times read minutes:seconds; gaps are seconds behind the fastest lap in this list.
      </p>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 bg-panel px-4 py-3">
      <dt className="label">{label}</dt>
      <dd className="flex min-w-0 items-baseline gap-2.5">{children}</dd>
    </div>
  );
}

function ChampionSummary({
  champion,
  championData,
  entry,
  fastest,
  limit,
  humansOnly,
}: {
  champion: Resource<ChampionResponse>;
  championData: ChampionResponse | null;
  entry: { rank: number; lap_time_s: number } | null;
  fastest: number | null;
  limit: number;
  humansOnly: boolean;
}) {
  if (champion.status === "loading" || champion.status === "idle")
    return <Skeleton className="h-7 w-32" />;
  if (champion.status === "error") {
    const none = isHttpStatus(champion.error, 404);
    return (
      <span className="text-muted">
        {none ? "No champion registered" : "Registry unavailable"}{" "}
        <Link
          href="/governance"
          className="text-ink underline decoration-line-strong underline-offset-4 hover:decoration-ink"
        >
          Details
        </Link>
      </span>
    );
  }
  if (championData === null) return null;
  if (entry === null && humansOnly) {
    return (
      <span className="text-muted">
        <span className="font-mono text-ink">v{championData.version}</span> hidden by the Human
        filter
      </span>
    );
  }
  if (entry === null) {
    return (
      <span className="text-muted">
        <span className="font-mono text-ink">v{championData.version}</span> not in this top {limit}
      </span>
    );
  }
  return (
    <>
      <span className="font-mono text-[22px] font-medium tracking-[-0.01em] text-accent">
        P{entry.rank}
      </span>
      <span className="truncate text-muted">
        v{championData.version}
        {fastest !== null && entry.lap_time_s > fastest
          ? `, ${formatGap(entry.lap_time_s - fastest)} s`
          : ", fastest"}
      </span>
    </>
  );
}
