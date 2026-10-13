import type { ChampionResponse, TrackSummary } from "../../api/types";
import { LEADERBOARD_MAX_LIMIT } from "../../api/types";
import { useLeaderboard } from "../../hooks/useApi";
import { isChampion } from "../../lib/champion";
import { formatGap, formatLapTime } from "../../lib/formatters";
import { fastestTime } from "../../lib/leaderboard";
import { hrefFor } from "../../lib/router";
import { Link } from "../common/Link";
import { Skeleton } from "../common/States";

/** The champion's best ranked lap on every track, against each track's fastest lap. */
export function TrackRecord({
  tracks,
  champion,
}: {
  tracks: TrackSummary[];
  champion: ChampionResponse;
}) {
  return (
    <section aria-labelledby="record-heading" className="grid gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 id="record-heading" className="text-[15px] font-semibold text-ink">
          Champion on the leaderboards
        </h2>
        <p className="text-xs text-faint">
          Uploaded laps by model version {champion.version}, among each track&apos;s top{" "}
          {LEADERBOARD_MAX_LIMIT} drivers.
        </p>
      </div>
      <div className="overflow-hidden rounded-[var(--radius-ctl)] border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-panel">
            <tr className="h-10 border-b border-line">
              <th scope="col" className="label px-4 text-left font-normal">
                Track
              </th>
              <th scope="col" className="label w-20 px-3 text-left font-normal">
                Pos
              </th>
              <th scope="col" className="label w-28 px-3 text-right font-normal">
                Lap time
              </th>
              <th
                scope="col"
                className="label hidden w-28 px-3 text-right font-normal sm:table-cell"
              >
                Gap to P1
              </th>
              <th scope="col" className="label w-24 px-4 text-right font-normal">
                Replay
              </th>
            </tr>
          </thead>
          <tbody>
            {tracks.map((track) => (
              <TrackRow key={track.track_id} track={track} champion={champion} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TrackRow({ track, champion }: { track: TrackSummary; champion: ChampionResponse }) {
  const board = useLeaderboard(track.track_id, null, LEADERBOARD_MAX_LIMIT);
  const cells = (() => {
    if (board.status === "loading" || board.status === "idle") {
      return (
        <td colSpan={4} className="px-3">
          <Skeleton className="h-4 w-full" />
        </td>
      );
    }
    if (board.status === "error") {
      return (
        <td colSpan={4} className="px-3 text-warn">
          Leaderboard unavailable.{" "}
          <button type="button" onClick={board.reload} className="underline underline-offset-4">
            Retry
          </button>
        </td>
      );
    }
    const entry = board.data.entries.find((candidate) => isChampion(candidate, champion));
    if (entry === undefined) {
      return (
        <td colSpan={4} className="px-3 text-muted">
          No completed lap by v{champion.version} uploaded
        </td>
      );
    }
    const fastest = fastestTime(board.data.entries);
    return (
      <>
        <td className="px-3 font-mono text-accent">P{entry.rank}</td>
        <td className="px-3 text-right font-mono text-ink">{formatLapTime(entry.lap_time_s)}</td>
        <td className="hidden px-3 text-right font-mono text-muted sm:table-cell">
          {fastest === null || entry.lap_time_s === fastest
            ? "Leader"
            : formatGap(entry.lap_time_s - fastest)}
        </td>
        <td className="px-4 text-right">
          <Link
            href={hrefFor({ name: "track", trackId: track.track_id, lapId: entry.lap_id })}
            className="text-muted transition-colors hover:text-ink"
          >
            Replay <span aria-hidden="true">→</span>
          </Link>
        </td>
      </>
    );
  })();
  return (
    <tr className="h-11 border-b border-line last:border-b-0">
      <th scope="row" className="px-4 text-left font-medium text-ink">
        {track.name}
      </th>
      {cells}
    </tr>
  );
}
