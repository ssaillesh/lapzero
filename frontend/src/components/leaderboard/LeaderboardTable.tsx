import type { MouseEvent } from "react";

import type { ChampionResponse, LeaderboardEntry } from "../../api/types";
import { isChampion } from "../../lib/champion";
import { formatDateTime, formatGap, formatLapTime } from "../../lib/formatters";
import type { SortKey, SortState } from "../../lib/leaderboard";
import { hrefFor, navigate } from "../../lib/router";
import { Link } from "../common/Link";

interface LeaderboardTableProps {
  trackId: string;
  entries: readonly LeaderboardEntry[];
  /** Reference for the gap column: the fastest time in the response, s. */
  fastest: number | null;
  champion: ChampionResponse | null;
  sort: SortState;
  onSort: (key: SortKey) => void;
}

const COLUMNS: readonly { key: SortKey | null; label: string; className: string }[] = [
  { key: "rank", label: "Pos", className: "w-14 pl-4 text-left" },
  { key: "driver", label: "Driver", className: "text-left" },
  { key: null, label: "Type", className: "hidden w-28 text-left md:table-cell" },
  { key: "lap_time", label: "Lap time", className: "w-28 text-right" },
  { key: null, label: "Gap", className: "hidden w-24 text-right sm:table-cell" },
  { key: "recorded", label: "Recorded", className: "hidden w-48 text-right lg:table-cell" },
  { key: null, label: "Replay", className: "w-12 pr-4 text-right sm:w-24" },
];

function SortButton({
  label,
  active,
  direction,
  onClick,
}: {
  label: string;
  active: boolean;
  direction: "asc" | "desc";
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`label inline-flex items-center gap-1 rounded-[2px] transition-colors hover:text-ink ${active ? "text-ink" : ""}`}
    >
      {label}
      <span
        aria-hidden="true"
        className={`w-2 text-center ${active ? "opacity-100" : "opacity-0"}`}
      >
        {direction === "asc" ? "↑" : "↓"}
      </span>
    </button>
  );
}

/**
 * The ranked laps. Each row opens its replay; the "Replay" link is the keyboard target and the
 * whole row is a larger pointer target for the same destination.
 */
export function LeaderboardTable({
  trackId,
  entries,
  fastest,
  champion,
  sort,
  onSort,
}: LeaderboardTableProps) {
  return (
    <div className="overflow-hidden rounded-[var(--radius-ctl)] border border-line">
      <table className="w-full border-collapse text-[13px]">
        <caption className="sr-only">
          Fastest completed lap of each driver on {trackId}. Select a row to open its replay.
        </caption>
        <thead className="bg-panel">
          <tr className="h-10 border-b border-line">
            {COLUMNS.map((column) => {
              const active = column.key !== null && sort.key === column.key;
              return (
                <th
                  key={column.label}
                  scope="col"
                  aria-sort={
                    active ? (sort.direction === "asc" ? "ascending" : "descending") : undefined
                  }
                  className={`px-3 font-normal ${column.className}`}
                >
                  {column.key === null ? (
                    <span
                      className={`label ${column.label === "Replay" ? "sr-only sm:not-sr-only" : ""}`}
                    >
                      {column.label}
                    </span>
                  ) : (
                    <SortButton
                      label={column.label}
                      active={active}
                      direction={active ? sort.direction : "asc"}
                      onClick={() => {
                        if (column.key !== null) onSort(column.key);
                      }}
                    />
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <Row
              key={entry.lap_id}
              trackId={trackId}
              entry={entry}
              fastest={fastest}
              champion={isChampion(entry, champion)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Row({
  trackId,
  entry,
  fastest,
  champion,
}: {
  trackId: string;
  entry: LeaderboardEntry;
  fastest: number | null;
  champion: boolean;
}) {
  const href = hrefFor({ name: "track", trackId, lapId: entry.lap_id });
  const gap = fastest === null ? null : entry.lap_time_s - fastest;
  const openFromRow = (event: MouseEvent<HTMLTableRowElement>) => {
    if (event.target instanceof Element && event.target.closest("a, button")) return;
    if (event.metaKey || event.ctrlKey) {
      window.open(href, "_blank", "noopener");
      return;
    }
    navigate(href);
  };
  return (
    <tr
      onClick={openFromRow}
      data-champion={champion || undefined}
      className={`group h-12 cursor-pointer border-b border-line transition-colors duration-150 last:border-b-0 hover:bg-panel ${
        champion ? "bg-accent-soft hover:bg-accent-soft" : ""
      }`}
    >
      <td className="relative pl-4 pr-3 font-mono text-muted">
        {champion && (
          <span aria-hidden="true" className="absolute inset-y-0 left-0 w-0.5 bg-accent" />
        )}
        P{entry.rank}
      </td>
      <td className="max-w-0 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="min-w-[2.5rem] truncate font-medium text-ink">{entry.driver_label}</span>
          {champion && (
            <span className="shrink-0 rounded-[2px] bg-accent px-1.5 py-px font-mono text-[10px] font-semibold uppercase tracking-[0.06em] text-accent-ink">
              Champion
            </span>
          )}
        </div>
        <div className="truncate text-xs text-faint">
          {entry.driver_label !== entry.driver_name ? entry.driver_name : null}
          <span className="md:hidden">
            {entry.driver_label !== entry.driver_name ? " / " : ""}
            {entry.driver_type === "human" ? "Human" : "Model"}
          </span>
        </div>
      </td>
      <td className="hidden px-3 text-muted md:table-cell">
        {entry.driver_type === "human"
          ? "Human"
          : `Model${entry.model_version !== null ? ` v${entry.model_version}` : ""}`}
      </td>
      <td className="px-3 text-right font-mono text-[14px] font-medium text-ink">
        {formatLapTime(entry.lap_time_s)}
      </td>
      <td className="hidden px-3 text-right font-mono text-muted sm:table-cell">
        {gap === null || gap === 0 ? <span className="text-faint">Leader</span> : formatGap(gap)}
      </td>
      <td className="hidden px-3 text-right text-muted lg:table-cell">
        <time dateTime={entry.created_at}>{formatDateTime(entry.created_at)}</time>
      </td>
      <td className="pl-3 pr-4 text-right">
        <Link
          href={href}
          className="inline-flex items-center gap-1 rounded-[2px] text-muted transition-colors group-hover:text-ink"
          aria-label={`Replay ${entry.driver_label}, lap time ${formatLapTime(entry.lap_time_s)}`}
        >
          <span className="hidden sm:inline">Replay</span> <span aria-hidden="true">→</span>
        </Link>
      </td>
    </tr>
  );
}
