/** Client-side ordering and filtering of the leaderboard entries the API returned. */

import type { LeaderboardEntry } from "../api/types";

export type SortKey = "rank" | "driver" | "lap_time" | "recorded";
export type SortDirection = "asc" | "desc";

export interface SortState {
  key: SortKey;
  direction: SortDirection;
}

export const DEFAULT_SORT: SortState = { key: "rank", direction: "asc" };

/** Default direction when a column is first chosen: fastest, A to Z, newest. */
export function initialDirection(key: SortKey): SortDirection {
  return key === "recorded" ? "desc" : "asc";
}

/** Clicking a column: flip the direction if it is already the sort key, else start it fresh. */
export function nextSort(current: SortState, key: SortKey): SortState {
  if (current.key === key) return { key, direction: current.direction === "asc" ? "desc" : "asc" };
  return { key, direction: initialDirection(key) };
}

function compare(a: LeaderboardEntry, b: LeaderboardEntry, key: SortKey): number {
  switch (key) {
    case "rank":
      return a.rank - b.rank;
    case "driver":
      return a.driver_label.localeCompare(b.driver_label, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    case "lap_time":
      return a.lap_time_s - b.lap_time_s;
    case "recorded":
      return Date.parse(a.created_at) - Date.parse(b.created_at);
  }
}

/** A sorted copy; ties fall back to rank so the order is stable. */
export function sortEntries(
  entries: readonly LeaderboardEntry[],
  sort: SortState,
): LeaderboardEntry[] {
  const sign = sort.direction === "asc" ? 1 : -1;
  return [...entries].sort((a, b) => sign * compare(a, b, sort.key) || a.rank - b.rank);
}

/** Entries whose driver label, name, model version or lap ID contains `query` (any case). */
export function filterEntries(
  entries: readonly LeaderboardEntry[],
  query: string,
): LeaderboardEntry[] {
  const needle = query.trim().toLowerCase();
  if (needle === "") return [...entries];
  return entries.filter((entry) =>
    [entry.driver_label, entry.driver_name, entry.model_version ?? "", entry.lap_id].some((field) =>
      field.toLowerCase().includes(needle),
    ),
  );
}

/** The fastest lap time among the entries, s, or null for an empty list. */
export function fastestTime(entries: readonly LeaderboardEntry[]): number | null {
  let best: number | null = null;
  for (const entry of entries)
    if (best === null || entry.lap_time_s < best) best = entry.lap_time_s;
  return best;
}
