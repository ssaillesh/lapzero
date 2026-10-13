/**
 * Tracks the dashboard never shows, although the API still serves them.
 *
 * - spielberg_wide: human laps recorded on a widened copy of the Spielberg map. They stay in the
 *   data (the BC policy behind v5 and v8 was trained on them), but their times are not comparable
 *   with laps on the real track.
 * - oval: a short test loop from the first recording session; every lap on it crashed, and its
 *   map is not among the tracks, so it has neither a ranking nor a drawable track.
 */

export const HIDDEN_TRACK_IDS: ReadonlySet<string> = new Set(["spielberg_wide", "oval"]);

/** True if the dashboard must not show this track. */
export function isHiddenTrack(trackId: string): boolean {
  return HIDDEN_TRACK_IDS.has(trackId);
}

/** The tracks the dashboard lists: every track that is not hidden, in the API's order. */
export function visibleTracks<T extends { track_id: string }>(tracks: readonly T[]): T[] {
  return tracks.filter((track) => !isHiddenTrack(track.track_id));
}
