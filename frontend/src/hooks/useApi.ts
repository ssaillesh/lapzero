/**
 * One hook per API resource, each backed by a cache shared across views. Track geometry and
 * replays are immutable once uploaded, so they are fetched once per session (contract Section 6:
 * geometry is downloaded once). Leaderboards and the champion change; their views offer reload.
 */

import { api } from "../api/client";
import type {
  ChampionResponse,
  DriverType,
  LeaderboardResponse,
  Replay,
  TrackDetail,
  TrackSummary,
} from "../api/types";
import { visibleTracks } from "../lib/tracks";
import { createResourceCache, useResource } from "./useResource";
import type { Resource } from "./useResource";

const trackListCache = createResourceCache<TrackSummary[]>();
const trackCache = createResourceCache<TrackDetail>();
const leaderboardCache = createResourceCache<LeaderboardResponse>();
const replayCache = createResourceCache<Replay>();
const championCache = createResourceCache<ChampionResponse>();

/** Forget every cached response, so the next render of each view fetches again. */
export function clearApiCaches(): void {
  for (const cache of [trackListCache, trackCache, leaderboardCache, replayCache, championCache])
    cache.clear();
}

/** Every track the dashboard shows: the API's list without the hidden tracks (lib/tracks). */
export function useTrackList(): Resource<TrackSummary[]> {
  return useResource(trackListCache, "tracks", (signal) =>
    api.listTracks({ signal }).then((tracks) => visibleTracks(tracks)),
  );
}

export function useTrack(trackId: string | null): Resource<TrackDetail> {
  return useResource(trackCache, trackId, (signal) => api.getTrack(trackId ?? "", { signal }));
}

export function useLeaderboard(
  trackId: string | null,
  driverType: DriverType | null,
  limit: number,
): Resource<LeaderboardResponse> {
  const key = trackId === null ? null : `${trackId}|${driverType ?? "all"}|${limit}`;
  return useResource(leaderboardCache, key, (signal) =>
    api.getLeaderboard(trackId ?? "", { driverType, limit }, { signal }),
  );
}

export function useReplay(lapId: string | null): Resource<Replay> {
  return useResource(replayCache, lapId, (signal) => api.getReplay(lapId ?? "", { signal }));
}

export function useChampion(): Resource<ChampionResponse> {
  return useResource(championCache, "champion", (signal) => api.getChampion({ signal }));
}
