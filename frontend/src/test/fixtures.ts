/**
 * Test fixtures shaped like real API responses (backend/app/schemas, schemas/replay.schema.json).
 * Used only by tests; production code never falls back to sample data.
 */

import type {
  ChampionResponse,
  LeaderboardEntry,
  LeaderboardResponse,
  Replay,
  TrackDetail,
  TrackSummary,
} from "../api/types";

export const TRACKS: TrackSummary[] = [
  {
    track_id: "oschersleben",
    name: "oschersleben",
    length_m: 3678.2,
    created_at: "2026-10-12T09:14:03Z",
  },
  {
    track_id: "spielberg",
    name: "spielberg",
    length_m: 4318.6,
    created_at: "2026-10-11T18:02:41Z",
  },
];

/** A 40 m x 20 m loop with 2 m edges either side, in driving order (counter-clockwise). */
export function squareTrack(trackId = "spielberg"): TrackDetail {
  const centre = { x: [0, 40, 40, 0], y: [0, 0, 20, 20] };
  return {
    ...(TRACKS.find((t) => t.track_id === trackId) ?? {
      track_id: trackId,
      name: trackId,
      length_m: 120,
      created_at: "2026-10-11T18:02:41Z",
    }),
    metadata_json: {},
    geometry: {
      centreline: centre,
      left: { x: [2, 38, 38, 2], y: [2, 2, 18, 18] },
      right: { x: [-2, 42, 42, -2], y: [-2, -2, 22, 22] },
    },
  };
}

export function entry(
  overrides: Partial<LeaderboardEntry> & Pick<LeaderboardEntry, "rank" | "lap_id">,
): LeaderboardEntry {
  return {
    driver_type: "human",
    driver_name: "saillesh",
    driver_label: "saillesh",
    model_version: null,
    lap_time_s: 46,
    created_at: "2026-10-12T10:00:00Z",
    ...overrides,
  };
}

export const LAP_P1 = "01M4TH84XCQYGAD34XR0C5BTDH";
export const LAP_P2 = "01M4R0P48JA6JMPQ1WR91DCP3S";
export const LAP_P3 = "01M4R0P8W0WS12C7HSENZ9628E";

export function leaderboard(
  trackId = "spielberg",
  entries?: LeaderboardEntry[],
): LeaderboardResponse {
  return {
    track_id: trackId,
    driver_type: null,
    entries: entries ?? [
      entry({
        rank: 1,
        lap_id: LAP_P1,
        driver_type: "model",
        driver_name: "lapzero-ppo",
        driver_label: "AI v8",
        model_version: "8",
        lap_time_s: 45.98,
        created_at: "2026-10-13T08:00:00Z",
      }),
      entry({
        rank: 2,
        lap_id: LAP_P2,
        driver_type: "model",
        driver_name: "lapzero-ppo",
        driver_label: "AI v5",
        model_version: "5",
        lap_time_s: 46.78,
        created_at: "2026-10-12T08:00:00Z",
      }),
      entry({
        rank: 3,
        lap_id: LAP_P3,
        driver_label: "Mara Okafor",
        driver_name: "Mara Okafor",
        lap_time_s: 59.6,
      }),
    ],
  };
}

export function champion(overrides: Partial<ChampionResponse> = {}): ChampionResponse {
  return {
    model_name: "lapzero-policy",
    version: "8",
    run_id: "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
    contract_version: "1.1.0",
    tags: {
      algorithm: "ppo",
      gate_result: "promote",
      gate_track: "oschersleben",
      gate_champion_sweep: "01M4TGPRVQY5F71QAQKBSR5RKQ",
      gate_challenger_sweep: "01M4TH84XCQYGAD34XR0C5BTDH",
      replaced_champion_version: "5",
    },
    created_at: "2026-10-13T07:30:00Z",
    ...overrides,
  };
}

/** A 2 s replay at 10 Hz driving along y = 0 at 5 m/s, heading 0, steering ramping left. */
export function straightReplay(lapId = LAP_P1, trackId = "spielberg"): Replay {
  const t = Array.from({ length: 21 }, (_, i) => Math.round(i) / 10);
  return {
    schema: "lapzero.replay",
    schema_version: "1.0.0",
    lap_id: lapId,
    track_id: trackId,
    driver_label: "AI v8",
    lap_time_s: 45.98,
    hz: 10,
    points: {
      t,
      x: t.map((s) => 5 * s),
      y: t.map(() => 0),
      theta: t.map(() => 0),
      v: t.map(() => 5),
      steer: t.map((s) => Math.round((s / 2) * 1000) / 1000),
    },
  };
}

/** The contract Section 6 example replay. */
export const CONTRACT_EXAMPLE_REPLAY = {
  schema: "lapzero.replay",
  schema_version: "1.0.0",
  lap_id: "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
  track_id: "track_02",
  driver_label: "AI v7",
  lap_time_s: 35.97,
  hz: 10,
  points: {
    t: [0.0, 0.1, 0.2],
    x: [0.0, 0.162, 0.412],
    y: [0.0, 0.009, 0.031],
    theta: [0.0, 0.011, 0.024],
    v: [0.0, 1.46, 2.81],
    steer: [0.0, 0.071, 0.09],
  },
} as const;

/** A Response carrying `body` as JSON. */
export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
