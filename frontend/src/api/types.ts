/**
 * Response types of the LapZero API, written once (CLAUDE.md section 7).
 *
 * Each type mirrors a backend response model; field names are the JSON names. Sources:
 *   HealthResponse       backend/app/schemas/health.py
 *   TrackSummary         backend/app/schemas/tracks.py (TrackSummary)
 *   TrackDetail          backend/app/schemas/tracks.py (TrackResponse) + lapzero/io/centreline.py
 *   LeaderboardResponse  backend/app/schemas/leaderboard.py
 *   Replay               lapzero/io/replay.py, schemas/replay.schema.json (contract Section 6)
 *   ChampionResponse     backend/app/schemas/models.py
 *
 * Units follow contract Section 0: metres, seconds, radians, m/s. Angles are counter-clockwise
 * positive, wrapped to (-pi, pi]. World coordinates are the F1TENTH map frame (y up).
 */

/** Who produced a lap's actions (lapzero.io.telemetry.DriverType). */
export type DriverType = "human" | "model";

export const DRIVER_TYPES: readonly DriverType[] = ["human", "model"];

/** ISO 8601 date-time with a UTC offset, as Pydantic serializes AwareDatetime. */
export type IsoDateTime = string;

/** Any JSON value; used for free-form metadata the API does not type further. */
export type JsonValue =
  string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

/** GET /api/health. */
export interface HealthResponse {
  status: "ok";
  /** State space contract version the API validates uploads against, e.g. "1.1.0". */
  contract_version: string;
}

/** One entry of GET /api/tracks: a track with at least one uploaded lap. */
export interface TrackSummary {
  track_id: string;
  /** Display name; the track ID until someone names it. */
  name: string;
  /** Centreline length, metres. */
  length_m: number;
  /** When the track was first seen. */
  created_at: IsoDateTime;
}

/** Columnar vertices of a closed loop in the world frame, metres. The last vertex joins the first. */
export interface Polyline {
  x: number[];
  y: number[];
}

/** Track outline in driving order. Edges are null when the centreline CSV has no widths. */
export interface TrackGeometry {
  centreline: Polyline;
  /** Left edge, seen in the driving direction. */
  left: Polyline | null;
  right: Polyline | null;
}

/** GET /api/tracks/{track_id}. */
export interface TrackDetail extends TrackSummary {
  metadata_json: Record<string, JsonValue>;
  /** Null when the server has no TRACKS_DIR/<track_id>/centreline.csv for the track. */
  geometry: TrackGeometry | null;
}

/** One driver's fastest completed lap on a track. Only completed laps rank. */
export interface LeaderboardEntry {
  /** 1 is the fastest driver. */
  rank: number;
  lap_id: string;
  driver_type: DriverType;
  driver_name: string;
  /** What the dashboard shows, e.g. "AI v7" for model version 7, or a human's name. */
  driver_label: string;
  /** Registry version of a model driver; null for humans. */
  model_version: string | null;
  /** Lap time, seconds; always positive. */
  lap_time_s: number;
  /** When the lap was recorded. */
  created_at: IsoDateTime;
}

/** GET /api/leaderboard/{track_id}: fastest drivers, fastest first. */
export interface LeaderboardResponse {
  track_id: string;
  /** The driver type filter applied, or null for all drivers. */
  driver_type: DriverType | null;
  entries: LeaderboardEntry[];
}

/**
 * Columnar 10 Hz replay samples; every array has the same length (at least one sample).
 *
 * t: seconds since the lap's first frame. x, y: world frame, metres. theta: heading, rad.
 * v: longitudinal speed, m/s. steer: normalized steering action in [-1, 1], -1 full right,
 * +1 full left (contract Section 2.1). The replay carries no lateral velocity, acceleration or
 * friction coefficient.
 */
export interface ReplayPoints {
  t: number[];
  x: number[];
  y: number[];
  theta: number[];
  v: number[];
  steer: number[];
}

/** GET /api/replays/{lap_id}: one lap downsampled to 10 Hz (contract Section 6). */
export interface Replay {
  schema: "lapzero.replay";
  schema_version: string;
  lap_id: string;
  track_id: string;
  driver_label: string;
  /** Lap time, seconds; null for a lap that did not complete. */
  lap_time_s: number | null;
  /** Sample rate, Hz; 10 under contract 1.x. */
  hz: number;
  points: ReplayPoints;
}

/** GET /api/models/champion: the registry version the champion alias points at. */
export interface ChampionResponse {
  /** Registered model name, e.g. "lapzero-policy". */
  model_name: string;
  /** Registry version number, e.g. "8". */
  version: string;
  /** MLflow run that trained the version; null if it was registered without one. */
  run_id: string | null;
  /** Contract version the training run logged; null if unknown. */
  contract_version: string | null;
  /** Version tags, e.g. gate_result set by the release gate (lapzero/registry.py). */
  tags: Record<string, string>;
  /** When the version was registered. */
  created_at: IsoDateTime;
}

/** Version tags the release gate writes on a promoted champion (lapzero/registry.py). */
export const GATE_TAGS = {
  result: "gate_result",
  track: "gate_track",
  championSweep: "gate_champion_sweep",
  challengerSweep: "gate_challenger_sweep",
  replacedVersion: "replaced_champion_version",
} as const;

/** Largest `limit` GET /api/leaderboard accepts. */
export const LEADERBOARD_MAX_LIMIT = 100;
