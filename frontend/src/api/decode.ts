/**
 * Runtime decoders for API responses.
 *
 * Responses are untrusted until decoded: each decoder checks the shape against src/api/types.ts
 * and throws DecodeError naming the first bad field (as a JSON path). Unknown extra fields are
 * ignored so a contract minor version that adds optional fields does not break the dashboard.
 * Numbers must be finite; JSON cannot carry NaN or inf (CLAUDE.md rule 7).
 */

import type {
  ChampionResponse,
  DriverType,
  HealthResponse,
  JsonValue,
  LeaderboardEntry,
  LeaderboardResponse,
  Polyline,
  Replay,
  ReplayPoints,
  TrackDetail,
  TrackGeometry,
  TrackSummary,
} from "./types";
import { DRIVER_TYPES } from "./types";

/** A response field does not have the shape its type promises. */
export class DecodeError extends Error {
  readonly path: string;

  constructor(path: string, reason: string) {
    super(`${path}: ${reason}`);
    this.name = "DecodeError";
    this.path = path;
  }
}

type Fields = Record<string, unknown>;

function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function isFields(value: unknown): value is Fields {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function object(value: unknown, path: string): Fields {
  if (!isFields(value)) throw new DecodeError(path, `expected object, got ${describe(value)}`);
  return value;
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string")
    throw new DecodeError(path, `expected string, got ${describe(value)}`);
  return value;
}

function nonEmptyString(value: unknown, path: string): string {
  const text = string(value, path);
  if (text.length === 0) throw new DecodeError(path, "expected a non-empty string");
  return text;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DecodeError(path, `expected a finite number, got ${describe(value)}`);
  }
  return value;
}

function integer(value: unknown, path: string): number {
  const number = finite(value, path);
  if (!Number.isInteger(number)) throw new DecodeError(path, `expected an integer, got ${number}`);
  return number;
}

function nullable<T>(value: unknown, path: string, decode: (v: unknown, p: string) => T): T | null {
  return value === null ? null : decode(value, path);
}

function array<T>(value: unknown, path: string, decode: (v: unknown, p: string) => T): T[] {
  if (!Array.isArray(value)) throw new DecodeError(path, `expected array, got ${describe(value)}`);
  return value.map((item: unknown, index) => decode(item, `${path}[${index}]`));
}

function numbers(value: unknown, path: string): number[] {
  return array(value, path, finite);
}

function isOneOf<T extends string>(value: string, options: readonly T[]): value is T {
  return options.some((option) => option === value);
}

function oneOf<T extends string>(value: unknown, path: string, options: readonly T[]): T {
  const text = string(value, path);
  if (!isOneOf(text, options)) {
    throw new DecodeError(path, `expected one of ${options.join(", ")}, got "${text}"`);
  }
  return text;
}

function dateTime(value: unknown, path: string): string {
  const text = string(value, path);
  if (Number.isNaN(Date.parse(text)))
    throw new DecodeError(path, `expected a date-time, got "${text}"`);
  return text;
}

function json(value: unknown, path: string): JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return finite(value, path);
  if (Array.isArray(value)) return array(value, path, json);
  return stringRecord(value, path, json);
}

function stringRecord<T>(
  value: unknown,
  path: string,
  decode: (v: unknown, p: string) => T,
): Record<string, T> {
  const fields = object(value, path);
  const result: Record<string, T> = {};
  for (const [key, item] of Object.entries(fields)) result[key] = decode(item, `${path}.${key}`);
  return result;
}

function driverType(value: unknown, path: string): DriverType {
  return oneOf(value, path, DRIVER_TYPES);
}

export function decodeHealth(value: unknown, path = "$"): HealthResponse {
  const o = object(value, path);
  return {
    status: oneOf(o.status, `${path}.status`, ["ok"]),
    contract_version: nonEmptyString(o.contract_version, `${path}.contract_version`),
  };
}

export function decodeTrackSummary(value: unknown, path = "$"): TrackSummary {
  const o = object(value, path);
  return {
    track_id: nonEmptyString(o.track_id, `${path}.track_id`),
    name: string(o.name, `${path}.name`),
    length_m: finite(o.length_m, `${path}.length_m`),
    created_at: dateTime(o.created_at, `${path}.created_at`),
  };
}

export function decodeTrackList(value: unknown, path = "$"): TrackSummary[] {
  return array(value, path, decodeTrackSummary);
}

function decodePolyline(value: unknown, path: string): Polyline {
  const o = object(value, path);
  const x = numbers(o.x, `${path}.x`);
  const y = numbers(o.y, `${path}.y`);
  if (x.length !== y.length) {
    throw new DecodeError(path, `x has ${x.length} vertices but y has ${y.length}`);
  }
  if (x.length < 3) throw new DecodeError(path, `a closed loop needs 3 vertices, got ${x.length}`);
  return { x, y };
}

function decodeGeometry(value: unknown, path: string): TrackGeometry {
  const o = object(value, path);
  return {
    centreline: decodePolyline(o.centreline, `${path}.centreline`),
    left: nullable(o.left, `${path}.left`, decodePolyline),
    right: nullable(o.right, `${path}.right`, decodePolyline),
  };
}

export function decodeTrackDetail(value: unknown, path = "$"): TrackDetail {
  const o = object(value, path);
  return {
    ...decodeTrackSummary(o, path),
    metadata_json: stringRecord(o.metadata_json, `${path}.metadata_json`, json),
    geometry: nullable(o.geometry, `${path}.geometry`, decodeGeometry),
  };
}

function decodeLeaderboardEntry(value: unknown, path: string): LeaderboardEntry {
  const o = object(value, path);
  const rank = integer(o.rank, `${path}.rank`);
  if (rank < 1) throw new DecodeError(`${path}.rank`, `expected rank >= 1, got ${rank}`);
  const lapTime = finite(o.lap_time_s, `${path}.lap_time_s`);
  if (lapTime <= 0) throw new DecodeError(`${path}.lap_time_s`, `expected > 0, got ${lapTime}`);
  return {
    rank,
    lap_id: nonEmptyString(o.lap_id, `${path}.lap_id`),
    driver_type: driverType(o.driver_type, `${path}.driver_type`),
    driver_name: string(o.driver_name, `${path}.driver_name`),
    driver_label: string(o.driver_label, `${path}.driver_label`),
    model_version: nullable(o.model_version, `${path}.model_version`, string),
    lap_time_s: lapTime,
    created_at: dateTime(o.created_at, `${path}.created_at`),
  };
}

export function decodeLeaderboard(value: unknown, path = "$"): LeaderboardResponse {
  const o = object(value, path);
  return {
    track_id: nonEmptyString(o.track_id, `${path}.track_id`),
    driver_type: nullable(o.driver_type, `${path}.driver_type`, driverType),
    entries: array(o.entries, `${path}.entries`, decodeLeaderboardEntry),
  };
}

const REPLAY_COLUMNS = ["t", "x", "y", "theta", "v", "steer"] as const;

function decodeReplayPoints(value: unknown, path: string): ReplayPoints {
  const o = object(value, path);
  const points: ReplayPoints = {
    t: numbers(o.t, `${path}.t`),
    x: numbers(o.x, `${path}.x`),
    y: numbers(o.y, `${path}.y`),
    theta: numbers(o.theta, `${path}.theta`),
    v: numbers(o.v, `${path}.v`),
    steer: numbers(o.steer, `${path}.steer`),
  };
  const lengths = REPLAY_COLUMNS.map((name) => points[name].length);
  if (new Set(lengths).size !== 1) {
    const listed = REPLAY_COLUMNS.map((name, i) => `${name}=${lengths[i] ?? 0}`).join(", ");
    throw new DecodeError(path, `columns must have equal length, got ${listed}`);
  }
  if (points.t.length === 0) throw new DecodeError(`${path}.t`, "expected at least one sample");
  return points;
}

export function decodeReplay(value: unknown, path = "$"): Replay {
  const o = object(value, path);
  const lapTime = nullable(o.lap_time_s, `${path}.lap_time_s`, finite);
  if (lapTime !== null && lapTime <= 0) {
    throw new DecodeError(`${path}.lap_time_s`, `expected > 0 or null, got ${lapTime}`);
  }
  return {
    schema: oneOf(o.schema, `${path}.schema`, ["lapzero.replay"]),
    schema_version: nonEmptyString(o.schema_version, `${path}.schema_version`),
    lap_id: nonEmptyString(o.lap_id, `${path}.lap_id`),
    track_id: nonEmptyString(o.track_id, `${path}.track_id`),
    driver_label: nonEmptyString(o.driver_label, `${path}.driver_label`),
    lap_time_s: lapTime,
    hz: integer(o.hz, `${path}.hz`),
    points: decodeReplayPoints(o.points, `${path}.points`),
  };
}

export function decodeChampion(value: unknown, path = "$"): ChampionResponse {
  const o = object(value, path);
  return {
    model_name: nonEmptyString(o.model_name, `${path}.model_name`),
    version: nonEmptyString(o.version, `${path}.version`),
    run_id: nullable(o.run_id, `${path}.run_id`, string),
    contract_version: nullable(o.contract_version, `${path}.contract_version`, string),
    tags: stringRecord(o.tags, `${path}.tags`, string),
    created_at: dateTime(o.created_at, `${path}.created_at`),
  };
}
