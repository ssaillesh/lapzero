/**
 * The only module that talks HTTP to the LapZero backend (CLAUDE.md section 7).
 *
 * Every backend router is mounted under /api (backend/app/main.py), including /api/health, so
 * one base URL covers all endpoints. Responses are decoded by src/api/decode.ts before they reach
 * a component; failures are thrown as ApiError with a typed reason.
 */

import {
  DecodeError,
  decodeChampion,
  decodeHealth,
  decodeLeaderboard,
  decodeReplay,
  decodeTrackDetail,
  decodeTrackList,
} from "./decode";
import { ApiError } from "./errors";
import type {
  ChampionResponse,
  DriverType,
  HealthResponse,
  LeaderboardResponse,
  Replay,
  TrackDetail,
  TrackSummary,
} from "./types";

export const DEFAULT_API_BASE_URL = "http://localhost:8000/api";
const DEFAULT_TIMEOUT_MS = 20_000;
const HEALTH_TIMEOUT_MS = 5_000;

/**
 * Normalize the configured API root.
 *
 * Trailing slashes are dropped. A bare origin ("http://host:8000") gets the /api prefix the
 * backend mounts its routers under; a value that already has a path ("…/api", or a proxy path)
 * is used as given, so /api is never doubled.
 *
 * @param raw VITE_API_BASE_URL, or undefined/empty for the default.
 * @returns The base URL without a trailing slash.
 */
export function resolveApiBaseUrl(raw: string | undefined): string {
  const value = raw?.trim() ?? "";
  if (value === "") return DEFAULT_API_BASE_URL;
  const trimmed = value.replace(/\/+$/, "");
  if (trimmed === "") return "/api"; // "/": same origin, e.g. behind a reverse proxy
  if (trimmed.startsWith("/")) return trimmed;
  const url = URL.canParse(trimmed) ? new URL(trimmed) : null;
  if (url === null || (url.protocol !== "http:" && url.protocol !== "https:")) {
    throw new Error(`VITE_API_BASE_URL must be an http(s) URL or an absolute path, got "${value}"`);
  }
  return url.pathname === "/" || url.pathname === "" ? `${url.origin}/api` : trimmed;
}

export interface LeaderboardQuery {
  /** Rank only humans or only models; null ranks both. */
  driverType: DriverType | null;
  /** Number of drivers returned, 1 to 100 (the backend's bound). */
  limit: number;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export interface ApiClientOptions {
  baseUrl: string;
  /** Injected in tests; defaults to the global fetch. */
  fetch?: Fetch;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImpl: Fetch;

  constructor(options: ApiClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
  }

  /** GET /api/health: the API is up, and its contract version. */
  health(options: RequestOptions = {}): Promise<HealthResponse> {
    return this.get("/health", decodeHealth, options, HEALTH_TIMEOUT_MS);
  }

  /** GET /api/tracks: every track that has uploaded laps, ordered by track ID. */
  listTracks(options: RequestOptions = {}): Promise<TrackSummary[]> {
    return this.get("/tracks", decodeTrackList, options);
  }

  /** GET /api/tracks/{track_id}: a track with its centreline and edges (404 if unknown). */
  getTrack(trackId: string, options: RequestOptions = {}): Promise<TrackDetail> {
    return this.get(`/tracks/${encodeURIComponent(trackId)}`, decodeTrackDetail, options);
  }

  /** GET /api/leaderboard/{track_id}: each driver's fastest completed lap, fastest first. */
  getLeaderboard(
    trackId: string,
    query: LeaderboardQuery,
    options: RequestOptions = {},
  ): Promise<LeaderboardResponse> {
    const params = new URLSearchParams({ limit: String(query.limit) });
    if (query.driverType !== null) params.set("driver_type", query.driverType);
    return this.get(
      `/leaderboard/${encodeURIComponent(trackId)}?${params.toString()}`,
      decodeLeaderboard,
      options,
    );
  }

  /** GET /api/replays/{lap_id}: the 10 Hz replay of a lap (404 if unknown). */
  getReplay(lapId: string, options: RequestOptions = {}): Promise<Replay> {
    return this.get(`/replays/${encodeURIComponent(lapId)}`, decodeReplay, options);
  }

  /** GET /api/models/champion: 404 without a champion, 503 if the registry cannot be read. */
  getChampion(options: RequestOptions = {}): Promise<ChampionResponse> {
    return this.get("/models/champion", decodeChampion, options);
  }

  private async get<T>(
    path: string,
    decode: (value: unknown) => T,
    options: RequestOptions,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const timeout = AbortSignal.timeout(timeoutMs);
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;

    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
    } catch {
      if (options.signal?.aborted) throw new ApiError({ kind: "aborted", url });
      if (timeout.aborted) throw new ApiError({ kind: "timeout", url, timeoutMs });
      throw new ApiError({ kind: "network", url });
    }

    if (!response.ok) {
      throw new ApiError({
        kind: "http",
        url,
        status: response.status,
        detail: await readDetail(response),
      });
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      if (options.signal?.aborted) throw new ApiError({ kind: "aborted", url });
      throw new ApiError({ kind: "malformed-json", url, status: response.status });
    }

    try {
      return decode(body);
    } catch (error) {
      if (error instanceof DecodeError) {
        throw new ApiError({
          kind: "invalid-response",
          url,
          path: error.path,
          reason: error.message,
        });
      }
      throw error;
    }
  }
}

/**
 * Read FastAPI's error detail: a string for HTTPException, or a list of {loc, msg} for 422s.
 * Returns null when the body has no usable detail.
 */
async function readDetail(response: Response): Promise<string | null> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return null;
  }
  if (typeof body !== "object" || body === null || !("detail" in body)) return null;
  const detail = body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.flatMap((issue: unknown) => {
      if (typeof issue !== "object" || issue === null || !("msg" in issue)) return [];
      const field: unknown =
        "loc" in issue && Array.isArray(issue.loc) ? issue.loc.at(-1) : undefined;
      const msg = typeof issue.msg === "string" ? issue.msg : "invalid";
      return [typeof field === "string" ? `${field}: ${msg}` : msg];
    });
    return messages.length > 0 ? messages.join("; ") : null;
  }
  return null;
}

/** The app's client, configured from VITE_API_BASE_URL (see frontend/.env.example). */
export const api = new ApiClient({ baseUrl: resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL) });
