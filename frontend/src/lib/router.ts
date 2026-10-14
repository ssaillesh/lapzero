/**
 * A small History API router: three views do not justify a routing dependency (CLAUDE.md rule 4).
 *
 * Routes:
 *   /leaderboard?track=<id>&driver=<human|model>&limit=<n>
 *   /tracks
 *   /tracks/<track_id>
 *   /tracks/<track_id>/replays/<lap_id>?vs=<lap_id>   (vs: a second lap to compare with)
 *   /governance
 * "/" redirects to /leaderboard. Anything else is a not-found route.
 */

import { useSyncExternalStore } from "react";

import type { DriverType } from "../api/types";
import { LEADERBOARD_MAX_LIMIT } from "../api/types";

export const LEADERBOARD_LIMITS = [10, 25, 50, 100] as const;
export const DEFAULT_LEADERBOARD_LIMIT = 25;

export interface LeaderboardParams {
  trackId: string | null;
  driverType: DriverType | null;
  limit: number;
}

export type Route =
  | { name: "home" }
  | { name: "leaderboard"; params: LeaderboardParams }
  | { name: "tracks" }
  | { name: "track"; trackId: string; lapId: string | null; compareLapId?: string }
  | { name: "governance" }
  | { name: "not-found"; path: string };

function decodeSegment(segment: string): string | null {
  try {
    const value = decodeURIComponent(segment);
    return value === "" ? null : value;
  } catch {
    return null;
  }
}

function parseLimit(raw: string | null): number {
  const value = raw === null ? Number.NaN : Number(raw);
  if (!Number.isInteger(value) || value < 1) return DEFAULT_LEADERBOARD_LIMIT;
  return Math.min(value, LEADERBOARD_MAX_LIMIT);
}

/** Parse a path and query string into a route. */
export function parseRoute(pathname: string, search: string): Route {
  const segments = pathname.split("/").filter((segment) => segment !== "");
  const query = new URLSearchParams(search);
  const [first, second, third, fourth, ...rest] = segments;
  if (first === undefined) return { name: "home" };
  if (first === "leaderboard" && second === undefined) {
    const driver = query.get("driver");
    return {
      name: "leaderboard",
      params: {
        trackId: query.get("track") || null,
        driverType: driver === "human" || driver === "model" ? driver : null,
        limit: parseLimit(query.get("limit")),
      },
    };
  }
  if (first === "governance" && second === undefined) return { name: "governance" };
  if (first === "tracks") {
    if (second === undefined) return { name: "tracks" };
    const trackId = decodeSegment(second);
    if (trackId !== null && third === undefined) return { name: "track", trackId, lapId: null };
    if (trackId !== null && third === "replays" && fourth !== undefined && rest.length === 0) {
      const lapId = decodeSegment(fourth);
      const compareLapId = query.get("vs") || null;
      if (lapId !== null)
        return { name: "track", trackId, lapId, ...(compareLapId !== null && { compareLapId }) };
    }
  }
  return { name: "not-found", path: pathname };
}

/** The URL of a route; inverse of parseRoute. */
export function hrefFor(route: Exclude<Route, { name: "not-found" }>): string {
  switch (route.name) {
    case "home":
      return "/";
    case "leaderboard": {
      const query = new URLSearchParams();
      if (route.params.trackId !== null) query.set("track", route.params.trackId);
      if (route.params.driverType !== null) query.set("driver", route.params.driverType);
      if (route.params.limit !== DEFAULT_LEADERBOARD_LIMIT)
        query.set("limit", String(route.params.limit));
      const text = query.toString();
      return text === "" ? "/leaderboard" : `/leaderboard?${text}`;
    }
    case "tracks":
      return "/tracks";
    case "track": {
      const track = `/tracks/${encodeURIComponent(route.trackId)}`;
      if (route.lapId === null) return track;
      const replay = `${track}/replays/${encodeURIComponent(route.lapId)}`;
      const vs = route.compareLapId;
      return vs === undefined ? replay : `${replay}?vs=${encodeURIComponent(vs)}`;
    }
    case "governance":
      return "/governance";
  }
}

const NAVIGATE_EVENT = "lapzero:navigate";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  window.addEventListener(NAVIGATE_EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(NAVIGATE_EVENT, onChange);
  };
}

function currentLocation(): string {
  return `${window.location.pathname}${window.location.search}`;
}

export interface NavigateOptions {
  /** Replace the current history entry instead of pushing one. */
  replace?: boolean;
  /** Keep the scroll position (filter changes); otherwise scroll to the top. */
  keepScroll?: boolean;
}

/** Go to `href` inside the app without a page load. */
export function navigate(href: string, options: NavigateOptions = {}): void {
  if (href === currentLocation()) return;
  if (options.replace) window.history.replaceState(null, "", href);
  else window.history.pushState(null, "", href);
  window.dispatchEvent(new Event(NAVIGATE_EVENT));
  if (!options.keepScroll) window.scrollTo({ top: 0 });
}

/** The current route; re-renders on navigation and on the browser's back and forward. */
export function useRoute(): Route {
  const location = useSyncExternalStore(subscribe, currentLocation, currentLocation);
  const queryStart = location.indexOf("?");
  const pathname = queryStart === -1 ? location : location.slice(0, queryStart);
  const search = queryStart === -1 ? "" : location.slice(queryStart);
  return parseRoute(pathname, search);
}
