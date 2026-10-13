/**
 * The last URL seen in each section, so the navigation returns to the same track, filters or
 * replay instead of resetting them. Lives for the page session only.
 */

import type { Route } from "./router";

export type Section = "leaderboard" | "tracks" | "governance";

const lastHref: Record<Section, string> = {
  leaderboard: "/leaderboard",
  tracks: "/tracks",
  governance: "/governance",
};

export function rememberHref(section: Section, href: string): void {
  lastHref[section] = href;
}

export function lastHrefFor(section: Section): string {
  return lastHref[section];
}

/** The navigation section a route belongs to; null for not-found. */
export function sectionOf(route: Route): Section | null {
  switch (route.name) {
    case "home":
    case "leaderboard":
      return "leaderboard";
    case "tracks":
    case "track":
      return "tracks";
    case "governance":
      return "governance";
    case "not-found":
      return null;
  }
}
