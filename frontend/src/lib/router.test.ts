import { describe, expect, it } from "vitest";

import { DEFAULT_LEADERBOARD_LIMIT, hrefFor, navigate, parseRoute } from "./router";

describe("parseRoute", () => {
  it("parses every view", () => {
    expect(parseRoute("/", "")).toEqual({ name: "home" });
    expect(parseRoute("/governance", "")).toEqual({ name: "governance" });
    expect(parseRoute("/tracks", "")).toEqual({ name: "tracks" });
    expect(parseRoute("/tracks/spielberg", "")).toEqual({
      name: "track",
      trackId: "spielberg",
      lapId: null,
    });
    expect(parseRoute("/tracks/spielberg/replays/01M4TH84XCQYGAD34XR0C5BTDH", "")).toEqual({
      name: "track",
      trackId: "spielberg",
      lapId: "01M4TH84XCQYGAD34XR0C5BTDH",
    });
  });

  it("reads leaderboard filters from the query and ignores invalid ones", () => {
    expect(parseRoute("/leaderboard", "?track=oval&driver=model&limit=50")).toEqual({
      name: "leaderboard",
      params: { trackId: "oval", driverType: "model", limit: 50 },
    });
    expect(parseRoute("/leaderboard", "?driver=robot&limit=-3")).toEqual({
      name: "leaderboard",
      params: { trackId: null, driverType: null, limit: DEFAULT_LEADERBOARD_LIMIT },
    });
    expect(parseRoute("/leaderboard", "?limit=5000")).toMatchObject({ params: { limit: 100 } });
  });

  it("treats unknown paths, extra segments and bad encodings as not found", () => {
    expect(parseRoute("/laps", "")).toEqual({ name: "not-found", path: "/laps" });
    expect(parseRoute("/tracks/a/replays", "")).toMatchObject({ name: "not-found" });
    expect(parseRoute("/tracks/a/replays/b/c", "")).toMatchObject({ name: "not-found" });
    expect(parseRoute("/tracks/%E0%A4%A", "")).toMatchObject({ name: "not-found" });
  });
});

describe("hrefFor", () => {
  it("round-trips through parseRoute", () => {
    const routes = [
      { name: "track", trackId: "spielberg wide", lapId: "01M4TH84XCQYGAD34XR0C5BTDH" },
      { name: "leaderboard", params: { trackId: "oval", driverType: "human", limit: 10 } },
      {
        name: "leaderboard",
        params: { trackId: null, driverType: null, limit: DEFAULT_LEADERBOARD_LIMIT },
      },
    ] as const;
    for (const route of routes) {
      const href = hrefFor(route);
      const url = new URL(href, "http://localhost");
      expect(parseRoute(url.pathname, url.search)).toEqual(route);
    }
  });

  it("omits default filters from the URL", () => {
    expect(
      hrefFor({
        name: "leaderboard",
        params: { trackId: null, driverType: null, limit: DEFAULT_LEADERBOARD_LIMIT },
      }),
    ).toBe("/leaderboard");
  });
});

describe("navigate", () => {
  it("pushes or replaces history entries", () => {
    const before = window.history.length;
    navigate("/governance");
    expect(window.location.pathname).toBe("/governance");
    expect(window.history.length).toBe(before + 1);
    navigate("/tracks", { replace: true });
    expect(window.location.pathname).toBe("/tracks");
    expect(window.history.length).toBe(before + 1);
  });
});

describe("comparison lap (?vs=)", () => {
  it("reads the lap to compare with on a replay URL", () => {
    expect(parseRoute("/tracks/spielberg/replays/A1", "?vs=B2")).toEqual({
      name: "track",
      trackId: "spielberg",
      lapId: "A1",
      compareLapId: "B2",
    });
  });

  it("ignores it without an open lap, or when empty", () => {
    expect(parseRoute("/tracks/spielberg", "?vs=B2")).toStrictEqual({
      name: "track",
      trackId: "spielberg",
      lapId: null,
    });
    expect(parseRoute("/tracks/spielberg/replays/A1", "?vs=")).toStrictEqual({
      name: "track",
      trackId: "spielberg",
      lapId: "A1",
    });
  });

  it("round-trips through hrefFor, encoded", () => {
    const route = {
      name: "track",
      trackId: "spielberg",
      lapId: "A1",
      compareLapId: "B 2",
    } as const;
    const href = hrefFor(route);
    expect(href).toBe("/tracks/spielberg/replays/A1?vs=B%202");
    const [path = "", search = ""] = href.split("?");
    expect(parseRoute(path, `?${search}`)).toEqual(route);
  });
});
