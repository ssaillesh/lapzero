import { describe, expect, it } from "vitest";

import { TRACKS } from "../test/fixtures";
import { isHiddenTrack, visibleTracks } from "./tracks";

const WIDE = { ...TRACKS[1]!, track_id: "spielberg_wide", name: "spielberg_wide" };
const OVAL = { ...TRACKS[1]!, track_id: "oval", name: "oval", length_m: 60.73 };

describe("hidden tracks", () => {
  it("hides spielberg_wide and oval and nothing else", () => {
    expect(isHiddenTrack("spielberg_wide")).toBe(true);
    expect(isHiddenTrack("oval")).toBe(true);
    expect(isHiddenTrack("spielberg")).toBe(false);
    expect(isHiddenTrack("moscowraceway")).toBe(false);
    expect(isHiddenTrack("oschersleben")).toBe(false);
  });

  it("drops hidden tracks from a list and keeps the others in order", () => {
    expect(visibleTracks([OVAL, TRACKS[0]!, WIDE, TRACKS[1]!])).toEqual(TRACKS);
  });

  it("leaves a list without hidden tracks unchanged", () => {
    expect(visibleTracks(TRACKS)).toEqual(TRACKS);
    expect(visibleTracks([])).toEqual([]);
  });
});
