import { describe, expect, it } from "vitest";

import { champion, entry, leaderboard } from "../test/fixtures";
import { isChampion } from "./champion";
import { fastestTime, filterEntries, nextSort, sortEntries } from "./leaderboard";

const entries = leaderboard().entries;

describe("sortEntries", () => {
  it("sorts by driver label with numbers in order", () => {
    const sorted = sortEntries(
      [
        entry({ rank: 1, lap_id: "a", driver_label: "AI v10" }),
        entry({ rank: 2, lap_id: "b", driver_label: "AI v9" }),
      ],
      { key: "driver", direction: "asc" },
    );
    expect(sorted.map((e) => e.driver_label)).toEqual(["AI v9", "AI v10"]);
  });

  it("sorts by lap time and recording date in either direction", () => {
    expect(sortEntries(entries, { key: "lap_time", direction: "desc" }).map((e) => e.rank)).toEqual(
      [3, 2, 1],
    );
    // Recorded: P1 on 13 Oct, P3 on 12 Oct 10:00, P2 on 12 Oct 08:00.
    expect(sortEntries(entries, { key: "recorded", direction: "desc" }).map((e) => e.rank)).toEqual(
      [1, 3, 2],
    );
  });

  it("does not modify its input", () => {
    const copy = [...entries];
    sortEntries(entries, { key: "lap_time", direction: "desc" });
    expect(entries).toEqual(copy);
  });
});

describe("nextSort", () => {
  it("flips the direction of the active column and starts a new one at its default", () => {
    expect(nextSort({ key: "rank", direction: "asc" }, "rank")).toEqual({
      key: "rank",
      direction: "desc",
    });
    expect(nextSort({ key: "rank", direction: "asc" }, "recorded")).toEqual({
      key: "recorded",
      direction: "desc",
    });
  });
});

describe("filterEntries", () => {
  it("matches driver, model version and lap ID, ignoring case", () => {
    expect(filterEntries(entries, "mara").map((e) => e.rank)).toEqual([3]);
    expect(filterEntries(entries, "v5").map((e) => e.rank)).toEqual([2]);
    expect(filterEntries(entries, "01M4TH").map((e) => e.rank)).toEqual([1]);
    expect(filterEntries(entries, "  ")).toHaveLength(3);
  });
});

describe("fastestTime and isChampion", () => {
  it("finds the fastest time", () => {
    expect(fastestTime(entries)).toBe(45.98);
    expect(fastestTime([])).toBeNull();
  });

  it("matches the champion by model registry version, never a human", () => {
    const current = champion({ version: "5" });
    expect(entries.map((e) => isChampion(e, current))).toEqual([false, true, false]);
    expect(isChampion(entry({ rank: 1, lap_id: "h", model_version: null }), current)).toBe(false);
    expect(isChampion(entries[1]!, null)).toBe(false);
  });
});
