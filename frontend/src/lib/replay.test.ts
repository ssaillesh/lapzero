import { describe, expect, it } from "vitest";

import type { ReplayPoints } from "../api/types";
import {
  chartRows,
  emptySample,
  indexAt,
  lerpAngle,
  prepareReplay,
  sampleAt,
  wrapAngle,
} from "./replay";

function points(
  rows: [t: number, x: number, y: number, theta: number, v: number, steer: number][],
): ReplayPoints {
  return {
    t: rows.map((r) => r[0]),
    x: rows.map((r) => r[1]),
    y: rows.map((r) => r[2]),
    theta: rows.map((r) => r[3]),
    v: rows.map((r) => r[4]),
    steer: rows.map((r) => r[5]),
  };
}

describe("prepareReplay", () => {
  it("handles an empty replay", () => {
    const replay = prepareReplay(points([]));
    expect(replay.length).toBe(0);
    expect(replay.duration).toBe(0);
    const sample = sampleAt(replay, 3, emptySample());
    expect([sample.time, sample.x, sample.y]).toEqual([0, 0, 0]);
  });

  it("gives a single-sample replay zero duration and returns that sample at any time", () => {
    const replay = prepareReplay(points([[0, 4, 5, 1, 2, 0.5]]));
    expect(replay.duration).toBe(0);
    const sample = sampleAt(replay, 10, emptySample());
    expect([sample.x, sample.y, sample.theta, sample.v, sample.steer]).toEqual([4, 5, 1, 2, 0.5]);
  });

  it("drops non-finite values and times that do not increase, and counts them", () => {
    const replay = prepareReplay(
      points([
        [0, 0, 0, 0, 1, 0],
        [0.1, Number.NaN, 0, 0, 1, 0],
        [0.1, 1, 0, 0, 1, 0],
        [0.05, 2, 0, 0, 1, 0],
        [0.2, 3, 0, 0, Number.POSITIVE_INFINITY, 0],
        [0.3, 4, 0, 0, 1, 2],
      ]),
    );
    expect(Array.from(replay.t)).toEqual([0, 0.1, 0.3]);
    expect(Array.from(replay.x)).toEqual([0, 1, 4]);
    expect(replay.dropped).toBe(3);
    expect(replay.steer[2]).toBe(1); // clamped to the contract's [-1, 1]
  });

  it("measures time from the first sample", () => {
    const replay = prepareReplay(
      points([
        [2, 0, 0, 0, 0, 0],
        [3, 1, 0, 0, 0, 0],
      ]),
    );
    expect(replay.startTime).toBe(2);
    expect(replay.duration).toBe(1);
    expect(sampleAt(replay, 0.5, emptySample()).x).toBeCloseTo(0.5);
  });
});

describe("indexAt", () => {
  const replay = prepareReplay(points([0, 0.1, 0.25, 0.4, 1].map((t) => [t, t, 0, 0, 0, 0])));

  it("finds the sample at or before a time on an irregular grid", () => {
    expect(indexAt(replay, 0)).toBe(0);
    expect(indexAt(replay, 0.1)).toBe(1);
    expect(indexAt(replay, 0.3)).toBe(2);
    expect(indexAt(replay, 0.99)).toBe(3);
    expect(indexAt(replay, 1)).toBe(4);
  });

  it("clamps outside the replay", () => {
    expect(indexAt(replay, -5)).toBe(0);
    expect(indexAt(replay, 50)).toBe(4);
  });

  it("gives the same answer with a wrong hint", () => {
    expect(indexAt(replay, 0.3, 4)).toBe(2);
    expect(indexAt(replay, 0.3, 1)).toBe(2);
  });
});

describe("sampleAt", () => {
  it("interpolates position, speed and steering linearly on t", () => {
    const replay = prepareReplay(
      points([
        [0, 0, 10, 0, 2, -1],
        [0.1, 1, 12, 0, 4, 1],
      ]),
    );
    const sample = sampleAt(replay, 0.025, emptySample());
    expect(sample.x).toBeCloseTo(0.25);
    expect(sample.y).toBeCloseTo(10.5);
    expect(sample.v).toBeCloseTo(2.5);
    expect(sample.steer).toBeCloseTo(-0.5);
    expect(sample.index).toBe(0);
  });

  it("turns the short way across the +-pi wrap instead of spinning", () => {
    const replay = prepareReplay(
      points([
        [0, 0, 0, 3.1, 1, 0],
        [0.1, 0, 0, -3.1, 1, 0],
      ]),
    );
    const sample = sampleAt(replay, 0.05, emptySample());
    expect(Math.abs(sample.theta)).toBeCloseTo(Math.PI, 5);
  });

  it("treats a non-finite time as the start", () => {
    const replay = prepareReplay(
      points([
        [0, 1, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0],
      ]),
    );
    expect(sampleAt(replay, Number.NaN, emptySample()).x).toBe(1);
  });
});

describe("angles", () => {
  it("wraps to (-pi, pi]", () => {
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI);
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(Math.PI);
    expect(wrapAngle(2 * Math.PI + 0.5)).toBeCloseTo(0.5);
    expect(wrapAngle(-0.5)).toBeCloseTo(-0.5);
  });

  it("interpolates the short way in both directions", () => {
    expect(lerpAngle(-3, 3, 0.5)).toBeCloseTo(Math.PI, 1);
    expect(lerpAngle(0.2, -0.2, 0.5)).toBeCloseTo(0);
    expect(lerpAngle(1, 2, 0.25)).toBeCloseTo(1.25);
  });
});

describe("chartRows", () => {
  it("returns every sample of a short replay", () => {
    const replay = prepareReplay(
      points([
        [0, 0, 0, 0, 1, 0],
        [0.1, 0, 0, 0, 2, 0],
      ]),
    );
    expect(chartRows(replay)).toEqual([
      { t: 0, v: 1, steer: 0 },
      { t: 0.1, v: 2, steer: 0 },
    ]);
  });

  it("decimates a long replay but keeps its speed and steering peaks", () => {
    const rows: [number, number, number, number, number, number][] = [];
    for (let i = 0; i < 5000; i++) rows.push([i / 10, 0, 0, 0, 5, 0]);
    rows[1234] = [123.4, 0, 0, 0, 19.5, 0];
    rows[4321] = [432.1, 0, 0, 0, 5, -0.97];
    const decimated = chartRows(prepareReplay(points(rows)), 500);
    expect(decimated.length).toBeLessThanOrEqual(510);
    expect(Math.max(...decimated.map((r) => r.v))).toBe(19.5);
    expect(Math.min(...decimated.map((r) => r.steer))).toBe(-0.97);
    expect(decimated.at(-1)?.t).toBeCloseTo(499.9);
    const times = decimated.map((r) => r.t);
    expect([...times].sort((a, b) => a - b)).toEqual(times);
  });
});
