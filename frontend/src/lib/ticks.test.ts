import { describe, expect, it } from "vitest";

import { niceStep, niceTicks } from "./ticks";

describe("niceStep", () => {
  it("rounds to 1, 2 or 5 x 10^n", () => {
    expect(niceStep(8, 4)).toBe(2);
    expect(niceStep(77.1, 5)).toBe(20);
    expect(niceStep(0.9, 4)).toBeCloseTo(0.5);
    expect(niceStep(0, 4)).toBe(1);
  });
});

describe("niceTicks", () => {
  it("extends a value axis past its maximum", () => {
    expect(niceTicks(7.84, 4, true)).toEqual([0, 2, 4, 6, 8]);
    expect(niceTicks(19.5, 4, true)).toEqual([0, 5, 10, 15, 20]);
  });

  it("stops a time axis at or before the end of the data", () => {
    expect(niceTicks(77.1, 5, false)).toEqual([0, 20, 40, 60]);
    expect(niceTicks(2, 5, false)).toEqual([0, 0.5, 1, 1.5, 2]);
  });
});
