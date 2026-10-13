import { describe, expect, it } from "vitest";

import {
  formatGap,
  formatHeading,
  formatLapTime,
  formatLength,
  formatPercent,
  formatSteer,
  majorVersion,
} from "./formatters";

describe("formatLapTime", () => {
  it("keeps the 2 decimal places the telemetry records", () => {
    expect(formatLapTime(45.98)).toBe("45.98");
    expect(formatLapTime(0)).toBe("0.00");
  });

  it("adds minutes from 60 s and carries rounding into them", () => {
    expect(formatLapTime(61.84)).toBe("1:01.84");
    expect(formatLapTime(89.04)).toBe("1:29.04");
    expect(formatLapTime(59.999)).toBe("1:00.00");
  });

  it("shows n/a for values that are not a time", () => {
    expect(formatLapTime(Number.NaN)).toBe("n/a");
    expect(formatLapTime(-1)).toBe("n/a");
  });
});

describe("formatGap and formatPercent", () => {
  it("signs the gap", () => {
    expect(formatGap(0.8)).toBe("+0.80");
    expect(formatGap(-0.4)).toBe("-0.40");
    expect(formatGap(0.001)).toBe("0.00");
  });

  it("formats a relative change", () => {
    expect(formatPercent(-0.0171)).toBe("-1.71 %");
    expect(formatPercent(0.02)).toBe("+2.00 %");
  });
});

describe("formatSteer", () => {
  it("names the side: -1 is full right, +1 full left (contract Section 2.1)", () => {
    expect(formatSteer(0.42)).toBe("0.42 L");
    expect(formatSteer(-1)).toBe("1.00 R");
    expect(formatSteer(0.001)).toBe("0.00");
  });
});

describe("other formatters", () => {
  it("formats lengths in m and km", () => {
    expect(formatLength(82.4)).toBe("82.4 m");
    expect(formatLength(4318.6)).toBe("4.32 km");
  });

  it("formats headings counter-clockwise from +x in degrees", () => {
    expect(formatHeading(0)).toBe("0°");
    expect(formatHeading(Math.PI / 2)).toBe("90°");
    expect(formatHeading(-Math.PI / 2)).toBe("270°");
  });

  it("reads the major number of a semantic version", () => {
    expect(majorVersion("1.1.0")).toBe("1");
    expect(majorVersion("12.0.3")).toBe("12");
    expect(majorVersion("v1")).toBeNull();
  });
});
