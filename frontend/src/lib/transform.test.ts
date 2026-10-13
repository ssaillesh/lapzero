import { describe, expect, it } from "vitest";

import {
  boundsOf,
  fitTransform,
  headingToCanvas,
  niceScaleLength,
  toCanvasX,
  toCanvasY,
} from "./transform";

describe("boundsOf", () => {
  it("spans every finite point of every set", () => {
    expect(
      boundsOf([
        { x: [0, 4], y: [1, 2] },
        { x: [-3], y: [9] },
      ]),
    ).toEqual({ minX: -3, minY: 1, maxX: 4, maxY: 9 });
  });

  it("skips non-finite points and returns null with none left", () => {
    expect(boundsOf([{ x: [Number.NaN, 2], y: [0, 3] }])).toEqual({
      minX: 2,
      minY: 3,
      maxX: 2,
      maxY: 3,
    });
    expect(boundsOf([{ x: [Number.NaN], y: [1] }])).toBeNull();
  });
});

describe("fitTransform", () => {
  const bounds = { minX: 0, minY: 0, maxX: 40, maxY: 20 };

  it("uses one scale for both axes, limited by the tighter one", () => {
    const view = fitTransform(bounds, 500, 500, 50); // 400 px for 40 m wide, 400 px for 20 m high
    expect(view.scale).toBe(10);
  });

  it("centres the world bounds in the canvas", () => {
    const view = fitTransform(bounds, 500, 500, 50);
    expect(toCanvasX(view, 20)).toBe(250);
    expect(toCanvasY(view, 10)).toBe(250);
  });

  it("flips y: world up is canvas up", () => {
    const view = fitTransform(bounds, 500, 500, 50);
    expect(toCanvasY(view, 20)).toBeLessThan(toCanvasY(view, 0));
    expect(toCanvasY(view, 20)).toBe(150);
    expect(toCanvasX(view, 0)).toBe(50);
  });

  it("gives a single point a finite, positive scale", () => {
    const view = fitTransform({ minX: 5, minY: 5, maxX: 5, maxY: 5 }, 200, 100, 10);
    expect(Number.isFinite(view.scale)).toBe(true);
    expect(view.scale).toBeGreaterThan(0);
    expect(toCanvasX(view, 5)).toBe(100);
    expect(toCanvasY(view, 5)).toBe(50);
  });

  it("survives a canvas smaller than its padding", () => {
    const view = fitTransform(bounds, 10, 10, 50);
    expect(view.scale).toBeGreaterThan(0);
  });
});

describe("headingToCanvas", () => {
  it("negates the counter-clockwise world heading for the y-down canvas", () => {
    // Heading +pi/2 points to world +y; on the canvas that is up, i.e. rotation -pi/2.
    expect(headingToCanvas(Math.PI / 2)).toBe(-Math.PI / 2);
    const angle = headingToCanvas(Math.PI / 2);
    expect(Math.sin(angle)).toBeCloseTo(-1); // canvas y decreases: up
  });
});

describe("niceScaleLength", () => {
  it("picks 1, 2 or 5 x 10^n metres near the target length", () => {
    expect(niceScaleLength(10, 96)).toBe(10); // 9.6 m wanted
    expect(niceScaleLength(4, 96)).toBe(20); // 24 m wanted
    expect(niceScaleLength(2, 96)).toBe(50); // 48 m wanted
    expect(niceScaleLength(100, 96)).toBe(1); // 0.96 m wanted
  });

  it("falls back to 1 m for a degenerate scale", () => {
    expect(niceScaleLength(0)).toBe(1);
    expect(niceScaleLength(Number.NaN)).toBe(1);
  });
});
