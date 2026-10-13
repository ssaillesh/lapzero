import { describe, expect, it } from "vitest";

import { squareTrack, straightReplay } from "../test/fixtures";
import { emptySample, prepareReplay, sampleAt } from "./replay";
import type { Canvas2D } from "./trackRenderer";
import {
  SPEED_COLOURS,
  drawCar,
  drawStaticLayer,
  drawTrace,
  layoutScene,
  speedBins,
} from "./trackRenderer";
import { toCanvasX, toCanvasY } from "./transform";

type Call = [name: string, ...args: unknown[]];

/** A 2D context that records the drawing calls the renderer makes. */
function recordingContext(): { ctx: Canvas2D; calls: Call[] } {
  const calls: Call[] = [];
  const record =
    (name: string) =>
    (...args: unknown[]) => {
      calls.push([name, ...args]);
    };
  const ctx: Canvas2D = {
    save: record("save"),
    restore: record("restore"),
    beginPath: record("beginPath"),
    closePath: record("closePath"),
    moveTo: record("moveTo"),
    lineTo: record("lineTo"),
    arc: record("arc"),
    fill: record("fill"),
    stroke: record("stroke"),
    fillRect: record("fillRect"),
    clearRect: record("clearRect"),
    setTransform: record("setTransform"),
    translate: record("translate"),
    rotate: record("rotate"),
    setLineDash: record("setLineDash"),
    fillText: record("fillText"),
    drawImage: record("drawImage"),
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    lineJoin: "miter",
    font: "",
    textBaseline: "alphabetic",
    globalAlpha: 1,
  };
  return { ctx, calls };
}

describe("layoutScene", () => {
  it("returns null with nothing to draw or no room", () => {
    expect(layoutScene({ geometry: null, replay: null }, 800, 400)).toBeNull();
    expect(layoutScene({ geometry: squareTrack().geometry, replay: null }, 0, 400)).toBeNull();
  });

  it("fits track edges and the replay together", () => {
    const layout = layoutScene({ geometry: squareTrack().geometry, replay: null }, 800, 400);
    expect(layout).not.toBeNull();
    // Outer edge spans x -2..42: the whole track stays inside the canvas.
    expect(toCanvasX(layout!.view, -2)).toBeGreaterThanOrEqual(0);
    expect(toCanvasX(layout!.view, 42)).toBeLessThanOrEqual(800);
  });
});

describe("drawCar", () => {
  it("draws at the transformed position and rotates by -theta (world CCW, canvas y down)", () => {
    const replay = prepareReplay(straightReplay().points);
    const layout = layoutScene({ geometry: squareTrack().geometry, replay }, 800, 400)!;
    const sample = sampleAt(replay, 1, emptySample());
    sample.theta = Math.PI / 2; // facing world +y
    const { ctx, calls } = recordingContext();
    drawCar(ctx, layout.view, sample);
    const translate = calls.find((c) => c[0] === "translate");
    expect(translate?.[1]).toBeCloseTo(toCanvasX(layout.view, sample.x));
    expect(translate?.[2]).toBeCloseTo(toCanvasY(layout.view, sample.y));
    expect(calls.find((c) => c[0] === "rotate")?.[1]).toBeCloseTo(-Math.PI / 2);
  });
});

describe("speedBins and drawTrace", () => {
  it("puts the fastest segment in the brightest bin", () => {
    const replay = prepareReplay({
      t: [0, 1, 2, 3],
      x: [0, 1, 2, 3],
      y: [0, 0, 0, 0],
      theta: [0, 0, 0, 0],
      v: [0, 0, 10, 10],
      steer: [0, 0, 0, 0],
    });
    const bins = speedBins(replay);
    expect(bins).toHaveLength(3);
    expect(bins[0]).toBe(0);
    expect(bins[2]).toBe(SPEED_COLOURS.length - 1);
  });

  it("strokes only up to the car, one path per run of equal speed", () => {
    const replay = prepareReplay(straightReplay().points); // constant speed: one bin
    const layout = layoutScene({ geometry: null, replay }, 800, 400)!;
    const { ctx, calls } = recordingContext();
    drawTrace(ctx, layout.view, replay, speedBins(replay), sampleAt(replay, 1.05, emptySample()));
    // One run for samples 0..10 plus the partial segment to the interpolated car.
    expect(calls.filter((c) => c[0] === "stroke")).toHaveLength(2);
    const lastLine = calls.filter((c) => c[0] === "lineTo").at(-1);
    expect(lastLine?.[1]).toBeCloseTo(toCanvasX(layout.view, 5.25));
  });
});

describe("drawStaticLayer", () => {
  it("fills the track surface even-odd between the edges and marks the lap start", () => {
    const replay = prepareReplay(straightReplay().points);
    const layout = layoutScene({ geometry: squareTrack().geometry, replay }, 800, 400)!;
    const { ctx, calls } = recordingContext();
    drawStaticLayer(ctx, { geometry: squareTrack().geometry, replay }, layout);
    expect(calls).toContainEqual(["fill", "evenodd"]);
    expect(calls).toContainEqual(["fillText", "START", expect.any(Number), expect.any(Number)]);
  });
});
