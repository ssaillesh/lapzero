/**
 * Canvas drawing for the track map: a static layer (metric grid, track surface and edges,
 * centreline, the whole driven line, lap start) drawn once per size change into an offscreen
 * canvas, and a dynamic layer (the driven line so far, coloured by speed, and the car) drawn every
 * frame on top. Geometry and replay data stay in world metres; only the transform knows pixels.
 */

import type { TrackGeometry } from "../api/types";
import type { PreparedReplay, ReplaySample } from "./replay";
import type { ViewTransform } from "./transform";
import {
  boundsOf,
  fitTransform,
  headingToCanvas,
  niceScaleLength,
  toCanvasX,
  toCanvasY,
} from "./transform";
import type { ColumnarPoints as Points } from "./transform";

/** The subset of CanvasRenderingContext2D the renderer uses (keeps tests honest and small). */
export type Canvas2D = Pick<
  CanvasRenderingContext2D,
  | "save"
  | "restore"
  | "beginPath"
  | "closePath"
  | "moveTo"
  | "lineTo"
  | "arc"
  | "fill"
  | "stroke"
  | "fillRect"
  | "clearRect"
  | "setTransform"
  | "translate"
  | "rotate"
  | "setLineDash"
  | "fillText"
  | "drawImage"
  | "fillStyle"
  | "strokeStyle"
  | "lineWidth"
  | "lineCap"
  | "lineJoin"
  | "font"
  | "textBaseline"
  | "globalAlpha"
>;

/** Colours, kept in step with the tokens in src/index.css. */
export const PALETTE = {
  grid: "rgba(255, 255, 255, 0.035)",
  surface: "#15181c",
  edge: "#3b4148",
  centreline: "#2f353c",
  ghost: "rgba(232, 233, 235, 0.16)",
  start: "#e8e9eb",
  label: "#767c84",
  accent: "#d2f34c",
  accentHalo: "rgba(210, 243, 76, 0.16)",
  carOutline: "#0b0c0e",
} as const;

/** Speed bins of the driven line, slow to fast: the accent from dim to full strength. */
export const SPEED_COLOURS = [
  "rgba(210, 243, 76, 0.22)",
  "rgba(210, 243, 76, 0.36)",
  "rgba(210, 243, 76, 0.52)",
  "rgba(210, 243, 76, 0.68)",
  "rgba(210, 243, 76, 0.84)",
  "rgba(210, 243, 76, 1)",
] as const;

export interface Scene {
  geometry: TrackGeometry | null;
  replay: PreparedReplay | null;
}

export interface Layout {
  view: ViewTransform;
  width: number;
  height: number;
  /** World metres between grid lines (also the scale bar length). */
  gridStep: number;
}

const PADDING_PX = 28;

/** Fit the scene into width x height CSS pixels, or null if it has nothing to draw. */
export function layoutScene(scene: Scene, width: number, height: number): Layout | null {
  if (width <= 0 || height <= 0) return null;
  const sets: { x: ArrayLike<number>; y: ArrayLike<number> }[] = [];
  const { geometry, replay } = scene;
  if (geometry !== null) {
    sets.push(geometry.centreline);
    if (geometry.left !== null) sets.push(geometry.left);
    if (geometry.right !== null) sets.push(geometry.right);
  }
  if (replay !== null && replay.length > 0) sets.push(replay);
  const bounds = boundsOf(sets);
  if (bounds === null) return null;
  const view = fitTransform(bounds, width, height, PADDING_PX);
  return { view, width, height, gridStep: niceScaleLength(view.scale, 96) };
}

/**
 * Speed bin of each driven segment (sample i to i + 1), relative to the replay's top speed.
 * Computed once per replay.
 */
export function speedBins(replay: PreparedReplay): Uint8Array {
  const bins = new Uint8Array(Math.max(replay.length - 1, 0));
  const top = replay.maxSpeed > 0 ? replay.maxSpeed : 1;
  const last = SPEED_COLOURS.length - 1;
  for (let i = 0; i < bins.length; i++) {
    const mean = (Math.abs(replay.v[i] ?? 0) + Math.abs(replay.v[i + 1] ?? 0)) / 2;
    bins[i] = Math.min(last, Math.floor((mean / top) * SPEED_COLOURS.length));
  }
  return bins;
}

function tracePolyline(ctx: Canvas2D, view: ViewTransform, line: Points, close: boolean): void {
  const count = Math.min(line.x.length, line.y.length);
  for (let i = 0; i < count; i++) {
    const x = toCanvasX(view, line.x[i] ?? 0);
    const y = toCanvasY(view, line.y[i] ?? 0);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  if (close && count > 0) ctx.closePath();
}

function drawGrid(ctx: Canvas2D, layout: Layout): void {
  const { view, width, height, gridStep } = layout;
  if (!(gridStep * view.scale >= 12)) return;
  // Lines at whole multiples of gridStep metres, so the grid is anchored to the world frame.
  const leftWorld = view.minX - view.originX / view.scale;
  const topWorld = view.maxY + view.originY / view.scale;
  ctx.beginPath();
  for (let k = Math.ceil(leftWorld / gridStep); ; k++) {
    const x = Math.round(toCanvasX(view, k * gridStep)) + 0.5;
    if (x > width) break;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let k = Math.floor(topWorld / gridStep); ; k--) {
    const y = Math.round(toCanvasY(view, k * gridStep)) + 0.5;
    if (y > height) break;
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.strokeStyle = PALETTE.grid;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawTrack(ctx: Canvas2D, view: ViewTransform, geometry: TrackGeometry): void {
  const { centreline, left, right } = geometry;
  if (left !== null && right !== null) {
    // Two closed loops filled even-odd: the ring between the edges is the track surface.
    ctx.beginPath();
    tracePolyline(ctx, view, left, true);
    tracePolyline(ctx, view, right, true);
    ctx.fillStyle = PALETTE.surface;
    ctx.fill("evenodd");
    ctx.beginPath();
    tracePolyline(ctx, view, left, true);
    tracePolyline(ctx, view, right, true);
    ctx.strokeStyle = PALETTE.edge;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }
  ctx.beginPath();
  tracePolyline(ctx, view, centreline, true);
  ctx.setLineDash(left === null || right === null ? [] : [3, 5]);
  ctx.strokeStyle = left === null || right === null ? PALETTE.edge : PALETTE.centreline;
  ctx.lineWidth = left === null || right === null ? 1.5 : 1;
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawGhostLine(ctx: Canvas2D, view: ViewTransform, replay: PreparedReplay): void {
  if (replay.length < 2) return;
  ctx.beginPath();
  tracePolyline(ctx, view, replay, false);
  ctx.strokeStyle = PALETTE.ghost;
  ctx.lineWidth = 1.25;
  ctx.lineJoin = "round";
  ctx.stroke();
}

function drawStart(ctx: Canvas2D, view: ViewTransform, replay: PreparedReplay): void {
  if (replay.length === 0) return;
  const x = toCanvasX(view, replay.x[0] ?? 0);
  const y = toCanvasY(view, replay.y[0] ?? 0);
  const angle = headingToCanvas(replay.theta[0] ?? 0);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0, -9);
  ctx.lineTo(0, 9);
  ctx.strokeStyle = PALETTE.start;
  ctx.lineWidth = 2;
  ctx.lineCap = "butt";
  ctx.stroke();
  ctx.restore();
  ctx.font = '500 10px ui-monospace, "SF Mono", Menlo, monospace';
  ctx.textBaseline = "middle";
  ctx.fillStyle = PALETTE.label;
  ctx.fillText("START", x + 12, y - 12);
}

/** Draw everything that does not change during playback. Coordinates are CSS pixels. */
export function drawStaticLayer(ctx: Canvas2D, scene: Scene, layout: Layout): void {
  ctx.clearRect(0, 0, layout.width, layout.height);
  drawGrid(ctx, layout);
  if (scene.geometry !== null) drawTrack(ctx, layout.view, scene.geometry);
  if (scene.replay !== null) {
    drawGhostLine(ctx, layout.view, scene.replay);
    drawStart(ctx, layout.view, scene.replay);
  }
}

/** Draw the driven line up to `sample`, coloured by speed bin, in as few strokes as possible. */
export function drawTrace(
  ctx: Canvas2D,
  view: ViewTransform,
  replay: PreparedReplay,
  bins: Uint8Array,
  sample: ReplaySample,
): void {
  if (replay.length < 2) return;
  ctx.lineWidth = 2.25;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  let i = 0;
  const end = Math.min(sample.index, replay.length - 1);
  while (i < end) {
    const bin = bins[i] ?? 0;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(view, replay.x[i] ?? 0), toCanvasY(view, replay.y[i] ?? 0));
    while (i < end && (bins[i] ?? 0) === bin) {
      i++;
      ctx.lineTo(toCanvasX(view, replay.x[i] ?? 0), toCanvasY(view, replay.y[i] ?? 0));
    }
    ctx.strokeStyle = SPEED_COLOURS[bin] ?? PALETTE.accent;
    ctx.stroke();
  }
  // The partial segment from the last sample to the interpolated car position.
  ctx.beginPath();
  ctx.moveTo(toCanvasX(view, replay.x[end] ?? 0), toCanvasY(view, replay.y[end] ?? 0));
  ctx.lineTo(toCanvasX(view, sample.x), toCanvasY(view, sample.y));
  ctx.strokeStyle = SPEED_COLOURS[bins[end] ?? bins[end - 1] ?? 0] ?? PALETTE.accent;
  ctx.stroke();
}

/**
 * Draw the car at the sample: a chevron pointing along the heading. World theta is
 * counter-clockwise with y up; the canvas flips y, so the chevron rotates by -theta.
 */
export function drawCar(ctx: Canvas2D, view: ViewTransform, sample: ReplaySample): void {
  const x = toCanvasX(view, sample.x);
  const y = toCanvasY(view, sample.y);
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  ctx.arc(0, 0, 12, 0, Math.PI * 2);
  ctx.fillStyle = PALETTE.accentHalo;
  ctx.fill();
  ctx.rotate(headingToCanvas(sample.theta));
  ctx.beginPath();
  ctx.moveTo(9, 0);
  ctx.lineTo(-6, 5.5);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-6, -5.5);
  ctx.closePath();
  ctx.lineJoin = "round";
  ctx.lineWidth = 2;
  ctx.strokeStyle = PALETTE.carOutline;
  ctx.stroke();
  ctx.fillStyle = PALETTE.accent;
  ctx.fill();
  ctx.restore();
}
