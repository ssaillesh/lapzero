/**
 * World-to-canvas transform for the track map.
 *
 * World frame (contract Section 0): x right, y up, metres, angles counter-clockwise positive.
 * Canvas: x right, y down, CSS pixels. The transform keeps proportions (one scale for both axes),
 * centres the content, and flips y. A heading theta therefore draws at canvas angle -theta.
 */

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface ColumnarPoints {
  readonly x: ArrayLike<number>;
  readonly y: ArrayLike<number>;
}

/** Bounds of every finite point in `sets`, or null if there is none. */
export function boundsOf(sets: Iterable<ColumnarPoints>): Bounds | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const set of sets) {
    const count = Math.min(set.x.length, set.y.length);
    for (let i = 0; i < count; i++) {
      const x = set.x[i] ?? Number.NaN;
      const y = set.y[i] ?? Number.NaN;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return Number.isFinite(minX) ? { minX, minY, maxX, maxY } : null;
}

export interface ViewTransform {
  /** CSS pixels per metre. */
  readonly scale: number;
  /** Canvas x of world x = bounds.minX. */
  readonly originX: number;
  /** Canvas y of world y = bounds.maxY (the top edge). */
  readonly originY: number;
  readonly minX: number;
  readonly maxY: number;
}

/** Smallest world span drawn, metres, so a single point or a straight line still gets a scale. */
const MIN_SPAN_M = 1;

/**
 * Fit `bounds` into a width x height canvas with `padding` pixels on every side, centred.
 *
 * @returns A transform; its scale is at least a tiny positive number even for a zero-size canvas.
 */
export function fitTransform(
  bounds: Bounds,
  width: number,
  height: number,
  padding: number,
): ViewTransform {
  const spanX = Math.max(bounds.maxX - bounds.minX, MIN_SPAN_M);
  const spanY = Math.max(bounds.maxY - bounds.minY, MIN_SPAN_M);
  const availableX = Math.max(width - 2 * padding, 1);
  const availableY = Math.max(height - 2 * padding, 1);
  const scale = Math.max(Math.min(availableX / spanX, availableY / spanY), 1e-9);
  const centreX = (bounds.minX + bounds.maxX) / 2;
  const centreY = (bounds.minY + bounds.maxY) / 2;
  // Place the world centre at the canvas centre.
  const originX = width / 2 - (centreX - bounds.minX) * scale;
  const originY = height / 2 - (bounds.maxY - centreY) * scale;
  return { scale, originX, originY, minX: bounds.minX, maxY: bounds.maxY };
}

export function toCanvasX(view: ViewTransform, x: number): number {
  return view.originX + (x - view.minX) * view.scale;
}

export function toCanvasY(view: ViewTransform, y: number): number {
  return view.originY + (view.maxY - y) * view.scale;
}

/** Canvas rotation (rad, clockwise positive as canvas draws it) for a world heading. */
export function headingToCanvas(theta: number): number {
  return -theta;
}

/**
 * A "nice" scale bar length (1, 2 or 5 x 10^n metres) whose drawn length is closest to
 * `targetPx` without exceeding it by much.
 */
export function niceScaleLength(pixelsPerMetre: number, targetPx = 96): number {
  if (!(pixelsPerMetre > 0) || !Number.isFinite(pixelsPerMetre)) return 1;
  const raw = targetPx / pixelsPerMetre;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const steps = [1, 2, 5, 10];
  let best = magnitude;
  for (const step of steps) {
    const candidate = step * magnitude;
    if (candidate <= raw * 1.25) best = candidate;
  }
  return best;
}
