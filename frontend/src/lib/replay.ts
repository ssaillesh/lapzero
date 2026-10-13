/**
 * Replay sampling for the canvas and readouts: sanitize the columnar points once, then look up
 * an interpolated sample at any time without allocating per frame.
 *
 * Contract Section 6: t is seconds since the lap's first frame at 10 Hz; x, y in metres (world
 * frame, y up); theta in rad, counter-clockwise positive, wrapped to (-pi, pi]; v in m/s; steer
 * is the normalized action in [-1, 1]. The backend guarantees strictly increasing t and finite
 * values; this module still drops any sample that breaks that, rather than drawing garbage.
 */

import type { ReplayPoints } from "../api/types";

/** Sanitized replay columns. `t` is strictly increasing; every column has `length` values. */
export interface PreparedReplay {
  readonly t: Float64Array;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly theta: Float64Array;
  readonly v: Float64Array;
  readonly steer: Float64Array;
  readonly length: number;
  /** Time of the first sample, s (0 for a contract replay). */
  readonly startTime: number;
  /** Time span from the first to the last sample, s; 0 for a single sample. */
  readonly duration: number;
  /** Highest speed in the replay, m/s (0 if all samples are stationary). */
  readonly maxSpeed: number;
  /** Samples dropped because a value was not finite or t did not increase. */
  readonly dropped: number;
}

/** One interpolated sample. Mutated in place by sampleAt to avoid per-frame allocation. */
export interface ReplaySample {
  /** Replay time, s since the first sample. */
  time: number;
  x: number;
  y: number;
  theta: number;
  v: number;
  steer: number;
  /** Index of the sample at or before `time`. */
  index: number;
}

export function emptySample(): ReplaySample {
  return { time: 0, x: 0, y: 0, theta: 0, v: 0, steer: 0, index: 0 };
}

/**
 * Sanitize replay points: keep a sample only if every value is finite and its t is greater than
 * the previous kept sample's. Steering is clamped to [-1, 1].
 *
 * @returns The prepared replay; `length` is 0 if no sample survives.
 */
export function prepareReplay(points: ReplayPoints): PreparedReplay {
  const count = Math.min(
    points.t.length,
    points.x.length,
    points.y.length,
    points.theta.length,
    points.v.length,
    points.steer.length,
  );
  const t = new Float64Array(count);
  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const theta = new Float64Array(count);
  const v = new Float64Array(count);
  const steer = new Float64Array(count);
  let kept = 0;
  let maxSpeed = 0;
  for (let i = 0; i < count; i++) {
    const ti = points.t[i];
    const xi = points.x[i];
    const yi = points.y[i];
    const thi = points.theta[i];
    const vi = points.v[i];
    const si = points.steer[i];
    if (
      ti === undefined ||
      xi === undefined ||
      yi === undefined ||
      thi === undefined ||
      vi === undefined ||
      si === undefined ||
      !Number.isFinite(ti) ||
      !Number.isFinite(xi) ||
      !Number.isFinite(yi) ||
      !Number.isFinite(thi) ||
      !Number.isFinite(vi) ||
      !Number.isFinite(si)
    ) {
      continue;
    }
    if (kept > 0 && ti <= (t[kept - 1] ?? Number.NEGATIVE_INFINITY)) continue;
    t[kept] = ti;
    x[kept] = xi;
    y[kept] = yi;
    theta[kept] = thi;
    v[kept] = vi;
    steer[kept] = Math.max(-1, Math.min(1, si));
    maxSpeed = Math.max(maxSpeed, Math.abs(vi));
    kept++;
  }
  const startTime = kept > 0 ? (t[0] ?? 0) : 0;
  const endTime = kept > 0 ? (t[kept - 1] ?? 0) : 0;
  return {
    t: t.subarray(0, kept),
    x: x.subarray(0, kept),
    y: y.subarray(0, kept),
    theta: theta.subarray(0, kept),
    v: v.subarray(0, kept),
    steer: steer.subarray(0, kept),
    length: kept,
    startTime,
    duration: endTime - startTime,
    maxSpeed,
    dropped: Math.max(points.t.length, count) - kept,
  };
}

/** Wrap an angle in rad to (-pi, pi]. */
export function wrapAngle(angle: number): number {
  const twoPi = 2 * Math.PI;
  const wrapped = angle - twoPi * Math.floor((angle + Math.PI) / twoPi);
  // floor maps the interval to [-pi, pi); move -pi to +pi to match the contract's (-pi, pi].
  return wrapped <= -Math.PI ? wrapped + twoPi : wrapped;
}

/**
 * Interpolate between two headings the short way round, so crossing +-pi does not spin.
 *
 * @param from Heading at fraction 0, rad.
 * @param to Heading at fraction 1, rad.
 * @param fraction 0 to 1.
 * @returns The heading, wrapped to (-pi, pi].
 */
export function lerpAngle(from: number, to: number, fraction: number): number {
  const delta = wrapAngle(to - from);
  return wrapAngle(from + delta * fraction);
}

/**
 * Index of the last sample with t <= time (binary search), clamped to [0, length - 1].
 *
 * @param hint A likely answer, e.g. the previous frame's index; checked first so sequential
 *   playback is O(1).
 */
export function indexAt(replay: PreparedReplay, time: number, hint = 0): number {
  const { t, length } = replay;
  if (length <= 1) return 0;
  const absolute = replay.startTime + time;
  const last = length - 1;
  if (absolute <= (t[0] ?? 0)) return 0;
  if (absolute >= (t[last] ?? 0)) return last;
  if (hint >= 0 && hint < last && (t[hint] ?? 0) <= absolute && absolute < (t[hint + 1] ?? 0))
    return hint;
  const next = hint + 1;
  if (next < last && (t[next] ?? 0) <= absolute && absolute < (t[next + 1] ?? 0)) return next;
  let low = 0;
  let high = last;
  while (high - low > 1) {
    const mid = (low + high) >>> 1;
    if ((t[mid] ?? 0) <= absolute) low = mid;
    else high = mid;
  }
  return low;
}

/**
 * Write the replay state at `time` into `out`: x, y, v and steer interpolate linearly on t,
 * theta the short way round. Times outside the replay clamp to its first or last sample.
 *
 * @param time Seconds since the first sample.
 * @param out Sample to overwrite; its `index` is used as the lookup hint.
 * @returns `out`.
 */
export function sampleAt(replay: PreparedReplay, time: number, out: ReplaySample): ReplaySample {
  const clamped = Math.max(0, Math.min(replay.duration, Number.isFinite(time) ? time : 0));
  out.time = clamped;
  if (replay.length === 0) {
    out.x = out.y = out.theta = out.v = out.steer = 0;
    out.index = 0;
    return out;
  }
  const i = indexAt(replay, clamped, out.index);
  const j = Math.min(i + 1, replay.length - 1);
  const t0 = replay.t[i] ?? 0;
  const t1 = replay.t[j] ?? t0;
  const fraction = j === i || t1 === t0 ? 0 : (replay.startTime + clamped - t0) / (t1 - t0);
  const f = Math.max(0, Math.min(1, fraction));
  out.index = i;
  out.x = lerp(replay.x[i] ?? 0, replay.x[j] ?? 0, f);
  out.y = lerp(replay.y[i] ?? 0, replay.y[j] ?? 0, f);
  out.theta = lerpAngle(replay.theta[i] ?? 0, replay.theta[j] ?? 0, f);
  out.v = lerp(replay.v[i] ?? 0, replay.v[j] ?? 0, f);
  out.steer = lerp(replay.steer[i] ?? 0, replay.steer[j] ?? 0, f);
  return out;
}

function lerp(a: number, b: number, f: number): number {
  return a + (b - a) * f;
}

/** One row of the telemetry charts: replay time (s since start), speed (m/s), steering. */
export interface ChartRow {
  t: number;
  v: number;
  steer: number;
}

/**
 * Rows for the telemetry charts, at most about `maxPoints` of them.
 *
 * Long replays are decimated by keeping, per bucket, the samples with the lowest and highest
 * speed and steering, so peaks survive; the replay itself is unchanged.
 */
export function chartRows(replay: PreparedReplay, maxPoints = 1200): ChartRow[] {
  const rows: ChartRow[] = [];
  const row = (i: number): ChartRow => ({
    t: (replay.t[i] ?? 0) - replay.startTime,
    v: replay.v[i] ?? 0,
    steer: replay.steer[i] ?? 0,
  });
  if (replay.length <= maxPoints) {
    for (let i = 0; i < replay.length; i++) rows.push(row(i));
    return rows;
  }
  // Each bucket can contribute up to 4 extremes plus its first sample.
  const buckets = Math.max(1, Math.floor(maxPoints / 5));
  const size = replay.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size);
    const end = Math.min(replay.length, Math.floor((b + 1) * size));
    const picks = new Set<number>([start]);
    let vMin = start,
      vMax = start,
      sMin = start,
      sMax = start;
    for (let i = start; i < end; i++) {
      const v = replay.v[i] ?? 0;
      const s = replay.steer[i] ?? 0;
      if (v < (replay.v[vMin] ?? 0)) vMin = i;
      if (v > (replay.v[vMax] ?? 0)) vMax = i;
      if (s < (replay.steer[sMin] ?? 0)) sMin = i;
      if (s > (replay.steer[sMax] ?? 0)) sMax = i;
    }
    for (const i of [vMin, vMax, sMin, sMax]) picks.add(i);
    for (const i of [...picks].sort((a, b2) => a - b2)) rows.push(row(i));
  }
  const last = replay.length - 1;
  if (rows.at(-1)?.t !== (replay.t[last] ?? 0) - replay.startTime) rows.push(row(last));
  return rows;
}
