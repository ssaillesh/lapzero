/** Axis ticks at round numbers (1, 2 or 5 x 10^n apart). */

/** The round step that splits `span` into about `count` intervals. */
export function niceStep(span: number, count: number): number {
  if (!(span > 0) || !Number.isFinite(span) || count < 1) return 1;
  const raw = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / magnitude;
  const factor = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return factor * magnitude;
}

/**
 * Ticks from 0 at a round step covering [0, max].
 *
 * @param extend Round the last tick up past `max` (for a value axis); otherwise stop at or below
 *   `max` (for a time axis whose domain ends exactly at the data).
 */
export function niceTicks(max: number, count: number, extend: boolean): number[] {
  const step = niceStep(max, count);
  const last = extend ? Math.ceil(max / step - 1e-9) * step : Math.floor(max / step + 1e-9) * step;
  const ticks: number[] = [];
  for (let i = 0; i * step <= last + step * 1e-9; i++) ticks.push(Math.round(i * step * 1e6) / 1e6);
  return ticks.length > 0 ? ticks : [0];
}
