/**
 * Display formatting. Times keep the 2 decimal places the telemetry records (contract Section
 * 5.2: t to 2 dp), so the dashboard never shows more precision than the data has.
 */

const TIME_DECIMALS = 2;

/** Lap or replay time: "45.98" under a minute, "1:21.04" above. Non-finite values give "n/a". */
export function formatLapTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "n/a";
  const factor = 10 ** TIME_DECIMALS;
  const hundredths = Math.round(seconds * factor);
  const minutes = Math.floor(hundredths / (60 * factor));
  const rest = (hundredths - minutes * 60 * factor) / factor;
  if (minutes === 0) return rest.toFixed(TIME_DECIMALS);
  return `${minutes}:${rest.toFixed(TIME_DECIMALS).padStart(TIME_DECIMALS + 3, "0")}`;
}

/** Signed gap to a reference time, s: "+1.24", "-0.40", "0.00". */
export function formatGap(seconds: number): string {
  if (!Number.isFinite(seconds)) return "n/a";
  const rounded = Math.round(seconds * 100) / 100;
  if (rounded === 0) return "0.00";
  return `${rounded > 0 ? "+" : "-"}${Math.abs(rounded).toFixed(TIME_DECIMALS)}`;
}

/** Relative difference as a percentage with sign, e.g. "-1.71 %". */
export function formatPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return "n/a";
  const value = Math.round(fraction * 10_000) / 100;
  return `${value > 0 ? "+" : value < 0 ? "-" : ""}${Math.abs(value).toFixed(2)} %`;
}

/** Speed in m/s with 2 decimals (contract speeds are 2 to 3 dp). */
export function formatSpeed(metresPerSecond: number): string {
  return Number.isFinite(metresPerSecond) ? metresPerSecond.toFixed(2) : "n/a";
}

/** Length in metres: "82.4 m", or km from 1000 m. */
export function formatLength(metres: number): string {
  if (!Number.isFinite(metres)) return "n/a";
  if (Math.abs(metres) >= 1000) return `${(metres / 1000).toFixed(2)} km`;
  return `${metres.toFixed(1)} m`;
}

/** Normalized steering (-1 full right, +1 full left; contract Section 2.1): "0.42 L". */
export function formatSteer(steer: number): string {
  if (!Number.isFinite(steer)) return "n/a";
  const magnitude = Math.abs(steer).toFixed(2);
  if (magnitude === "0.00") return "0.00";
  return `${magnitude} ${steer > 0 ? "L" : "R"}`;
}

/** Position in metres with 1 decimal. */
export function formatMetres(metres: number): string {
  return Number.isFinite(metres) ? metres.toFixed(1) : "n/a";
}

/** Heading in degrees, from rad, 0 to 359 (counter-clockwise from world +x). */
export function formatHeading(theta: number): string {
  if (!Number.isFinite(theta)) return "n/a";
  const degrees = ((((theta * 180) / Math.PI) % 360) + 360) % 360;
  return `${Math.round(degrees) % 360}°`;
}

const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

const timeFormat = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

/** A timestamp in the viewer's locale and time zone, e.g. "13 Oct 2026, 14:32". */
export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "n/a" : dateTimeFormat.format(date);
}

/** A clock time in the viewer's time zone, e.g. "14:32:05". */
export function formatClockTime(date: Date): string {
  return timeFormat.format(date);
}

/** Major number of a semantic version ("1.1.0" gives "1"), or null if it is not semver. */
export function majorVersion(version: string): string | null {
  const match = /^(\d+)\.\d+\.\d+$/.exec(version);
  return match?.[1] ?? null;
}
