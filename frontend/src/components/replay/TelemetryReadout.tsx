import { useRef } from "react";
import type { ReactNode } from "react";

import { useClockFrame } from "../../hooks/useReplayPlayback";
import {
  formatHeading,
  formatLapTime,
  formatMetres,
  formatSpeed,
  formatSteer,
} from "../../lib/formatters";
import type { PlaybackClock } from "../../lib/playback";
import type { PreparedReplay } from "../../lib/replay";
import { emptySample, sampleAt } from "../../lib/replay";

/**
 * The current sample as text: the canvas's text alternative and the precise numbers behind it.
 * Values are written to the DOM on each frame without re-rendering; screen readers read them on
 * demand. They are plain spans, not <output> (an implicit live region that would announce 60 times
 * a second).
 */
export function TelemetryReadout({
  clock,
  replay,
}: {
  clock: PlaybackClock;
  replay: PreparedReplay;
}) {
  const timeRef = useRef<HTMLSpanElement>(null);
  const speedRef = useRef<HTMLSpanElement>(null);
  const steerRef = useRef<HTMLSpanElement>(null);
  const positionRef = useRef<HTMLSpanElement>(null);
  const headingRef = useRef<HTMLSpanElement>(null);
  const steerMarker = useRef<HTMLSpanElement>(null);
  const sample = useRef(emptySample());

  useClockFrame(clock, (time) => {
    const s = sampleAt(replay, time, sample.current);
    write(timeRef.current, formatLapTime(s.time));
    write(speedRef.current, formatSpeed(s.v));
    write(steerRef.current, formatSteer(s.steer));
    write(positionRef.current, `${formatMetres(s.x)}, ${formatMetres(s.y)}`);
    write(headingRef.current, formatHeading(s.theta));
    // Steering +1 is full left (contract Section 2.1), so left of centre on the bar.
    if (steerMarker.current !== null)
      steerMarker.current.style.transform = `translateX(${(-s.steer * 50).toFixed(2)}%)`;
  });

  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-line sm:grid-cols-3 xl:grid-cols-5">
      <Cell label="Time" unit="s">
        <span ref={timeRef} data-readout>
          0.00
        </span>
      </Cell>
      <Cell label="Speed" unit="m/s">
        <span ref={speedRef} data-readout>
          0.00
        </span>
      </Cell>
      <Cell label="Steering" unit="norm.">
        <span className="flex flex-col gap-1.5">
          <span ref={steerRef} data-readout>
            0.00
          </span>
          <span
            aria-hidden="true"
            className="relative block h-1 w-full max-w-[9rem] rounded-[2px] bg-line"
          >
            <span className="absolute left-1/2 top-[-2px] h-2 w-px bg-faint" />
            {/* The wrapper spans the bar, so translateX(%) is a fraction of the bar's width. */}
            <span ref={steerMarker} className="absolute inset-0 block">
              <span className="absolute left-1/2 top-[-2px] -ml-[3px] block h-2 w-[6px] rounded-[1px] bg-steer" />
            </span>
          </span>
        </span>
      </Cell>
      <Cell label="Position x, y" unit="m" className="sm:col-span-2 xl:col-span-1">
        <span ref={positionRef} data-readout>
          0.0, 0.0
        </span>
      </Cell>
      <Cell label="Heading" unit="from +x, CCW" className="col-span-2 sm:col-span-1">
        <span ref={headingRef} data-readout>
          0°
        </span>
      </Cell>
    </dl>
  );
}

function write(element: HTMLElement | null, text: string): void {
  if (element !== null && element.textContent !== text) element.textContent = text;
}

function Cell({
  label,
  unit,
  className = "",
  children,
}: {
  label: string;
  unit: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={`min-w-0 bg-panel px-3.5 py-2.5 ${className}`}>
      <dt className="label flex items-baseline justify-between gap-2">
        <span>{label}</span>
        <span className="normal-case tracking-normal text-faint">{unit}</span>
      </dt>
      <dd className="mt-1 font-mono text-[15px] text-ink">{children}</dd>
    </div>
  );
}
