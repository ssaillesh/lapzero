import { useRef } from "react";

import { useClockFrame } from "../../hooks/useReplayPlayback";
import { formatLapTime } from "../../lib/formatters";
import type { PlaybackClock, PlaybackRate, PlaybackSnapshot } from "../../lib/playback";
import { PLAYBACK_RATES } from "../../lib/playback";
import { SegmentedControl } from "../common/Controls";

interface ReplayControlsProps {
  clock: PlaybackClock;
  snapshot: PlaybackSnapshot;
}

const RATE_OPTIONS = PLAYBACK_RATES.map((rate) => ({ value: String(rate), label: `${rate}×` }));

function isRate(value: number): value is PlaybackRate {
  return PLAYBACK_RATES.some((rate) => rate === value);
}

/** Play and pause glyphs built from plain shapes (no icon dependency, CLAUDE.md rule 4). */
function PlayGlyph({ playing }: { playing: boolean }) {
  if (playing) {
    return (
      <span aria-hidden="true" className="flex h-3 w-3 justify-between">
        <span className="h-full w-[4px] bg-current" />
        <span className="h-full w-[4px] bg-current" />
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className="ml-0.5 h-3 w-3 bg-current [clip-path:polygon(0_0,100%_50%,0_100%)]"
    />
  );
}

function RestartGlyph() {
  return (
    <span aria-hidden="true" className="flex h-3 items-center gap-px">
      <span className="h-full w-[2px] bg-current" />
      <span className="h-full w-2.5 bg-current [clip-path:polygon(100%_0,0_50%,100%_100%)]" />
    </span>
  );
}

/**
 * Transport for a replay. The scrubber and time label are written directly on every frame (no
 * React render per frame). Dragging the scrubber pauses playback until release, so the running
 * clock never fights the user's thumb.
 */
export function ReplayControls({ clock, snapshot }: ReplayControlsProps) {
  const rangeRef = useRef<HTMLInputElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);
  const resumeAfterDrag = useRef(false);
  const duration = clock.duration;
  const durationText = formatLapTime(duration);

  useClockFrame(clock, (time) => {
    const fraction = duration > 0 ? time / duration : 0;
    if (rangeRef.current !== null) {
      rangeRef.current.value = String(time);
      rangeRef.current.setAttribute(
        "aria-valuetext",
        `${time.toFixed(1)} of ${duration.toFixed(1)} seconds`,
      );
    }
    if (fillRef.current !== null) fillRef.current.style.transform = `scaleX(${fraction})`;
    if (timeRef.current !== null) {
      const text = formatLapTime(time);
      if (timeRef.current.textContent !== text) timeRef.current.textContent = text;
    }
  });

  const beginDrag = () => {
    resumeAfterDrag.current = snapshot.playing;
    clock.pause();
  };
  const endDrag = () => {
    if (resumeAfterDrag.current) clock.play();
    resumeAfterDrag.current = false;
  };

  const playLabel = snapshot.playing ? "Pause" : snapshot.ended ? "Replay from start" : "Play";

  return (
    <div
      className="flex flex-wrap items-center gap-x-4 gap-y-3"
      role="group"
      aria-label="Replay controls"
    >
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={clock.toggle}
          disabled={!clock.canPlay}
          aria-label={playLabel}
          title={`${playLabel} (Space)`}
          className="press flex h-9 w-9 items-center justify-center rounded-[var(--radius-ctl)] bg-accent text-accent-ink hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <PlayGlyph playing={snapshot.playing} />
        </button>
        <button
          type="button"
          onClick={clock.restart}
          disabled={!clock.canPlay}
          aria-label="Restart"
          title="Restart (Home)"
          className="press flex h-9 w-9 items-center justify-center rounded-[var(--radius-ctl)] border border-line text-muted hover:border-line-strong hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          <RestartGlyph />
        </button>
      </div>

      <div className="flex min-w-[12rem] flex-1 items-center gap-3">
        <span className="w-14 text-right font-mono text-[13px] text-ink">
          <span className="sr-only">Elapsed </span>
          <span ref={timeRef}>{formatLapTime(0)}</span>
        </span>
        <div className="relative h-5 flex-1">
          <div
            aria-hidden="true"
            className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 overflow-hidden rounded-[2px] bg-line"
          >
            <div
              ref={fillRef}
              className="h-full w-full origin-left bg-accent"
              style={{ transform: "scaleX(0)" }}
            />
          </div>
          <input
            ref={rangeRef}
            type="range"
            min={0}
            max={duration}
            step={0.01}
            defaultValue={0}
            disabled={!clock.canPlay}
            aria-label="Replay position"
            onPointerDown={beginDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onChange={(event) => {
              clock.seek(Number(event.target.value));
            }}
            className="scrubber absolute inset-0 h-full w-full disabled:cursor-not-allowed"
          />
        </div>
        <span className="w-14 font-mono text-[13px] text-faint">
          <span className="sr-only">Duration </span>
          {durationText}
        </span>
      </div>

      <SegmentedControl
        label="Playback speed"
        hideLabel
        size="sm"
        value={String(snapshot.rate)}
        options={RATE_OPTIONS}
        onChange={(value) => {
          const rate = Number(value);
          if (isRate(rate)) clock.setRate(rate);
        }}
      />
    </div>
  );
}
