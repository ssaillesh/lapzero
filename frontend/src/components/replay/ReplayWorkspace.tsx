import { Suspense, lazy, useEffect, useMemo } from "react";
import type { ReactNode } from "react";

import type { Replay, TrackGeometry } from "../../api/types";
import { useReplayPlayback } from "../../hooks/useReplayPlayback";
import { compareLabel, formatLapTime, formatSpeed } from "../../lib/formatters";
import type { PlaybackClock } from "../../lib/playback";
import { prepareReplay } from "../../lib/replay";
import { Notice, Skeleton } from "../common/States";
import { ReplayCanvas } from "./ReplayCanvas";
import { ReplayControls } from "./ReplayControls";
import { TelemetryReadout } from "./TelemetryReadout";

// Recharts is most of the bundle; load it only when a replay opens.
const TelemetryCharts = lazy(() =>
  import("./TelemetryCharts").then((module) => ({ default: module.TelemetryCharts })),
);

/** Space or K play/pause, arrows seek 1 s (5 s with Shift), Home restarts. */
function usePlaybackShortcuts(clock: PlaybackClock): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable || target.closest("input, select, textarea") !== null)
      ) {
        return; // the scrubber, speed radios and filters handle their own keys
      }
      switch (event.key) {
        case " ":
          if (target instanceof HTMLButtonElement) return; // a focused button activates itself
          event.preventDefault();
          clock.toggle();
          break;
        case "k":
          clock.toggle();
          break;
        case "ArrowLeft":
          event.preventDefault();
          clock.seekBy(event.shiftKey ? -5 : -1);
          break;
        case "ArrowRight":
          event.preventDefault();
          clock.seekBy(event.shiftKey ? 5 : 1);
          break;
        case "Home":
          event.preventDefault();
          clock.restart();
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [clock]);
}

interface ReplayWorkspaceProps {
  replay: Replay;
  geometry: TrackGeometry | null;
  trackName: string;
  /** A second lap on the same track, raced against `replay` on the same clock; null for none. */
  compare?: Replay | null;
}

/**
 * One lap's replay: map, transport, readout and charts, all driven by one clock. Render it with
 * key={lap_id} (and the compared lap's ID) so each pairing gets a fresh clock; nothing plays
 * until the user presses play. With a compared lap the clock runs to the end of the longer lap,
 * and the car that finishes first waits at its last sample.
 */
export function ReplayWorkspace({
  replay,
  geometry,
  trackName,
  compare = null,
}: ReplayWorkspaceProps) {
  const prepared = useMemo(() => prepareReplay(replay.points), [replay.points]);
  const preparedCompare = useMemo(
    () => (compare === null ? null : prepareReplay(compare.points)),
    [compare],
  );
  const racing = compare !== null && preparedCompare !== null && preparedCompare.length > 0;
  const duration = racing
    ? Math.max(prepared.duration, preparedCompare.duration)
    : prepared.duration;
  const { clock, snapshot } = useReplayPlayback(duration);
  usePlaybackShortcuts(clock);

  const description =
    `Map of ${trackName} with the driven line of ${replay.driver_label}` +
    (replay.lap_time_s === null
      ? ", a lap that did not finish"
      : `, lap time ${formatLapTime(replay.lap_time_s)}`) +
    (racing ? `, compared with ${compareLabel(replay, compare)}` : "") +
    `. Top speed ${formatSpeed(prepared.maxSpeed)} metres per second. The readout below the controls gives the car's current values.`;

  if (prepared.length === 0) {
    return (
      <Notice tone="warn">
        This replay has no usable samples: every sample had a missing value or a time that did not
        increase.
      </Notice>
    );
  }

  return (
    <div className="grid gap-4">
      <MapFrame>
        <ReplayCanvas
          geometry={geometry}
          replay={prepared}
          clock={clock}
          description={description}
          compare={racing ? preparedCompare : null}
          {...(racing && { compareLabel: compareLabel(replay, compare) })}
        />
      </MapFrame>
      <ReplayControls clock={clock} snapshot={snapshot} />
      <p role="status" className="sr-only">
        {snapshot.playing
          ? `Playing at ${snapshot.rate} times speed`
          : snapshot.ended
            ? "Replay ended"
            : "Paused"}
      </p>
      <TelemetryReadout clock={clock} replay={prepared} />
      {prepared.dropped > 0 && (
        <Notice tone="warn">
          {prepared.dropped} sample{prepared.dropped === 1 ? " was" : "s were"} skipped because a
          value was missing or time did not increase.
        </Notice>
      )}
      {geometry === null && (
        <Notice>
          The server has no centreline for this track (TRACKS_DIR/{replay.track_id}/centreline.csv),
          so only the driven line is drawn.
        </Notice>
      )}
      <p className="hidden text-xs text-faint sm:block">
        Keys: <Kbd>Space</Kbd> play or pause, <Kbd>←</Kbd> <Kbd>→</Kbd> seek 1 s (<Kbd>Shift</Kbd>{" "}
        for 5 s), <Kbd>Home</Kbd> restart.
      </p>
      <Suspense fallback={<Skeleton className="h-[360px]" />}>
        <TelemetryCharts clock={clock} replay={prepared} />
      </Suspense>
    </div>
  );
}

/** The map's frame: fixed height on desktop, 4:3 on narrow screens. */
export function MapFrame({ children }: { children: ReactNode }) {
  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-panel sm:aspect-auto sm:h-[clamp(320px,58vh,620px)]">
      {children}
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-[2px] border border-line-strong bg-panel px-1 py-px font-mono text-[10.5px] text-muted">
      {children}
    </kbd>
  );
}
