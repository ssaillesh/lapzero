import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";

import type { FrameListener, FrameScheduler, PlaybackSnapshot } from "../lib/playback";
import { PlaybackClock, browserScheduler } from "../lib/playback";

/**
 * A playback clock for one replay. Key the calling component on the lap so a new replay gets a
 * new clock. The animation loop stops on unmount.
 *
 * @param duration Replay length, s.
 */
export function useReplayPlayback(
  duration: number,
  scheduler: FrameScheduler = browserScheduler,
): { clock: PlaybackClock; snapshot: PlaybackSnapshot } {
  const [clock] = useState(() => new PlaybackClock(duration, scheduler));
  useEffect(() => clock.stop, [clock]);
  const snapshot = useSyncExternalStore(clock.subscribeState, clock.getSnapshot, clock.getSnapshot);
  return { clock, snapshot };
}

/**
 * Call `onFrame` with the replay time every animation frame while playing and after each seek,
 * outside React rendering. Use it to write canvas pixels or DOM text directly.
 */
export function useClockFrame(clock: PlaybackClock, onFrame: FrameListener): void {
  const callback = useRef(onFrame);
  useLayoutEffect(() => {
    callback.current = onFrame;
  });
  useEffect(
    () =>
      clock.subscribeFrame((time) => {
        callback.current(time);
      }),
    [clock],
  );
}
