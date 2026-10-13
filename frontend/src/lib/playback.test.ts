import { describe, expect, it } from "vitest";

import type { FrameScheduler } from "./playback";
import { PlaybackClock } from "./playback";

/** A scheduler driven by the test: advance() moves time and runs one animation frame. */
function manualScheduler() {
  let now = 1000;
  let next = 1;
  const pending = new Map<number, () => void>();
  const scheduler: FrameScheduler = {
    now: () => now,
    request: (callback) => {
      const handle = next++;
      pending.set(handle, callback);
      return handle;
    },
    cancel: (handle) => {
      pending.delete(handle);
    },
  };
  return {
    scheduler,
    pendingFrames: () => pending.size,
    /** Move the wall clock by `ms` and run the frames that were waiting. */
    advance(ms: number) {
      now += ms;
      const callbacks = [...pending.values()];
      pending.clear();
      for (const callback of callbacks) callback();
    },
  };
}

describe("PlaybackClock", () => {
  it("advances by wall-clock time, not by frame count", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    frames.advance(250);
    expect(clock.getTime()).toBeCloseTo(0.25);
    frames.advance(1000); // one long frame still moves a full second
    expect(clock.getTime()).toBeCloseTo(1.25);
  });

  it("scales elapsed time by the playback rate and re-anchors on a rate change", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.setRate(2);
    clock.play();
    frames.advance(500);
    expect(clock.getTime()).toBeCloseTo(1);
    clock.setRate(0.25);
    frames.advance(400);
    expect(clock.getTime()).toBeCloseTo(1.1);
    expect(clock.getSnapshot().rate).toBe(0.25);
  });

  it("stops scheduling frames when paused", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    frames.advance(100);
    clock.pause();
    expect(frames.pendingFrames()).toBe(0);
    frames.advance(5000);
    expect(clock.getTime()).toBeCloseTo(0.1);
    expect(clock.getSnapshot().playing).toBe(false);
  });

  it("seeks while playing and continues from the new time", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    frames.advance(100);
    clock.seek(6);
    frames.advance(500);
    expect(clock.getTime()).toBeCloseTo(6.5);
  });

  it("clamps seeks to the replay and treats NaN as the start", () => {
    const clock = new PlaybackClock(10, manualScheduler().scheduler);
    clock.seek(25);
    expect(clock.getTime()).toBe(10);
    clock.seek(-3);
    expect(clock.getTime()).toBe(0);
    clock.seek(Number.NaN);
    expect(clock.getTime()).toBe(0);
  });

  it("pauses at the end and marks the replay ended; play then restarts from 0", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(1, frames.scheduler);
    clock.play();
    frames.advance(1500);
    expect(clock.getTime()).toBe(1);
    expect(clock.getSnapshot()).toEqual({ playing: false, rate: 1, ended: true });
    clock.play();
    expect(clock.getTime()).toBe(0);
    expect(clock.getSnapshot().ended).toBe(false);
  });

  it("restart returns to 0 and keeps playing", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    frames.advance(3000);
    clock.restart();
    expect(clock.getTime()).toBe(0);
    expect(clock.getSnapshot().playing).toBe(true);
  });

  it("cannot play a zero-length replay", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(0, frames.scheduler);
    expect(clock.canPlay).toBe(false);
    clock.play();
    expect(clock.getSnapshot().playing).toBe(false);
    expect(frames.pendingFrames()).toBe(0);
  });

  it("never runs two animation loops, however often play is called", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    clock.play();
    clock.toggle();
    clock.toggle();
    expect(frames.pendingFrames()).toBe(1);
  });

  it("notifies frame listeners on subscribe, each frame and each seek", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    const seen: number[] = [];
    const unsubscribe = clock.subscribeFrame((t) => seen.push(Math.round(t * 1000) / 1000));
    clock.seek(2);
    clock.play();
    frames.advance(100);
    unsubscribe();
    frames.advance(100);
    expect(seen).toEqual([0, 2, 2, 2.1]);
  });

  it("notifies state listeners only when play state, rate or ended change", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    let changes = 0;
    clock.subscribeState(() => changes++);
    const before = clock.getSnapshot();
    clock.seek(3);
    expect(changes).toBe(0);
    expect(clock.getSnapshot()).toBe(before); // stable for useSyncExternalStore
    clock.play();
    clock.setRate(2);
    clock.pause();
    expect(changes).toBe(3);
  });

  it("stop() cancels the loop but leaves the clock usable (StrictMode remount)", () => {
    const frames = manualScheduler();
    const clock = new PlaybackClock(10, frames.scheduler);
    clock.play();
    clock.stop();
    expect(frames.pendingFrames()).toBe(0);
    clock.play();
    frames.advance(200);
    expect(clock.getTime()).toBeCloseTo(0.2);
  });
});
