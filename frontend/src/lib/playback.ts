/**
 * Replay playback clock, independent of React.
 *
 * Time advances from the wall clock (anchor + elapsed x rate), not by counting frames, so a slow
 * or dropped frame never slows the replay down. Frame listeners run once per animation frame
 * while playing and once after every seek; they write to the canvas and the DOM directly, so the
 * React tree does not re-render 60 times a second. State listeners run only when playing, rate
 * or the ended flag change, and back a useSyncExternalStore.
 */

export interface FrameScheduler {
  /** Milliseconds, monotonic. */
  now(): number;
  request(callback: () => void): number;
  cancel(handle: number): void;
}

export const browserScheduler: FrameScheduler = {
  now: () => performance.now(),
  request: (callback) =>
    requestAnimationFrame(() => {
      callback();
    }),
  cancel: (handle) => {
    cancelAnimationFrame(handle);
  },
};

export const PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4] as const;
export type PlaybackRate = (typeof PLAYBACK_RATES)[number];

export interface PlaybackSnapshot {
  readonly playing: boolean;
  readonly rate: PlaybackRate;
  /** Paused at the end of the replay. */
  readonly ended: boolean;
}

export type FrameListener = (time: number) => void;

export class PlaybackClock {
  readonly duration: number;
  private readonly scheduler: FrameScheduler;
  private time = 0;
  private playing = false;
  private rate: PlaybackRate = 1;
  private anchorWall = 0;
  private anchorTime = 0;
  private handle: number | null = null;
  private snapshot: PlaybackSnapshot;
  private readonly frameListeners = new Set<FrameListener>();
  private readonly stateListeners = new Set<() => void>();

  /**
   * @param duration Replay length, s; a clock of duration 0 (one sample) cannot play.
   */
  constructor(duration: number, scheduler: FrameScheduler = browserScheduler) {
    this.duration = Number.isFinite(duration) && duration > 0 ? duration : 0;
    this.scheduler = scheduler;
    this.snapshot = { playing: false, rate: 1, ended: false };
  }

  readonly getTime = (): number => this.time;

  readonly getSnapshot = (): PlaybackSnapshot => this.snapshot;

  get canPlay(): boolean {
    return this.duration > 0;
  }

  readonly play = (): void => {
    if (this.playing || !this.canPlay) return;
    if (this.time >= this.duration) this.time = 0;
    this.playing = true;
    this.anchor();
    this.schedule();
    this.publishState();
    this.emitFrame();
  };

  readonly pause = (): void => {
    if (!this.playing) return;
    this.advance();
    this.playing = false;
    this.cancel();
    this.publishState();
    this.emitFrame();
  };

  readonly toggle = (): void => {
    if (this.playing) this.pause();
    else this.play();
  };

  /** Jump to `time` (s, clamped to the replay); playback continues from there if playing. */
  readonly seek = (time: number): void => {
    const target = Number.isFinite(time) ? Math.max(0, Math.min(this.duration, time)) : 0;
    this.time = target;
    if (this.playing) this.anchor();
    this.publishState();
    this.emitFrame();
  };

  readonly seekBy = (delta: number): void => {
    this.seek(this.time + delta);
  };

  /** Back to the start, keeping the play state. */
  readonly restart = (): void => {
    this.seek(0);
  };

  readonly setRate = (rate: PlaybackRate): void => {
    if (rate === this.rate) return;
    if (this.playing) this.advance();
    this.rate = rate;
    if (this.playing) this.anchor();
    this.publishState();
  };

  /** Run `listener` every frame while playing and after every seek; it is called once now. */
  readonly subscribeFrame = (listener: FrameListener): (() => void) => {
    this.frameListeners.add(listener);
    listener(this.time);
    return () => {
      this.frameListeners.delete(listener);
    };
  };

  readonly subscribeState = (listener: () => void): (() => void) => {
    this.stateListeners.add(listener);
    return () => {
      this.stateListeners.delete(listener);
    };
  };

  /** Stop the animation loop. The clock stays usable; React StrictMode remounts call play again. */
  readonly stop = (): void => {
    this.pause();
    this.cancel();
  };

  private readonly tick = (): void => {
    this.handle = null;
    if (!this.playing) return;
    this.advance();
    if (this.time >= this.duration) {
      this.time = this.duration;
      this.playing = false;
      this.publishState();
      this.emitFrame();
      return;
    }
    this.emitFrame();
    this.schedule();
  };

  private advance(): void {
    const elapsed = (this.scheduler.now() - this.anchorWall) / 1000;
    this.time = Math.min(this.duration, this.anchorTime + Math.max(0, elapsed) * this.rate);
  }

  private anchor(): void {
    this.anchorWall = this.scheduler.now();
    this.anchorTime = this.time;
  }

  private schedule(): void {
    if (this.handle === null) this.handle = this.scheduler.request(this.tick);
  }

  private cancel(): void {
    if (this.handle !== null) {
      this.scheduler.cancel(this.handle);
      this.handle = null;
    }
  }

  private publishState(): void {
    const ended = !this.playing && this.canPlay && this.time >= this.duration;
    const previous = this.snapshot;
    if (
      previous.playing === this.playing &&
      previous.rate === this.rate &&
      previous.ended === ended
    )
      return;
    this.snapshot = { playing: this.playing, rate: this.rate, ended };
    for (const listener of this.stateListeners) listener();
  }

  private emitFrame(): void {
    for (const listener of this.frameListeners) listener(this.time);
  }
}
