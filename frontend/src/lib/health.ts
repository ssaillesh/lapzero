/**
 * Backend health monitor: polls GET /api/health, never with two requests in flight, pauses while
 * the tab is hidden, and reports "connected" only after a check actually succeeded.
 */

import type { ApiClient } from "../api/client";
import type { ApiFailure } from "../api/errors";
import { ApiError } from "../api/errors";

export type HealthStatus =
  /** No check has finished yet. */
  | "checking"
  | "connected"
  /** The API could not be reached or did not answer in time. */
  | "unavailable"
  /** The API answered, but not with a valid health response. */
  | "error";

export interface HealthSnapshot {
  readonly status: HealthStatus;
  /** Contract version from the last successful check. */
  readonly contractVersion: string | null;
  /** When the last check finished. */
  readonly lastChecked: Date | null;
  readonly failure: ApiFailure | null;
  /** A check is running now (also during later re-checks). */
  readonly inFlight: boolean;
}

export interface Timers {
  setTimeout(callback: () => void, ms: number): number;
  clearTimeout(handle: number): void;
  now(): number;
}

const browserTimers: Timers = {
  setTimeout: (callback, ms) => window.setTimeout(callback, ms),
  clearTimeout: (handle) => {
    window.clearTimeout(handle);
  },
  now: () => Date.now(),
};

export const HEALTH_INTERVAL_MS = 30_000;

export class HealthMonitor {
  private snapshot: HealthSnapshot = {
    status: "checking",
    contractVersion: null,
    lastChecked: null,
    failure: null,
    inFlight: false,
  };
  private readonly listeners = new Set<() => void>();
  private controller: AbortController | null = null;
  private timer: number | null = null;
  private running = false;
  private readonly client: Pick<ApiClient, "health">;
  private readonly intervalMs: number;
  private readonly timers: Timers;

  constructor(
    client: Pick<ApiClient, "health">,
    intervalMs = HEALTH_INTERVAL_MS,
    timers: Timers = browserTimers,
  ) {
    this.client = client;
    this.intervalMs = intervalMs;
    this.timers = timers;
  }

  readonly getSnapshot = (): HealthSnapshot => this.snapshot;

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** Check now and then every interval. Safe to call again after stop() (StrictMode remounts). */
  readonly start = (): void => {
    if (this.running) return;
    this.running = true;
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this.onVisibilityChange);
    }
    this.check();
  };

  /** Cancel the request in flight and the next scheduled check. */
  readonly stop = (): void => {
    this.running = false;
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.onVisibilityChange);
    }
    this.clearTimer();
    this.controller?.abort();
    this.controller = null;
    if (this.snapshot.inFlight) this.update({ inFlight: false });
  };

  /** Check now, unless a check is already in flight. */
  readonly check = (): void => {
    if (this.controller !== null) return;
    this.clearTimer();
    const controller = new AbortController();
    this.controller = controller;
    this.update({ inFlight: true });
    this.client.health({ signal: controller.signal }).then(
      (health) => {
        if (controller.signal.aborted) return;
        this.finish({
          status: "connected",
          contractVersion: health.contract_version,
          failure: null,
        });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        const failure: ApiFailure =
          error instanceof ApiError ? error.failure : { kind: "network", url: "health" };
        const unreachable = failure.kind === "network" || failure.kind === "timeout";
        this.finish({ status: unreachable ? "unavailable" : "error", failure });
      },
    );
  };

  private finish(
    result: Pick<HealthSnapshot, "status" | "failure"> & { contractVersion?: string },
  ): void {
    this.controller = null;
    this.update({
      status: result.status,
      failure: result.failure,
      contractVersion: result.contractVersion ?? this.snapshot.contractVersion,
      lastChecked: new Date(this.timers.now()),
      inFlight: false,
    });
    this.schedule();
  }

  private schedule(): void {
    this.clearTimer();
    if (!this.running) return;
    this.timer = this.timers.setTimeout(() => {
      this.timer = null;
      if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
      this.check();
    }, this.intervalMs);
  }

  private readonly onVisibilityChange = (): void => {
    if (document.visibilityState !== "visible" || !this.running) return;
    const last = this.snapshot.lastChecked?.getTime() ?? 0;
    if (this.timer === null || this.timers.now() - last >= this.intervalMs) this.check();
  };

  private clearTimer(): void {
    if (this.timer !== null) {
      this.timers.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private update(patch: Partial<HealthSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const listener of this.listeners) listener();
  }
}
