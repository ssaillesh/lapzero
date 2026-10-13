import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/errors";
import type { HealthResponse } from "../api/types";
import type { Timers } from "./health";
import { HealthMonitor } from "./health";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function manualTimers() {
  let now = 0;
  const pending = new Map<number, () => void>();
  let next = 1;
  const timers: Timers = {
    setTimeout: (callback) => {
      pending.set(next, callback);
      return next++;
    },
    clearTimeout: (handle) => {
      pending.delete(handle);
    },
    now: () => now,
  };
  return {
    timers,
    pending: () => pending.size,
    fire(ms: number) {
      now += ms;
      const callbacks = [...pending.values()];
      pending.clear();
      for (const callback of callbacks) callback();
    },
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("HealthMonitor", () => {
  it("is checking until the first check succeeds, then connected with the contract version", async () => {
    const response = deferred<HealthResponse>();
    const client = { health: vi.fn(() => response.promise) };
    const clock = manualTimers();
    const monitor = new HealthMonitor(client, 30_000, clock.timers);
    monitor.start();
    expect(monitor.getSnapshot()).toMatchObject({
      status: "checking",
      inFlight: true,
      lastChecked: null,
    });
    response.resolve({ status: "ok", contract_version: "1.1.0" });
    await flush();
    expect(monitor.getSnapshot()).toMatchObject({
      status: "connected",
      contractVersion: "1.1.0",
      inFlight: false,
    });
    expect(clock.pending()).toBe(1); // next check scheduled
  });

  it("reports unavailable on a network failure and error on a bad response", async () => {
    const clock = manualTimers();
    const client = { health: vi.fn<() => Promise<HealthResponse>>() };
    client.health.mockRejectedValueOnce(new ApiError({ kind: "network", url: "x" }));
    const monitor = new HealthMonitor(client, 30_000, clock.timers);
    monitor.start();
    await flush();
    expect(monitor.getSnapshot().status).toBe("unavailable");

    client.health.mockRejectedValueOnce(
      new ApiError({ kind: "http", url: "x", status: 500, detail: null }),
    );
    clock.fire(30_000);
    await flush();
    expect(monitor.getSnapshot().status).toBe("error");
  });

  it("never runs two checks at once", () => {
    const client = { health: vi.fn(() => new Promise<HealthResponse>(() => undefined)) };
    const monitor = new HealthMonitor(client, 30_000, manualTimers().timers);
    monitor.start();
    monitor.check();
    monitor.check();
    expect(client.health).toHaveBeenCalledTimes(1);
  });

  it("aborts the check in flight on stop and ignores its late answer", async () => {
    const response = deferred<HealthResponse>();
    let signal: AbortSignal | undefined;
    const client = {
      health: vi.fn((options?: { signal?: AbortSignal }) => {
        signal = options?.signal;
        return response.promise;
      }),
    };
    const clock = manualTimers();
    const monitor = new HealthMonitor(client, 30_000, clock.timers);
    monitor.start();
    monitor.stop();
    expect(signal?.aborted).toBe(true);
    response.resolve({ status: "ok", contract_version: "1.1.0" });
    await flush();
    expect(monitor.getSnapshot().status).toBe("checking");
    expect(clock.pending()).toBe(0);
  });

  it("starts again after stop, as React StrictMode does", async () => {
    const client = {
      health: vi.fn(() =>
        Promise.resolve<HealthResponse>({ status: "ok", contract_version: "1.1.0" }),
      ),
    };
    const monitor = new HealthMonitor(client, 30_000, manualTimers().timers);
    monitor.start();
    monitor.stop();
    monitor.start();
    await flush();
    expect(client.health).toHaveBeenCalledTimes(2);
    expect(monitor.getSnapshot().status).toBe("connected");
  });
});
