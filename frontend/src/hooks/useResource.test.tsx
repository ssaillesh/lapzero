import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/errors";
import { createResourceCache, useResource } from "./useResource";

describe("useResource", () => {
  it("is idle without a key and loads nothing", () => {
    const load = vi.fn(() => Promise.resolve(1));
    const { result } = renderHook(() => useResource(null, null, load));
    expect(result.current.status).toBe("idle");
    expect(load).not.toHaveBeenCalled();
  });

  it("goes from loading to success", async () => {
    const { result } = renderHook(() => useResource(null, "a", () => Promise.resolve(42)));
    expect(result.current.status).toBe("loading");
    await waitFor(() => {
      expect(result.current).toMatchObject({ status: "success", data: 42 });
    });
  });

  it("reports a failure and loads again on reload", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<number>>()
      .mockRejectedValueOnce(new ApiError({ kind: "network", url: "u" }))
      .mockResolvedValueOnce(7);
    const { result } = renderHook(() => useResource(null, "a", load));
    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });
    act(() => {
      result.current.reload();
    });
    expect(result.current.status).toBe("loading");
    await waitFor(() => {
      expect(result.current).toMatchObject({ status: "success", data: 7 });
    });
  });

  it("serves a cached value without loading, and reload bypasses the cache", async () => {
    const cache = createResourceCache<number>();
    cache.write("a", 1);
    const load = vi.fn(() => Promise.resolve(2));
    const { result } = renderHook(() => useResource(cache, "a", load));
    expect(result.current).toMatchObject({ status: "success", data: 1 });
    expect(load).not.toHaveBeenCalled();
    act(() => {
      result.current.reload();
    });
    await waitFor(() => {
      expect(result.current).toMatchObject({ status: "success", data: 2 });
    });
  });

  it("aborts the old request when the key changes and never shows its result", async () => {
    const signals: AbortSignal[] = [];
    let resolveFirst: (value: string) => void = () => undefined;
    const load = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return signals.length === 1
        ? new Promise<string>((resolve) => {
            resolveFirst = resolve;
          })
        : Promise.resolve("second");
    });
    const { result, rerender } = renderHook(({ key }) => useResource(null, key, load), {
      initialProps: { key: "first" },
    });
    rerender({ key: "second" });
    expect(signals[0]?.aborted).toBe(true);
    resolveFirst("first");
    await waitFor(() => {
      expect(result.current).toMatchObject({ status: "success", data: "second" });
    });
  });

  it("aborts its request on unmount", () => {
    let signal: AbortSignal | undefined;
    const { unmount } = renderHook(() =>
      useResource(null, "a", (s) => {
        signal = s;
        return new Promise<number>(() => undefined);
      }),
    );
    unmount();
    expect(signal?.aborted).toBe(true);
  });
});
