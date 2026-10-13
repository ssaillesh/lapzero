/**
 * Load one API resource per key, cancel it when the key changes or the component unmounts, and
 * optionally keep successful results in a cache shared across views, so returning to a view
 * (or a track whose geometry was already fetched) does not refetch.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { ApiError, isAbort } from "../api/errors";

export interface ResourceCache<T> {
  read(key: string): { value: T } | undefined;
  write(key: string, value: T): void;
  evict(key: string): void;
  clear(): void;
}

/** A cache of successful results by key; failures are never cached. */
export function createResourceCache<T>(): ResourceCache<T> {
  const entries = new Map<string, { value: T }>();
  return {
    read: (key) => entries.get(key),
    write: (key, value) => {
      entries.set(key, { value });
    },
    evict: (key) => {
      entries.delete(key);
    },
    clear: () => {
      entries.clear();
    },
  };
}

export type ResourceState<T> =
  /** No key: nothing to load yet. */
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: ApiError | Error };

export type Resource<T> = ResourceState<T> & {
  /** Drop any cached value for the current key and load it again. */
  reload: () => void;
};

type Settled<T> = { key: string; generation: number } & (
  { ok: true; data: T } | { ok: false; error: ApiError | Error }
);

/**
 * @param cache Shared cache, or null to always load on mount and key change.
 * @param key Identifies what `load` fetches; null loads nothing.
 * @param load Fetches the resource; must honour the abort signal. Only the latest function is
 *   used, so it may be an inline closure.
 */
export function useResource<T>(
  cache: ResourceCache<T> | null,
  key: string | null,
  load: (signal: AbortSignal) => Promise<T>,
): Resource<T> {
  const [generation, setGeneration] = useState(0);
  const [settled, setSettled] = useState<Settled<T> | null>(null);
  const loadRef = useRef(load);

  useLayoutEffect(() => {
    loadRef.current = load;
  });

  useEffect(() => {
    if (key === null || cache?.read(key) !== undefined) return;
    const controller = new AbortController();
    loadRef.current(controller.signal).then(
      (data) => {
        if (controller.signal.aborted) return;
        cache?.write(key, data);
        setSettled({ key, generation, ok: true, data });
      },
      (error: unknown) => {
        if (controller.signal.aborted || isAbort(error)) return;
        const failure = error instanceof Error ? error : new Error(String(error));
        if (!(failure instanceof ApiError)) console.error(`loading ${key} failed`, failure);
        setSettled({ key, generation, ok: false, error: failure });
      },
    );
    return () => {
      controller.abort();
    };
  }, [cache, key, generation]);

  const reload = useCallback(() => {
    if (key !== null) cache?.evict(key);
    setGeneration((value) => value + 1);
  }, [cache, key]);

  if (key === null) return { status: "idle", reload };
  const cached = cache?.read(key);
  if (cached !== undefined) return { status: "success", data: cached.value, reload };
  if (settled !== null && settled.key === key && settled.generation === generation) {
    return settled.ok
      ? { status: "success", data: settled.data, reload }
      : { status: "error", error: settled.error, reload };
  }
  return { status: "loading", reload };
}
