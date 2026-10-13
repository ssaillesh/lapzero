import { describe, expect, it, vi } from "vitest";

import { CONTRACT_EXAMPLE_REPLAY, jsonResponse, leaderboard } from "../test/fixtures";
import { ApiClient, DEFAULT_API_BASE_URL, resolveApiBaseUrl } from "./client";
import { ApiError } from "./errors";
import type { ApiFailure } from "./errors";

function clientWith(fetchImpl: (input: string, init?: RequestInit) => Promise<Response>) {
  const spy = vi.fn(fetchImpl);
  return { client: new ApiClient({ baseUrl: "http://api.test/api", fetch: spy }), spy };
}

async function failureOf(promise: Promise<unknown>): Promise<ApiFailure> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof ApiError) return error.failure;
    throw error;
  }
  throw new Error("expected the request to fail");
}

describe("resolveApiBaseUrl", () => {
  it("defaults to the local backend's /api root", () => {
    expect(resolveApiBaseUrl(undefined)).toBe(DEFAULT_API_BASE_URL);
    expect(resolveApiBaseUrl("  ")).toBe("http://localhost:8000/api");
  });

  it("drops trailing slashes without doubling /api", () => {
    expect(resolveApiBaseUrl("http://host:8000/api/")).toBe("http://host:8000/api");
    expect(resolveApiBaseUrl("http://host:8000/api///")).toBe("http://host:8000/api");
  });

  it("adds /api to a bare origin, where the backend mounts its routers", () => {
    expect(resolveApiBaseUrl("http://host:8000")).toBe("http://host:8000/api");
    expect(resolveApiBaseUrl("https://lapzero.example/")).toBe("https://lapzero.example/api");
  });

  it("keeps an explicit path, such as a reverse proxy prefix", () => {
    expect(resolveApiBaseUrl("https://example.org/lapzero/api")).toBe(
      "https://example.org/lapzero/api",
    );
    expect(resolveApiBaseUrl("/api/")).toBe("/api");
    expect(resolveApiBaseUrl("/")).toBe("/api");
  });

  it("rejects a value that is neither an http(s) URL nor an absolute path", () => {
    expect(() => resolveApiBaseUrl("localhost:8000")).toThrow(/VITE_API_BASE_URL/); // scheme-less
    expect(() => resolveApiBaseUrl("not a url")).toThrow(/VITE_API_BASE_URL/);
    expect(() => resolveApiBaseUrl("ftp://host/api")).toThrow(/VITE_API_BASE_URL/);
  });
});

describe("ApiClient", () => {
  it("requests health under the /api prefix and decodes it", async () => {
    const { client, spy } = clientWith(() =>
      Promise.resolve(jsonResponse({ status: "ok", contract_version: "1.1.0" })),
    );
    await expect(client.health()).resolves.toEqual({ status: "ok", contract_version: "1.1.0" });
    expect(spy.mock.calls[0]?.[0]).toBe("http://api.test/api/health");
  });

  it("sends the leaderboard filters as query parameters and encodes the track ID", async () => {
    const { client, spy } = clientWith(() =>
      Promise.resolve(jsonResponse(leaderboard("my track"))),
    );
    await client.getLeaderboard("my track", { driverType: "model", limit: 25 });
    expect(spy.mock.calls[0]?.[0]).toBe(
      "http://api.test/api/leaderboard/my%20track?limit=25&driver_type=model",
    );
    await client.getLeaderboard("my track", { driverType: null, limit: 10 });
    expect(spy.mock.calls[1]?.[0]).toBe("http://api.test/api/leaderboard/my%20track?limit=10");
  });

  it("decodes a contract example replay", async () => {
    const { client } = clientWith(() => Promise.resolve(jsonResponse(CONTRACT_EXAMPLE_REPLAY)));
    const replay = await client.getReplay(CONTRACT_EXAMPLE_REPLAY.lap_id);
    expect(replay.points.v).toEqual([0, 1.46, 2.81]);
    expect(replay.lap_time_s).toBe(35.97);
  });

  it("reports an HTTP error with FastAPI's string detail", async () => {
    const { client } = clientWith(() =>
      Promise.resolve(jsonResponse({ detail: "lapzero-policy has no champion alias" }, 404)),
    );
    expect(await failureOf(client.getChampion())).toEqual({
      kind: "http",
      url: "http://api.test/api/models/champion",
      status: 404,
      detail: "lapzero-policy has no champion alias",
    });
  });

  it("joins FastAPI's validation issues into one detail", async () => {
    const body = {
      detail: [
        {
          loc: ["path", "lap_id"],
          msg: "String should match pattern",
          type: "string_pattern_mismatch",
        },
      ],
    };
    const { client } = clientWith(() => Promise.resolve(jsonResponse(body, 422)));
    const failure = await failureOf(client.getReplay("bad"));
    expect(failure).toMatchObject({
      kind: "http",
      status: 422,
      detail: "lap_id: String should match pattern",
    });
  });

  it("reports an HTTP error without a JSON body", async () => {
    const { client } = clientWith(() =>
      Promise.resolve(new Response("Bad gateway", { status: 502 })),
    );
    expect(await failureOf(client.listTracks())).toMatchObject({
      kind: "http",
      status: 502,
      detail: null,
    });
  });

  it("reports a network failure when fetch rejects", async () => {
    const { client } = clientWith(() => Promise.reject(new TypeError("Failed to fetch")));
    expect(await failureOf(client.listTracks())).toEqual({
      kind: "network",
      url: "http://api.test/api/tracks",
    });
  });

  it("reports a cancelled request as aborted, not as a failure", async () => {
    const controller = new AbortController();
    const { client } = clientWith(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    );
    const pending = client.listTracks({ signal: controller.signal });
    controller.abort();
    expect(await failureOf(pending)).toEqual({
      kind: "aborted",
      url: "http://api.test/api/tracks",
    });
  });

  it("reports a 2xx body that is not JSON as malformed", async () => {
    const { client } = clientWith(() =>
      Promise.resolve(new Response("<html>proxy</html>", { status: 200 })),
    );
    expect(await failureOf(client.listTracks())).toEqual({
      kind: "malformed-json",
      url: "http://api.test/api/tracks",
      status: 200,
    });
  });

  it("reports a response of the wrong shape with the path of the bad field", async () => {
    const body = [
      {
        track_id: "spielberg",
        name: "spielberg",
        length_m: "long",
        created_at: "2026-10-11T18:02:41Z",
      },
    ];
    const { client } = clientWith(() => Promise.resolve(jsonResponse(body)));
    expect(await failureOf(client.listTracks())).toMatchObject({
      kind: "invalid-response",
      path: "$[0].length_m",
    });
  });
});
