import { describe, expect, it } from "vitest";

import { CONTRACT_EXAMPLE_REPLAY, champion, leaderboard, squareTrack } from "../test/fixtures";
import {
  DecodeError,
  decodeChampion,
  decodeLeaderboard,
  decodeReplay,
  decodeTrackDetail,
} from "./decode";

function pathOfError(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    if (error instanceof DecodeError) return error.path;
    throw error;
  }
  throw new Error("expected a DecodeError");
}

describe("decodeReplay", () => {
  it("accepts the contract Section 6 example", () => {
    expect(decodeReplay(CONTRACT_EXAMPLE_REPLAY).points.t).toEqual([0, 0.1, 0.2]);
  });

  it("accepts a null lap time for a lap that did not finish", () => {
    expect(decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, lap_time_s: null }).lap_time_s).toBeNull();
  });

  it("rejects columns of different lengths", () => {
    const points = { ...CONTRACT_EXAMPLE_REPLAY.points, v: [0, 1] };
    expect(pathOfError(() => decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, points }))).toBe(
      "$.points",
    );
  });

  it("rejects a replay with no samples", () => {
    const points = { t: [], x: [], y: [], theta: [], v: [], steer: [] };
    expect(pathOfError(() => decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, points }))).toBe(
      "$.points.t",
    );
  });

  it("rejects a non-numeric sample and names its index", () => {
    const points = { ...CONTRACT_EXAMPLE_REPLAY.points, theta: [0, "north", 0] };
    expect(pathOfError(() => decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, points }))).toBe(
      "$.points.theta[1]",
    );
  });

  it("rejects another schema", () => {
    expect(
      pathOfError(() =>
        decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, schema: "lapzero.lap_telemetry" }),
      ),
    ).toBe("$.schema");
  });

  it("ignores fields a later minor contract version may add", () => {
    expect(() => decodeReplay({ ...CONTRACT_EXAMPLE_REPLAY, friction_mu: 0.6 })).not.toThrow();
  });
});

describe("decodeTrackDetail", () => {
  it("keeps the edges and the free-form metadata", () => {
    const track = {
      ...squareTrack(),
      metadata_json: { country: "AT", corners: 10, tags: ["fast"] },
    };
    const decoded = decodeTrackDetail(track);
    expect(decoded.geometry?.left?.x).toHaveLength(4);
    expect(decoded.metadata_json).toEqual({ country: "AT", corners: 10, tags: ["fast"] });
  });

  it("accepts a track without geometry and a centreline without edges", () => {
    expect(decodeTrackDetail({ ...squareTrack(), geometry: null }).geometry).toBeNull();
    const geometry = { centreline: { x: [0, 1, 0], y: [0, 0, 1] }, left: null, right: null };
    expect(decodeTrackDetail({ ...squareTrack(), geometry }).geometry?.left).toBeNull();
  });

  it("rejects a polyline whose columns differ in length", () => {
    const geometry = { centreline: { x: [0, 1, 0], y: [0, 0] }, left: null, right: null };
    expect(pathOfError(() => decodeTrackDetail({ ...squareTrack(), geometry }))).toBe(
      "$.geometry.centreline",
    );
  });
});

describe("decodeLeaderboard", () => {
  it("decodes driver types and nullable model versions", () => {
    const decoded = decodeLeaderboard(leaderboard());
    expect(decoded.entries.map((e) => [e.driver_type, e.model_version])).toEqual([
      ["model", "8"],
      ["model", "5"],
      ["human", null],
    ]);
  });

  it("rejects an unknown driver type", () => {
    const board = leaderboard();
    const entries = [{ ...board.entries[0], driver_type: "robot" }];
    expect(pathOfError(() => decodeLeaderboard({ ...board, entries }))).toBe(
      "$.entries[0].driver_type",
    );
  });
});

describe("decodeChampion", () => {
  it("decodes tags as strings and a null run ID", () => {
    const decoded = decodeChampion({ ...champion(), run_id: null });
    expect(decoded.run_id).toBeNull();
    expect(decoded.tags.gate_result).toBe("promote");
  });

  it("rejects a non-string tag value", () => {
    expect(pathOfError(() => decodeChampion({ ...champion(), tags: { gate_result: 1 } }))).toBe(
      "$.tags.gate_result",
    );
  });
});
