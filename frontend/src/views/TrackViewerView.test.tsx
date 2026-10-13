import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { App } from "../App";
import {
  LAP_P1,
  LAP_P2,
  TRACKS,
  champion,
  leaderboard,
  squareTrack,
  straightReplay,
} from "../test/fixtures";
import { renderWithProviders } from "../test/render";

function mockApi() {
  vi.spyOn(api, "listTracks").mockResolvedValue(TRACKS);
  vi.spyOn(api, "getChampion").mockResolvedValue(champion());
  vi.spyOn(api, "getLeaderboard").mockImplementation((trackId) =>
    Promise.resolve(leaderboard(trackId)),
  );
  vi.spyOn(api, "getTrack").mockImplementation((trackId) => Promise.resolve(squareTrack(trackId)));
  return vi
    .spyOn(api, "getReplay")
    .mockImplementation((lapId) => Promise.resolve(straightReplay(lapId)));
}

/** A track the dashboard hides (lib/tracks), as the API would list it. */
const WIDE = { ...TRACKS[1]!, track_id: "spielberg_wide", name: "spielberg_wide" };

function open(path: string) {
  window.history.replaceState(null, "", path);
  return renderWithProviders(<App />);
}

describe("Track Explorer", () => {
  it("opens the first track from /tracks and then its fastest lap, paused", async () => {
    mockApi().mockImplementation((lapId) => Promise.resolve(straightReplay(lapId, "oschersleben")));
    open("/tracks");
    await waitFor(() => {
      expect(window.location.pathname).toBe(`/tracks/oschersleben/replays/${LAP_P1}`);
    });
    expect(await screen.findByRole("button", { name: "Play" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Map of oschersleben/ })).toBeInTheDocument();
  });

  it("loads the replay named in the URL and marks the champion's lap", async () => {
    const getReplay = mockApi();
    open(`/tracks/spielberg/replays/${LAP_P1}`);
    expect(await screen.findByRole("button", { name: "Play" })).toBeInTheDocument();
    expect(getReplay).toHaveBeenCalledWith(LAP_P1, expect.anything());
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getAllByText("Champion").length).toBeGreaterThan(0);
  });

  it("marks a lap that did not finish", async () => {
    mockApi().mockResolvedValue({ ...straightReplay(LAP_P2), lap_time_s: null });
    open(`/tracks/spielberg/replays/${LAP_P2}`);
    expect(await screen.findByText("Did not finish")).toBeInTheDocument();
  });

  it("shows Lap not found for an unknown lap ID", async () => {
    mockApi().mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 404, detail: "lap x not found" }),
    );
    open("/tracks/spielberg/replays/01M4R0P48JA6JMPQ1WR91DCP3Z");
    expect(await screen.findByText("Lap not found")).toBeInTheDocument();
  });

  it("shows Track not found for an unknown track", async () => {
    mockApi();
    vi.spyOn(api, "getTrack").mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 404, detail: "track nowhere not found" }),
    );
    vi.spyOn(api, "getLeaderboard").mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 404, detail: null }),
    );
    open("/tracks/nowhere");
    expect(await screen.findByText(`The API has no track "nowhere"`)).toBeInTheDocument();
  });

  it("moves a replay to the track it was driven on", async () => {
    mockApi().mockImplementation((lapId) => Promise.resolve(straightReplay(lapId, "oschersleben")));
    open(`/tracks/spielberg/replays/${LAP_P1}`);
    await waitFor(() => {
      expect(window.location.pathname).toBe(`/tracks/oschersleben/replays/${LAP_P1}`);
    });
  });

  it("never lists a hidden track", async () => {
    mockApi();
    vi.spyOn(api, "listTracks").mockResolvedValue([...TRACKS, WIDE]);
    open(`/tracks/spielberg/replays/${LAP_P1}`);
    const select = await screen.findByLabelText("Track");
    await waitFor(() => {
      expect(
        within(select)
          .getAllByRole("option")
          .map((o) => o.textContent),
      ).toEqual(["oschersleben", "spielberg"]);
    });
    expect(screen.queryByRole("link", { name: /spielberg_wide/ })).not.toBeInTheDocument();
  });

  it("opens the first visible track instead of a hidden track named in the URL", async () => {
    mockApi().mockImplementation((lapId) => Promise.resolve(straightReplay(lapId, "oschersleben")));
    vi.spyOn(api, "listTracks").mockResolvedValue([WIDE, ...TRACKS]);
    const getTrack = vi.spyOn(api, "getTrack");
    open(`/tracks/spielberg_wide/replays/${LAP_P1}`);
    await waitFor(() => {
      expect(window.location.pathname).toBe(`/tracks/oschersleben/replays/${LAP_P1}`);
    });
    expect(getTrack).not.toHaveBeenCalledWith("spielberg_wide", expect.anything());
  });
});
