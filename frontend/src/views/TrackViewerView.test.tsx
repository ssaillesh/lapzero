import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { App } from "../App";
import {
  LAP_P1,
  LAP_P2,
  LAP_P3,
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

  describe("comparing two laps", () => {
    /** The champion's lap (P1) normally; P3 is a human, Mara Okafor, 59.60. */
    function mockRace(compareTrack = "spielberg") {
      const getReplay = mockApi();
      getReplay.mockImplementation((lapId) =>
        Promise.resolve(
          lapId === LAP_P3
            ? {
                ...straightReplay(LAP_P3, compareTrack),
                driver_label: "Mara Okafor",
                lap_time_s: 59.6,
              }
            : straightReplay(lapId),
        ),
      );
      return getReplay;
    }

    it("races a second lap picked from the list and puts it in the URL", async () => {
      const getReplay = mockRace();
      const user = userEvent.setup();
      open(`/tracks/spielberg/replays/${LAP_P1}`);
      const picker = await screen.findByLabelText("Compare with");
      const options = within(picker)
        .getAllByRole("option")
        .map((o) => o.textContent);
      expect(options).toEqual(["No comparison", "P2 AI v5, 46.78", "P3 Mara Okafor, 59.60"]);

      await user.selectOptions(picker, LAP_P3);

      await waitFor(() => {
        expect(window.location.search).toBe(`?vs=${LAP_P3}`);
      });
      expect(window.location.pathname).toBe(`/tracks/spielberg/replays/${LAP_P1}`);
      expect(await screen.findByText("Mara Okafor, 59.60 (+13.62 s)")).toBeInTheDocument();
      expect(getReplay).toHaveBeenCalledWith(LAP_P3, expect.anything());
    });

    it("opens a comparison from the URL and clears it on request", async () => {
      mockRace();
      const user = userEvent.setup();
      open(`/tracks/spielberg/replays/${LAP_P1}?vs=${LAP_P3}`);
      expect(await screen.findByText("Mara Okafor, 59.60 (+13.62 s)")).toBeInTheDocument();

      await user.selectOptions(screen.getByLabelText("Compare with"), "");

      await waitFor(() => {
        expect(window.location.search).toBe("");
      });
      expect(screen.queryByText("Mara Okafor, 59.60 (+13.62 s)")).not.toBeInTheDocument();
      expect(screen.getByRole("img", { name: /Map of spielberg/ })).not.toHaveAccessibleName(
        /compared with/,
      );
    });

    it("shows only the open lap when the compared one is on another track", async () => {
      mockRace("oschersleben");
      open(`/tracks/spielberg/replays/${LAP_P1}?vs=${LAP_P3}`);
      expect(await screen.findByText(/driven on oschersleben, not this track/)).toBeInTheDocument();
      expect(screen.queryByText("Mara Okafor, 59.60 (+13.62 s)")).not.toBeInTheDocument();
      expect(screen.getByRole("img", { name: /Map of spielberg/ })).not.toHaveAccessibleName(
        /compared with/,
      );
    });

    it("says when the lap to compare with does not exist", async () => {
      mockApi().mockImplementation((lapId) =>
        lapId === LAP_P3
          ? Promise.reject(new ApiError({ kind: "http", url: "u", status: 404, detail: null }))
          : Promise.resolve(straightReplay(lapId)),
      );
      open(`/tracks/spielberg/replays/${LAP_P1}?vs=${LAP_P3}`);
      expect(
        await screen.findByText(
          "The lap to compare with was not found, so only the open lap is shown.",
        ),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    });
  });
});
