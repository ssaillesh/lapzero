import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { App } from "../App";
import { TRACKS, champion, leaderboard } from "../test/fixtures";
import { renderWithProviders } from "../test/render";
import { LeaderboardView } from "./LeaderboardView";

function mockApi() {
  vi.spyOn(api, "listTracks").mockResolvedValue(TRACKS);
  vi.spyOn(api, "getChampion").mockResolvedValue(champion());
  return vi
    .spyOn(api, "getLeaderboard")
    .mockImplementation((trackId) => Promise.resolve(leaderboard(trackId)));
}

const params = { trackId: "spielberg", driverType: null, limit: 25 } as const;

function bodyRows() {
  const table = screen.getByRole("table");
  return within(table).getAllByRole("row").slice(1);
}

describe("LeaderboardView", () => {
  it("shows skeletons while loading", () => {
    vi.spyOn(api, "listTracks").mockReturnValue(new Promise(() => undefined));
    vi.spyOn(api, "getChampion").mockReturnValue(new Promise(() => undefined));
    renderWithProviders(<LeaderboardView params={params} />);
    expect(screen.getAllByRole("status", { name: "Loading" }).length).toBeGreaterThan(0);
  });

  it("ranks the laps and highlights the champion from the registry, not a fixed version", async () => {
    mockApi();
    renderWithProviders(<LeaderboardView params={params} />);
    const rows = await waitFor(bodyRows);
    expect(rows).toHaveLength(3);
    expect(within(rows[0]!).getByText("AI v8")).toBeInTheDocument();
    expect(within(rows[0]!).getByText("Champion")).toBeInTheDocument();
    expect(within(rows[1]!).queryByText("Champion")).not.toBeInTheDocument();
    expect(within(rows[0]!).getByText("45.98")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("+0.80")).toBeInTheDocument();
  });

  it("moves the highlight when the champion alias points at another version", async () => {
    mockApi();
    vi.spyOn(api, "getChampion").mockResolvedValue(champion({ version: "5" }));
    renderWithProviders(<LeaderboardView params={params} />);
    const rows = await waitFor(bodyRows);
    expect(within(rows[1]!).getByText("Champion")).toBeInTheDocument();
    expect(within(rows[0]!).queryByText("Champion")).not.toBeInTheDocument();
  });

  it("sorts by a column and reflects it in aria-sort", async () => {
    mockApi();
    const user = userEvent.setup();
    renderWithProviders(<LeaderboardView params={params} />);
    await waitFor(bodyRows);
    await user.click(screen.getByRole("button", { name: /Lap time/ }));
    await user.click(screen.getByRole("button", { name: /Lap time/ }));
    expect(screen.getByRole("columnheader", { name: /Lap time/ })).toHaveAttribute(
      "aria-sort",
      "descending",
    );
    expect(within(bodyRows()[0]!).getByText("Mara Okafor")).toBeInTheDocument();
  });

  it("filters the returned rows and offers to clear an empty result", async () => {
    mockApi();
    const user = userEvent.setup();
    renderWithProviders(<LeaderboardView params={params} />);
    await waitFor(bodyRows);
    await user.type(screen.getByLabelText("Filter shown rows"), "v5");
    expect(bodyRows()).toHaveLength(1);
    await user.clear(screen.getByLabelText("Filter shown rows"));
    await user.type(screen.getByLabelText("Filter shown rows"), "nobody");
    expect(screen.getByText(/Nothing matches/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filter" }));
    expect(bodyRows()).toHaveLength(3);
  });

  it("asks the API for the chosen driver type and writes it to the URL", async () => {
    const getLeaderboard = mockApi();
    window.history.replaceState(null, "", "/leaderboard?track=spielberg");
    const user = userEvent.setup();
    renderWithProviders(<App />);
    await waitFor(bodyRows);
    await user.click(screen.getByLabelText("Model"));
    await waitFor(() => {
      expect(getLeaderboard).toHaveBeenLastCalledWith(
        "spielberg",
        { driverType: "model", limit: 25 },
        expect.anything(),
      );
    });
    expect(window.location.search).toBe("?track=spielberg&driver=model");
  });

  it("opens a lap's replay from its row", async () => {
    mockApi();
    vi.spyOn(api, "getTrack").mockReturnValue(new Promise(() => undefined));
    vi.spyOn(api, "getReplay").mockReturnValue(new Promise(() => undefined));
    window.history.replaceState(null, "", "/leaderboard?track=spielberg");
    const user = userEvent.setup();
    renderWithProviders(<App />);
    const rows = await waitFor(bodyRows);
    await user.click(within(rows[2]!).getByText("Mara Okafor"));
    expect(window.location.pathname).toBe("/tracks/spielberg/replays/01M4R0P8W0WS12C7HSENZ9628E");
  });

  it("says when a track has no completed laps", async () => {
    mockApi().mockResolvedValue(leaderboard("spielberg", []));
    renderWithProviders(<LeaderboardView params={params} />);
    expect(await screen.findByText("No completed laps on spielberg")).toBeInTheDocument();
  });

  it("explains an unreachable API and retries on request", async () => {
    const listTracks = vi
      .spyOn(api, "listTracks")
      .mockRejectedValueOnce(
        new ApiError({ kind: "network", url: "http://localhost:8000/api/tracks" }),
      )
      .mockResolvedValueOnce(TRACKS);
    vi.spyOn(api, "getChampion").mockResolvedValue(champion());
    vi.spyOn(api, "getLeaderboard").mockResolvedValue(leaderboard());
    const user = userEvent.setup();
    renderWithProviders(<LeaderboardView params={params} />);
    expect(await screen.findByText("API unreachable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(bodyRows);
    expect(listTracks).toHaveBeenCalledTimes(2);
  });

  it("keeps ranking when the registry is down, and says so", async () => {
    mockApi();
    vi.spyOn(api, "getChampion").mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 503, detail: "model registry failed" }),
    );
    renderWithProviders(<LeaderboardView params={params} />);
    await waitFor(bodyRows);
    expect(await screen.findByText(/Registry unavailable/)).toBeInTheDocument();
    expect(screen.queryByText("Champion", { selector: "span" })).not.toBeInTheDocument();
  });

  it("shows an empty state when no track has laps", async () => {
    vi.spyOn(api, "listTracks").mockResolvedValue([]);
    vi.spyOn(api, "getChampion").mockResolvedValue(champion());
    renderWithProviders(<LeaderboardView params={{ ...params, trackId: null }} />);
    expect(await screen.findByText("No tracks yet")).toBeInTheDocument();
  });
});
