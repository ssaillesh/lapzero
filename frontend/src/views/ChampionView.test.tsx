import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import type { HealthResponse } from "../api/types";
import { TRACKS, champion, leaderboard } from "../test/fixtures";
import { renderWithProviders } from "../test/render";
import { ChampionView } from "./ChampionView";

function mockApi() {
  vi.spyOn(api, "listTracks").mockResolvedValue(TRACKS);
  vi.spyOn(api, "getLeaderboard").mockImplementation((trackId) =>
    Promise.resolve(trackId === "spielberg" ? leaderboard("spielberg") : leaderboard(trackId, [])),
  );
  return vi.spyOn(api, "getChampion").mockResolvedValue(champion());
}

describe("ChampionView", () => {
  it("shows the champion's registry identity, run and contract compatibility", async () => {
    mockApi();
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText("v8")).toBeInTheDocument();
    expect(screen.getByText("3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d")).toBeInTheDocument();
    expect(await screen.findByText("Same major version (1) as the API")).toBeInTheDocument();
  });

  it("warns when the run's contract major differs from the API's", async () => {
    mockApi().mockResolvedValue(champion({ contract_version: "2.0.0" }));
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText(/Major 2 differs from the API's 1/)).toBeInTheDocument();
  });

  it("reports the gate evidence recorded in the version tags", async () => {
    mockApi();
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText("Promoted by the release gate")).toBeInTheDocument();
    const gate = screen.getByRole("region", { name: "Release gate record" });
    expect(within(gate).getByText("oschersleben")).toBeInTheDocument();
    expect(within(gate).getByText("v5")).toBeInTheDocument();
    expect(within(gate).getByText("01M4TGPRVQY5F71QAQKBSR5RKQ")).toBeInTheDocument();
  });

  it("says plainly when a champion has no gate record", async () => {
    mockApi().mockResolvedValue(champion({ tags: { algorithm: "bc" } }));
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText("No gate record on this version")).toBeInTheDocument();
    expect(screen.getByText("algorithm")).toBeInTheDocument();
  });

  it("ranks the champion on each track from the leaderboards", async () => {
    mockApi();
    renderWithProviders(<ChampionView />);
    const record = await screen.findByRole("region", { name: "Champion on the leaderboards" });
    const spielberg = await within(record).findByRole("row", { name: /spielberg/ });
    expect(within(spielberg).getByText("P1")).toBeInTheDocument();
    expect(await within(record).findByText("No completed lap by v8 uploaded")).toBeInTheDocument();
  });

  it("explains a missing champion without inventing one", async () => {
    mockApi().mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 404, detail: "no champion alias" }),
    );
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText("No champion registered")).toBeInTheDocument();
    expect(screen.queryByText(/^v\d+$/)).not.toBeInTheDocument();
  });

  it("separates an unreadable registry from a missing champion", async () => {
    mockApi().mockRejectedValue(
      new ApiError({ kind: "http", url: "u", status: 503, detail: "model registry failed" }),
    );
    renderWithProviders(<ChampionView />);
    expect(await screen.findByText("Model registry unavailable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("does not claim a contract match before the API answered", async () => {
    mockApi();
    const pending = { health: () => new Promise<HealthResponse>(() => undefined) };
    renderWithProviders(<ChampionView />, pending);
    expect(await screen.findByText(/Waiting for the API's contract version/)).toBeInTheDocument();
  });
});
