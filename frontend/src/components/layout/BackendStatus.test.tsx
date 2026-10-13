import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import type { HealthResponse } from "../../api/types";
import { renderWithProviders } from "../../test/render";
import { BackendStatus } from "./BackendStatus";

describe("BackendStatus", () => {
  it("shows Checking until the health check answers, then Connected", async () => {
    let answer: (value: HealthResponse) => void = () => undefined;
    const client = { health: () => new Promise<HealthResponse>((resolve) => (answer = resolve)) };
    renderWithProviders(<BackendStatus />, client);
    expect(screen.getByText("Checking API")).toBeInTheDocument();
    expect(screen.queryByText("API connected")).not.toBeInTheDocument();
    answer({ status: "ok", contract_version: "1.1.0" });
    expect(await screen.findByText("API connected")).toBeInTheDocument();
  });

  it("shows Unreachable when the API cannot be reached, and checks again on click", async () => {
    const health = vi
      .fn<() => Promise<HealthResponse>>()
      .mockRejectedValueOnce(
        new ApiError({ kind: "network", url: "http://localhost:8000/api/health" }),
      )
      .mockResolvedValueOnce({ status: "ok", contract_version: "1.1.0" });
    renderWithProviders(<BackendStatus />, { health });
    const button = await screen.findByRole("button", { name: /API unreachable/ });
    button.click();
    expect(await screen.findByText("API connected")).toBeInTheDocument();
    expect(health).toHaveBeenCalledTimes(2);
  });
});
