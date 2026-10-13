import { render } from "@testing-library/react";
import type { ReactElement } from "react";

import type { ApiClient } from "../api/client";
import type { HealthResponse } from "../api/types";
import { HealthProvider } from "../components/layout/HealthProvider";

/** A health client that reports a connected API with contract 1.1.0. */
export const connectedHealth: Pick<ApiClient, "health"> = {
  health: () => Promise.resolve<HealthResponse>({ status: "ok", contract_version: "1.1.0" }),
};

/** Render inside the providers every view expects. */
export function renderWithProviders(
  ui: ReactElement,
  health: Pick<ApiClient, "health"> = connectedHealth,
) {
  return render(<HealthProvider client={health}>{ui}</HealthProvider>);
}
