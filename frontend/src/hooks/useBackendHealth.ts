import { createContext, useContext } from "react";

import type { HealthSnapshot } from "../lib/health";

export interface BackendHealth {
  snapshot: HealthSnapshot;
  /** Check now (ignored while a check is in flight). */
  recheck: () => void;
}

export const BackendHealthContext = createContext<BackendHealth | null>(null);

/** The app-wide backend health, from the HealthProvider at the root. */
export function useBackendHealth(): BackendHealth {
  const health = useContext(BackendHealthContext);
  if (health === null) throw new Error("useBackendHealth must be used inside <HealthProvider>");
  return health;
}
