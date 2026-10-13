import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { ReactNode } from "react";

import { api } from "../../api/client";
import type { ApiClient } from "../../api/client";
import { BackendHealthContext } from "../../hooks/useBackendHealth";
import { HealthMonitor } from "../../lib/health";

/** Runs one health monitor for the whole app; start and stop follow the provider's lifetime. */
export function HealthProvider({
  children,
  client = api,
  intervalMs,
}: {
  children: ReactNode;
  client?: Pick<ApiClient, "health">;
  intervalMs?: number;
}) {
  const [monitor] = useState(() => new HealthMonitor(client, intervalMs));
  useEffect(() => {
    monitor.start();
    return monitor.stop;
  }, [monitor]);
  const snapshot = useSyncExternalStore(
    monitor.subscribe,
    monitor.getSnapshot,
    monitor.getSnapshot,
  );
  const value = useMemo(() => ({ snapshot, recheck: monitor.check }), [snapshot, monitor]);
  return <BackendHealthContext.Provider value={value}>{children}</BackendHealthContext.Provider>;
}
