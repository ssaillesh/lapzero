/**
 * Champion identity on leaderboards.
 *
 * A lap's model_version is the registry version of the model that drove it (lapzero.io.telemetry
 * DriverInfo), and GET /api/models/champion reports the version the registry's champion alias
 * points at. Leaderboard entries carry no MLflow run ID, so the version is the only key the API
 * exposes to match on.
 */

import type { ChampionResponse, DriverType } from "../api/types";

interface DriverIdentity {
  driver_type: DriverType;
  model_version: string | null;
}

/** True if this driver is the current champion model version. */
export function isChampion(entry: DriverIdentity, champion: ChampionResponse | null): boolean {
  return (
    champion !== null &&
    entry.driver_type === "model" &&
    entry.model_version !== null &&
    entry.model_version === champion.version
  );
}
