"""Lap ingestion (contract Section 6.1): telemetry file, laps row and replay, stored together."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.models.domain import Lap, LapReplay
from backend.app.services.tracks import ensure_track
from lapzero.contract import CONTRACT_VERSION, ContractViolationError
from lapzero.io.replay import Replay, build_replay
from lapzero.io.telemetry import LapTelemetry

logger = logging.getLogger(__name__)


class LapExistsError(Exception):
    """Raised when a lap with the same lap_id was already ingested."""


def ingest_lap(session: Session, telemetry: LapTelemetry, laps_dir: Path) -> Lap:
    """Store an uploaded lap: its telemetry file, its laps row and its replay.

    The telemetry is written to laps_dir/<track_id>/<lap_id>.json. A lap on a new track also
    registers the track. Nothing is stored unless everything is: on any failure the session is
    rolled back.

    Args:
        session: Open session; committed on success.
        telemetry: The validated upload.
        laps_dir: The LAPS_DIR setting.

    Returns:
        The new lap row.

    Raises:
        ContractViolationError: If the upload's contract_version major number differs from the
            backend's, or its frames.t does not strictly increase.
        LapExistsError: If the lap was already ingested.
        OSError: If the telemetry file cannot be written.
    """
    if _major(telemetry.contract_version) != _major(CONTRACT_VERSION):
        raise ContractViolationError(
            f"contract_version {telemetry.contract_version} does not match the backend's "
            f"{CONTRACT_VERSION}: major versions differ"
        )
    replay = build_replay(telemetry)
    if session.get(Lap, telemetry.lap_id) is not None:
        raise LapExistsError(f"lap {telemetry.lap_id} was already uploaded")
    relative_path = Path(telemetry.track.track_id) / f"{telemetry.lap_id}.json"
    try:
        ensure_track(session, telemetry.track)
        lap = _lap_row(telemetry, relative_path)
        session.add(lap)
        session.add(
            LapReplay(lap_id=lap.lap_id, replay_json=replay.model_dump(mode="json", by_alias=True))
        )
        session.flush()
        path = laps_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(telemetry.model_dump_json(by_alias=True))
        session.commit()
    except Exception:
        session.rollback()
        raise
    logger.info("ingested lap %s on %s (%s)", lap.lap_id, lap.track_id, lap.status)
    return lap


def get_replay(session: Session, lap_id: str) -> Replay | None:
    """Return the replay of a lap, or None if no lap with `lap_id` was ingested."""
    row = session.get(LapReplay, lap_id)
    return None if row is None else Replay.model_validate(row.replay_json)


def _lap_row(telemetry: LapTelemetry, telemetry_path: Path) -> Lap:
    """Return the laps row of an upload whose file is stored at `telemetry_path`."""
    driver, result = telemetry.driver, telemetry.result
    return Lap(
        lap_id=telemetry.lap_id,
        episode_id=telemetry.episode_id,
        lap_index=telemetry.lap_index,
        track_id=telemetry.track.track_id,
        driver_type=driver.type,
        driver_name=driver.name,
        model_version=driver.model_version,
        mlflow_run_id=driver.mlflow_run_id,
        human_input_device=driver.human_input_device,
        status=result.status,
        lap_time_s=result.lap_time_s,
        collision=result.collision,
        off_track_events=result.off_track_events,
        max_speed_mps=result.max_speed_mps,
        mean_speed_mps=result.mean_speed_mps,
        sample_count=result.sample_count,
        contract_version=telemetry.contract_version,
        telemetry_path=telemetry_path.as_posix(),
        created_at=telemetry.created_at,
    )


def _major(version: str) -> str:
    """Return the major number of a semantic version string."""
    return version.split(".")[0]
