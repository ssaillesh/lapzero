"""Unit tests for backend/app/services/laps.py: lap ingestion and replay lookup."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.domain import Lap, LapReplay, Track
from backend.app.services.laps import LapExistsError, get_replay, ingest_lap
from lapzero.contract import ContractViolationError
from lapzero.io.replay import build_replay
from lapzero.io.telemetry import (
    DriverType,
    InputDevice,
    LapStatus,
    LapTelemetry,
    read_lap_telemetry,
)
from tests.conftest import fixed_ulid, human_driver_json, lap_telemetry_document

LAP_ID = "01J9ZK8V3Q7M2T6C1R4X5N8B0A"


def _telemetry(**kwargs: Any) -> LapTelemetry:
    return LapTelemetry.model_validate(lap_telemetry_document(**kwargs))


def _count(session: Session, table: type[Lap] | type[LapReplay] | type[Track]) -> int:
    return session.scalar(select(func.count()).select_from(table)) or 0


def _assert_nothing_stored(session: Session, laps_dir: Path) -> None:
    assert (_count(session, Lap), _count(session, LapReplay), _count(session, Track)) == (0, 0, 0)
    assert not laps_dir.exists()


def test_ingested_lap_row_holds_the_telemetry_driver_and_result(
    session: Session, tmp_path: Path
) -> None:
    lap = ingest_lap(session, _telemetry(), tmp_path)

    assert session.get(Lap, LAP_ID) is lap
    assert (lap.track_id, lap.lap_index, lap.episode_id) == (
        "track_02",
        1,
        "01J9ZK8T9E0Y3H2P6W1D7F4G5S",
    )
    assert (lap.driver_type, lap.driver_name, lap.model_version) == (
        DriverType.MODEL,
        "lapzero-ppo",
        "7",
    )
    assert (lap.status, lap.lap_time_s, lap.sample_count) == (LapStatus.COMPLETED, 35.97, 26)
    assert lap.contract_version == "1.1.0"
    assert lap.created_at == datetime(2026, 10, 20, 14, 32, 5, tzinfo=UTC)


def test_telemetry_file_is_stored_under_laps_dir_by_track_and_lap_id(
    session: Session, tmp_path: Path
) -> None:
    telemetry = _telemetry()

    lap = ingest_lap(session, telemetry, tmp_path)

    assert lap.telemetry_path == f"track_02/{LAP_ID}.json"
    stored = tmp_path / lap.telemetry_path
    assert read_lap_telemetry(stored) == telemetry
    assert json.loads(stored.read_text())["schema"] == "lapzero.lap_telemetry"


def test_replay_is_generated_on_ingest(session: Session, tmp_path: Path) -> None:
    telemetry = _telemetry()

    ingest_lap(session, telemetry, tmp_path)

    assert get_replay(session, LAP_ID) == build_replay(telemetry)


def test_human_lap_is_stored_without_model_fields(session: Session, tmp_path: Path) -> None:
    lap = ingest_lap(session, _telemetry(driver=human_driver_json("sam")), tmp_path)

    assert (lap.driver_type, lap.driver_name) == (DriverType.HUMAN, "sam")
    assert (lap.model_version, lap.mlflow_run_id) == (None, None)
    assert lap.human_input_device is InputDevice.KEYBOARD


def test_crashed_lap_is_stored_without_a_lap_time(session: Session, tmp_path: Path) -> None:
    lap = ingest_lap(session, _telemetry(lap_time_s=None), tmp_path)

    assert (lap.status, lap.lap_time_s, lap.collision) == (LapStatus.CRASHED, None, True)
    replay = get_replay(session, LAP_ID)
    assert replay is not None and replay.lap_time_s is None


def test_second_lap_on_a_track_reuses_the_track(session: Session, tmp_path: Path) -> None:
    ingest_lap(session, _telemetry(lap_id=fixed_ulid(1)), tmp_path)
    ingest_lap(session, _telemetry(lap_id=fixed_ulid(2)), tmp_path)

    assert (_count(session, Track), _count(session, Lap)) == (1, 2)


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.9.3"])
def test_lap_with_the_same_contract_major_version_is_accepted(
    session: Session, tmp_path: Path, version: str
) -> None:
    assert ingest_lap(session, _telemetry(contract_version=version), tmp_path).contract_version == (
        version
    )


@pytest.mark.parametrize("version", ["0.9.0", "2.0.0"])
def test_lap_with_another_contract_major_version_is_rejected(
    session: Session, tmp_path: Path, version: str
) -> None:
    with pytest.raises(ContractViolationError, match="major versions differ"):
        ingest_lap(session, _telemetry(contract_version=version), tmp_path / "laps")

    _assert_nothing_stored(session, tmp_path / "laps")


def test_lap_whose_frame_times_do_not_increase_is_rejected(
    session: Session, tmp_path: Path
) -> None:
    document = lap_telemetry_document()
    document["frames"]["t"][5] = document["frames"]["t"][4]

    with pytest.raises(ContractViolationError, match=r"frames\.t\[5\]"):
        ingest_lap(session, LapTelemetry.model_validate(document), tmp_path / "laps")

    _assert_nothing_stored(session, tmp_path / "laps")


def test_lap_uploaded_twice_is_rejected_and_the_first_copy_kept(
    session: Session, tmp_path: Path
) -> None:
    ingest_lap(session, _telemetry(lap_time_s=35.97), tmp_path)

    with pytest.raises(LapExistsError, match=LAP_ID):
        ingest_lap(session, _telemetry(lap_time_s=30.0), tmp_path)

    assert _count(session, Lap) == 1
    assert read_lap_telemetry(tmp_path / f"track_02/{LAP_ID}.json").result.lap_time_s == 35.97


def test_failed_telemetry_write_rolls_the_lap_and_track_back(
    session: Session, tmp_path: Path
) -> None:
    laps_dir = tmp_path / "laps"
    laps_dir.write_text("a file where the directory should be")

    with pytest.raises(OSError):
        ingest_lap(session, _telemetry(), laps_dir)

    assert (_count(session, Lap), _count(session, LapReplay), _count(session, Track)) == (0, 0, 0)


def test_replay_of_an_unknown_lap_is_none(session: Session) -> None:
    assert get_replay(session, LAP_ID) is None
