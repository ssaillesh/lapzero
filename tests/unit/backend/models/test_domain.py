"""Unit tests for backend/app/models/domain.py: the tracks, laps and replays tables."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models.domain import Lap, LapReplay, Track
from lapzero.io.telemetry import DriverType, InputDevice, LapStatus

LAP_ID = "01J9ZK8V3Q7M2T6C1R4X5N8B0A"


def _lap(**overrides: Any) -> Lap:
    """A completed human lap on track "t" with field overrides."""
    fields: dict[str, Any] = {
        "lap_id": LAP_ID,
        "episode_id": "01J9ZK8T9E0Y3H2P6W1D7F4G5S",
        "lap_index": 1,
        "track_id": "t",
        "driver_type": DriverType.HUMAN,
        "driver_name": "sam",
        "model_version": None,
        "mlflow_run_id": None,
        "human_input_device": InputDevice.KEYBOARD,
        "status": LapStatus.COMPLETED,
        "lap_time_s": 35.97,
        "collision": False,
        "off_track_events": 0,
        "max_speed_mps": 7.84,
        "mean_speed_mps": 6.12,
        "sample_count": 900,
        "contract_version": "1.1.0",
        "telemetry_path": f"t/{LAP_ID}.json",
        "created_at": datetime(2026, 10, 20, 14, 32, 5, tzinfo=UTC),
    }
    return Lap(**(fields | overrides))


@pytest.fixture
def track(session: Session) -> Track:
    track = Track(track_id="t", name="Test track", length_m=82.4)
    session.add(track)
    session.flush()
    return track


def test_new_track_gets_empty_metadata_and_a_utc_creation_time(track: Track) -> None:
    assert track.metadata_json == {}
    assert track.created_at.tzinfo is UTC
    assert (datetime.now(UTC) - track.created_at).total_seconds() < 60


@pytest.mark.usefixtures("track")
def test_lap_enums_load_as_contract_enum_members(session: Session) -> None:
    lap = _lap()
    session.add(lap)
    session.flush()
    session.expire(lap)

    assert lap.driver_type is DriverType.HUMAN
    assert lap.human_input_device is InputDevice.KEYBOARD
    assert lap.status is LapStatus.COMPLETED


@pytest.mark.usefixtures("track")
def test_enums_are_stored_as_their_contract_values(session: Session) -> None:
    session.add(_lap())
    session.flush()

    row = session.execute(text("SELECT driver_type, human_input_device, status FROM laps")).one()

    assert tuple(row) == ("human", "keyboard", "completed")


@pytest.mark.usefixtures("track")
def test_database_rejects_a_value_outside_an_enum(session: Session) -> None:
    session.add(_lap())
    session.flush()

    with pytest.raises(IntegrityError, match="CHECK"), session.begin_nested():
        session.execute(text("UPDATE laps SET status = 'abandoned'"))


def test_lap_on_an_unknown_track_is_rejected(session: Session) -> None:
    session.add(_lap(track_id="nowhere"))

    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        session.flush()


@pytest.mark.usefixtures("track")
def test_deleting_a_lap_deletes_its_replay(session: Session) -> None:
    session.add(_lap())
    session.flush()
    session.add(LapReplay(lap_id=LAP_ID, replay_json={"schema": "lapzero.replay"}))
    session.flush()

    session.execute(text("DELETE FROM laps"))

    assert session.execute(text("SELECT count(*) FROM replays")).scalar_one() == 0


@pytest.mark.parametrize(
    ("driver_type", "name", "version", "label"),
    [(DriverType.MODEL, "lapzero-ppo", "9", "AI v9"), (DriverType.HUMAN, "sam", None, "sam")],
)
def test_lap_driver_label_names_models_by_version_and_humans_by_name(
    driver_type: DriverType, name: str, version: str | None, label: str
) -> None:
    lap = _lap(driver_type=driver_type, driver_name=name, model_version=version)

    assert lap.driver_label == label
