"""Unit tests for backend/app/services/leaderboard.py: each driver's best lap on a track."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from backend.app.services.laps import ingest_lap
from backend.app.services.leaderboard import best_laps
from lapzero.io.telemetry import DriverType, LapTelemetry
from tests.conftest import fixed_ulid, human_driver_json, lap_telemetry_document, model_driver_json


@pytest.fixture
def add_lap(session: Session, tmp_path: Path) -> Any:
    """Ingest lap number `n` (its ULID is fixed_ulid(n)) with lap_telemetry_document kwargs."""

    def _add(n: int, **kwargs: Any) -> None:
        document = lap_telemetry_document(fixed_ulid(n), **kwargs)
        ingest_lap(session, LapTelemetry.model_validate(document), tmp_path)

    return _add


def _ids(session: Session, track_id: str = "track_02", **kwargs: Any) -> list[str]:
    return [lap.lap_id for lap in best_laps(session, track_id, **kwargs)]


def test_each_driver_ranks_once_with_their_fastest_lap(session: Session, add_lap: Any) -> None:
    add_lap(1, driver=human_driver_json("sam"), lap_time_s=40.0)
    add_lap(2, driver=human_driver_json("sam"), lap_time_s=38.0)
    add_lap(3, driver=human_driver_json("alex"), lap_time_s=39.0)

    assert _ids(session) == [fixed_ulid(2), fixed_ulid(3)]


def test_every_model_version_is_its_own_driver(session: Session, add_lap: Any) -> None:
    add_lap(1, driver=model_driver_json("7"), lap_time_s=37.0)
    add_lap(2, driver=model_driver_json("8"), lap_time_s=36.0)
    add_lap(3, driver=model_driver_json("8"), lap_time_s=36.5)

    assert _ids(session) == [fixed_ulid(2), fixed_ulid(1)]


def test_incomplete_laps_and_other_tracks_do_not_rank(session: Session, add_lap: Any) -> None:
    add_lap(1, lap_time_s=None)
    add_lap(2, track_id="other", lap_time_s=30.0)
    add_lap(3, driver=model_driver_json("8"), lap_time_s=40.0)

    assert _ids(session) == [fixed_ulid(3)]
    assert _ids(session, "other") == [fixed_ulid(2)]


@pytest.mark.parametrize(
    ("driver_type", "expected"),
    [(DriverType.HUMAN, [2]), (DriverType.MODEL, [1]), (None, [1, 2])],
)
def test_leaderboard_can_rank_only_humans_or_only_models(
    session: Session, add_lap: Any, driver_type: DriverType | None, expected: list[int]
) -> None:
    add_lap(1, driver=model_driver_json("7"), lap_time_s=35.0)
    add_lap(2, driver=human_driver_json("sam"), lap_time_s=36.0)

    assert _ids(session, driver_type=driver_type) == [fixed_ulid(n) for n in expected]


def test_limit_caps_the_number_of_drivers(session: Session, add_lap: Any) -> None:
    for n in range(1, 6):
        add_lap(n, driver=human_driver_json(f"driver{n}"), lap_time_s=30.0 + n)

    assert _ids(session, limit=3) == [fixed_ulid(1), fixed_ulid(2), fixed_ulid(3)]


def test_equal_lap_times_rank_the_earlier_lap_first(session: Session, add_lap: Any) -> None:
    add_lap(1, driver=human_driver_json("late"), created_at="2026-10-21T09:00:00Z")
    add_lap(2, driver=human_driver_json("early"), created_at="2026-10-20T09:00:00Z")

    assert _ids(session) == [fixed_ulid(2), fixed_ulid(1)]


def test_track_without_completed_laps_has_an_empty_leaderboard(session: Session) -> None:
    assert best_laps(session, "track_02") == []
