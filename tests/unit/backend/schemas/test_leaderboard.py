"""Unit tests for backend/app/schemas/leaderboard.py."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.models.domain import Lap
from backend.app.schemas.leaderboard import LeaderboardEntry, LeaderboardResponse
from lapzero.io.telemetry import DriverType

CREATED = datetime(2026, 10, 20, 14, 32, 5, tzinfo=UTC)


def _lap(lap_time_s: float | None) -> Lap:
    return Lap(
        lap_id="01J9ZK8V3Q7M2T6C1R4X5N8B0A",
        driver_type=DriverType.HUMAN,
        driver_name="sam",
        model_version=None,
        lap_time_s=lap_time_s,
        created_at=CREATED,
    )


def test_entry_is_built_from_a_drivers_best_lap() -> None:
    entry = LeaderboardEntry.from_lap(3, _lap(35.97))

    assert entry.model_dump() == {
        "rank": 3,
        "lap_id": "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
        "driver_type": DriverType.HUMAN,
        "driver_name": "sam",
        "driver_label": "sam",
        "model_version": None,
        "lap_time_s": 35.97,
        "created_at": CREATED,
    }


def test_lap_without_a_time_cannot_rank() -> None:
    with pytest.raises(ValueError, match="only completed laps rank"):
        LeaderboardEntry.from_lap(1, _lap(None))


def test_ranks_start_at_one() -> None:
    with pytest.raises(ValidationError, match="rank"):
        LeaderboardEntry.from_lap(0, _lap(35.97))


def test_response_records_the_driver_type_filter() -> None:
    response = LeaderboardResponse(track_id="t", driver_type=DriverType.MODEL, entries=[])

    assert response.model_dump(mode="json") == {
        "track_id": "t",
        "driver_type": "model",
        "entries": [],
    }
