"""Unit tests for backend/app/schemas/laps.py."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.models.domain import Lap
from backend.app.schemas.laps import LapResponse, LapSubmitRequest
from lapzero.io.telemetry import DriverType, LapStatus, LapTelemetry
from tests.conftest import lap_telemetry_document


def _lap(created_at: datetime) -> Lap:
    return Lap(
        lap_id="01J9ZK8V3Q7M2T6C1R4X5N8B0A",
        episode_id="01J9ZK8T9E0Y3H2P6W1D7F4G5S",
        lap_index=2,
        track_id="track_02",
        driver_type=DriverType.MODEL,
        driver_name="lapzero-ppo",
        model_version="7",
        mlflow_run_id="3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
        human_input_device=None,
        status=LapStatus.COMPLETED,
        lap_time_s=35.97,
        collision=False,
        off_track_events=1,
        max_speed_mps=7.84,
        mean_speed_mps=6.12,
        sample_count=900,
        contract_version="1.1.0",
        telemetry_path="track_02/01J9ZK8V3Q7M2T6C1R4X5N8B0A.json",
        created_at=created_at,
    )


def test_upload_body_is_the_contract_lap_telemetry_document() -> None:
    assert LapSubmitRequest is LapTelemetry
    LapSubmitRequest.model_validate(lap_telemetry_document())


def test_lap_response_carries_the_driver_label_and_a_utc_timestamp() -> None:
    response = LapResponse.model_validate(_lap(datetime(2026, 10, 20, 14, 32, 5, tzinfo=UTC)))

    body = response.model_dump(mode="json")
    assert body["driver_label"] == "AI v7"
    assert body["created_at"] == "2026-10-20T14:32:05Z"
    assert body["lap_index"] == 2
    assert "telemetry_path" not in body


def test_lap_response_refuses_a_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="created_at"):
        LapResponse.model_validate(_lap(datetime(2026, 10, 20, 14, 32, 5)))
