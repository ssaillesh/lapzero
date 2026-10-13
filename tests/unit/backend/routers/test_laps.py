"""Unit tests for backend/app/routers/laps.py: POST /api/laps."""

import json
import math
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from backend.app.config import Settings
from tests.conftest import human_driver_json, lap_telemetry_document

LAP_ID = "01J9ZK8V3Q7M2T6C1R4X5N8B0A"


def _post(client: TestClient, document: dict[str, Any]) -> Response:
    # json.dumps writes NaN and Infinity literals, which FastAPI's JSON parser accepts.
    return client.post(
        "/api/laps", content=json.dumps(document), headers={"Content-Type": "application/json"}
    )


def _assert_nothing_stored(client: TestClient, settings: Settings) -> None:
    assert client.get("/api/tracks").json() == []
    assert not settings.laps_dir.exists()


def test_valid_lap_is_stored_and_described(
    client: TestClient, upload_lap: Callable[..., Response], api_settings: Settings
) -> None:
    response = upload_lap()

    assert response.status_code == 201
    body = response.json()
    assert body["lap_id"] == LAP_ID
    assert (body["driver_type"], body["driver_label"]) == ("model", "AI v7")
    assert (body["status"], body["lap_time_s"]) == ("completed", 35.97)
    assert body["created_at"] == "2026-10-20T14:32:05Z"
    assert (api_settings.laps_dir / "track_02" / f"{LAP_ID}.json").is_file()


def test_human_lap_is_stored(upload_lap: Callable[..., Response]) -> None:
    response = upload_lap(driver=human_driver_json("sam"))

    assert response.status_code == 201
    assert (response.json()["driver_label"], response.json()["human_input_device"]) == (
        "sam",
        "keyboard",
    )


def _drop_result(document: dict[str, Any]) -> None:
    del document["result"]


def _add_field(document: dict[str, Any]) -> None:
    document["comment"] = "fast lap"


def _steer_out_of_range(document: dict[str, Any]) -> None:
    document["frames"]["steer"][3] = 1.5


def _nan_position(document: dict[str, Any]) -> None:
    document["frames"]["x"][3] = math.nan


def _infinite_speed(document: dict[str, Any]) -> None:
    document["result"]["max_speed_mps"] = math.inf


def _short_column(document: dict[str, Any]) -> None:
    document["frames"]["y"].pop()


def _model_without_version(document: dict[str, Any]) -> None:
    document["driver"]["model_version"] = None


def _lap_time_on_a_crash(document: dict[str, Any]) -> None:
    document["result"]["status"] = "crashed"


def _bad_lap_id(document: dict[str, Any]) -> None:
    document["lap_id"] = "lap-1"


@pytest.mark.parametrize(
    "corrupt",
    [
        _drop_result,
        _add_field,
        _steer_out_of_range,
        _nan_position,
        _infinite_speed,
        _short_column,
        _model_without_version,
        _lap_time_on_a_crash,
        _bad_lap_id,
    ],
)
def test_lap_breaking_the_telemetry_schema_is_rejected(
    client: TestClient, api_settings: Settings, corrupt: Callable[[dict[str, Any]], None]
) -> None:
    document = lap_telemetry_document()
    corrupt(document)

    response = _post(client, document)

    assert response.status_code == 422
    _assert_nothing_stored(client, api_settings)


def test_body_that_is_not_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/laps", content="{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_lap_from_another_contract_major_version_is_rejected(
    client: TestClient, upload_lap: Callable[..., Response], api_settings: Settings
) -> None:
    response = upload_lap(contract_version="2.0.0")

    assert response.status_code == 422
    assert "major versions differ" in response.json()["detail"]
    _assert_nothing_stored(client, api_settings)


def test_lap_whose_frame_times_do_not_increase_is_rejected(
    client: TestClient, api_settings: Settings
) -> None:
    document = lap_telemetry_document()
    document["frames"]["t"][5] = document["frames"]["t"][4]

    response = _post(client, document)

    assert response.status_code == 422
    assert "frames.t[5]" in response.json()["detail"]
    _assert_nothing_stored(client, api_settings)


def test_lap_uploaded_twice_is_a_conflict(upload_lap: Callable[..., Response]) -> None:
    assert upload_lap().status_code == 201

    response = upload_lap()

    assert response.status_code == 409
    assert LAP_ID in response.json()["detail"]
