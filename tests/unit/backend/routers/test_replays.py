"""Unit tests for backend/app/routers/replays.py: GET /api/replays/{lap_id}."""

from collections.abc import Callable

from fastapi.testclient import TestClient
from httpx import Response

from lapzero.contract import REPLAY_HZ
from lapzero.io.replay import Replay

LAP_ID = "01J9ZK8V3Q7M2T6C1R4X5N8B0A"


def test_replay_of_an_uploaded_lap_is_the_contract_section_6_document(
    client: TestClient, upload_lap: Callable[..., Response]
) -> None:
    upload_lap(frames=26)

    response = client.get(f"/api/replays/{LAP_ID}")

    assert response.status_code == 200
    body = response.json()
    assert list(body) == [
        "schema",
        "schema_version",
        "lap_id",
        "track_id",
        "driver_label",
        "lap_time_s",
        "hz",
        "points",
    ]
    assert (body["schema"], body["hz"], body["driver_label"]) == (
        "lapzero.replay",
        REPLAY_HZ,
        "AI v7",
    )
    assert body["points"]["t"] == [round(k / 10, 1) for k in range(11)]  # 1.0 s of frames
    Replay.model_validate(body)


def test_replay_of_an_unknown_lap_is_not_found(client: TestClient) -> None:
    response = client.get(f"/api/replays/{LAP_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == f"lap {LAP_ID} not found"


def test_lap_id_that_is_not_a_ulid_is_rejected(client: TestClient) -> None:
    assert client.get("/api/replays/lap-1").status_code == 422
