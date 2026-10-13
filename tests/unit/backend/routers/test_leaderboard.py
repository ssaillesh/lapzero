"""Unit tests for backend/app/routers/leaderboard.py: GET /api/leaderboard/{track_id}."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from tests.conftest import fixed_ulid, human_driver_json, model_driver_json


@pytest.fixture
def three_drivers(upload_lap: Callable[..., Response]) -> None:
    upload_lap(lap_id=fixed_ulid(1), driver=model_driver_json("7"), lap_time_s=35.5)
    upload_lap(lap_id=fixed_ulid(2), driver=human_driver_json("sam"), lap_time_s=36.0)
    upload_lap(lap_id=fixed_ulid(3), driver=human_driver_json("sam"), lap_time_s=37.0)
    upload_lap(lap_id=fixed_ulid(4), driver=human_driver_json("alex"), lap_time_s=38.0)


@pytest.mark.usefixtures("three_drivers")
def test_drivers_are_ranked_by_their_fastest_lap(client: TestClient) -> None:
    response = client.get("/api/leaderboard/track_02")

    assert response.status_code == 200
    body = response.json()
    assert (body["track_id"], body["driver_type"]) == ("track_02", None)
    assert [(e["rank"], e["driver_label"], e["lap_time_s"]) for e in body["entries"]] == [
        (1, "AI v7", 35.5),
        (2, "sam", 36.0),
        (3, "alex", 38.0),
    ]
    assert body["entries"][1]["lap_id"] == fixed_ulid(2)


@pytest.mark.usefixtures("three_drivers")
def test_leaderboard_can_rank_humans_only(client: TestClient) -> None:
    body = client.get("/api/leaderboard/track_02", params={"driver_type": "human"}).json()

    assert body["driver_type"] == "human"
    assert [(e["rank"], e["driver_label"]) for e in body["entries"]] == [(1, "sam"), (2, "alex")]


@pytest.mark.usefixtures("three_drivers")
def test_limit_caps_the_entries(client: TestClient) -> None:
    body = client.get("/api/leaderboard/track_02", params={"limit": 1}).json()

    assert [e["driver_label"] for e in body["entries"]] == ["AI v7"]


def test_unknown_track_is_not_found(client: TestClient) -> None:
    response = client.get("/api/leaderboard/track_02")

    assert response.status_code == 404
    assert response.json()["detail"] == "track track_02 not found"


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 101}, {"driver_type": "robot"}, {"limit": "ten"}]
)
def test_invalid_query_parameters_are_rejected(
    client: TestClient, params: dict[str, object]
) -> None:
    assert client.get("/api/leaderboard/track_02", params=params).status_code == 422
