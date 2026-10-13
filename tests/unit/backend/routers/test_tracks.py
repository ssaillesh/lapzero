"""Unit tests for backend/app/routers/tracks.py: GET /api/tracks and /api/tracks/{track_id}."""

from collections.abc import Callable

from fastapi.testclient import TestClient
from httpx import Response

from backend.app.config import Settings
from tests.conftest import fixed_ulid


def _write_centreline(settings: Settings, track_id: str, text: str) -> None:
    directory = settings.tracks_dir / track_id
    directory.mkdir(parents=True)
    (directory / "centreline.csv").write_text(text)


def test_no_tracks_before_any_lap_is_uploaded(client: TestClient) -> None:
    response = client.get("/api/tracks")

    assert response.status_code == 200
    assert response.json() == []


def test_tracks_with_uploaded_laps_are_listed_by_id(
    client: TestClient, upload_lap: Callable[..., Response]
) -> None:
    upload_lap(lap_id=fixed_ulid(1), track_id="zandvoort")
    upload_lap(lap_id=fixed_ulid(2), track_id="oschersleben")
    upload_lap(lap_id=fixed_ulid(3), track_id="zandvoort")

    tracks = client.get("/api/tracks").json()

    assert [track["track_id"] for track in tracks] == ["oschersleben", "zandvoort"]
    assert set(tracks[0]) == {"track_id", "name", "length_m", "created_at"}


def test_track_detail_includes_centreline_and_edges(
    client: TestClient, upload_lap: Callable[..., Response], api_settings: Settings
) -> None:
    upload_lap(track_id="oschersleben")
    _write_centreline(
        api_settings,
        "oschersleben",
        "# x_m, y_m, w_tr_right_m, w_tr_left_m\n0,0,1,1\n4,0,1,1\n4,3,1,1\n",
    )

    response = client.get("/api/tracks/oschersleben")

    assert response.status_code == 200
    body = response.json()
    assert (body["track_id"], body["length_m"], body["metadata_json"]) == ("oschersleben", 82.4, {})
    assert body["geometry"]["centreline"] == {"x": [0.0, 4.0, 4.0], "y": [0.0, 0.0, 3.0]}
    assert len(body["geometry"]["left"]["x"]) == len(body["geometry"]["right"]["x"]) == 3


def test_track_without_a_centreline_file_has_null_geometry(
    client: TestClient, upload_lap: Callable[..., Response]
) -> None:
    upload_lap(track_id="oschersleben")

    assert client.get("/api/tracks/oschersleben").json()["geometry"] is None


def test_track_with_an_invalid_centreline_file_is_a_server_error(
    client: TestClient, upload_lap: Callable[..., Response], api_settings: Settings
) -> None:
    upload_lap(track_id="oschersleben")
    _write_centreline(api_settings, "oschersleben", "0,0\n1,0\n")

    response = client.get("/api/tracks/oschersleben")

    assert response.status_code == 500
    assert response.json()["detail"] == "centreline of track oschersleben is invalid"


def test_unknown_track_is_not_found(client: TestClient) -> None:
    response = client.get("/api/tracks/oschersleben")

    assert response.status_code == 404
    assert response.json()["detail"] == "track oschersleben not found"


def test_track_id_with_path_characters_never_reaches_the_tracks_directory(
    client: TestClient,
) -> None:
    assert client.get("/api/tracks/bad.id").status_code == 422  # fails TRACK_ID_PATTERN
    assert client.get("/api/tracks/..%2Fsecrets").status_code == 404  # matches no route
