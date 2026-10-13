"""Unit tests for backend/app/schemas/tracks.py."""

from datetime import UTC, datetime

from backend.app.models.domain import Track
from backend.app.schemas.tracks import TrackResponse, TrackSummary
from lapzero.io.centreline import Polyline, TrackGeometry

CREATED = datetime(2026, 10, 20, 14, 32, 5, tzinfo=UTC)
GEOMETRY = TrackGeometry(
    centreline=Polyline(x=[0.0, 4.0, 4.0], y=[0.0, 0.0, 3.0]), left=None, right=None
)


def _track() -> Track:
    return Track(
        track_id="oschersleben",
        name="Oschersleben",
        length_m=82.4,
        metadata_json={"country": "DE"},
        created_at=CREATED,
    )


def test_summary_holds_only_the_listed_track_fields() -> None:
    summary = TrackSummary.model_validate(_track())

    assert summary.model_dump() == {
        "track_id": "oschersleben",
        "name": "Oschersleben",
        "length_m": 82.4,
        "created_at": CREATED,
    }


def test_track_response_adds_metadata_and_geometry() -> None:
    response = TrackResponse.from_track(_track(), GEOMETRY)

    assert response.metadata_json == {"country": "DE"}
    assert response.geometry == GEOMETRY
    assert response.model_dump(mode="json")["created_at"] == "2026-10-20T14:32:05Z"


def test_track_without_a_centreline_file_has_null_geometry() -> None:
    assert TrackResponse.from_track(_track(), None).model_dump(mode="json")["geometry"] is None
