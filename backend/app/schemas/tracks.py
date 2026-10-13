"""Responses of GET /api/tracks and GET /api/tracks/{track_id}."""

from __future__ import annotations

from typing import Self

from pydantic import AwareDatetime

from backend.app.models.base import JsonObject
from backend.app.models.domain import Track
from backend.app.schemas.base import ResponseModel
from lapzero.io.centreline import TrackGeometry


class TrackSummary(ResponseModel):
    """A track with uploaded laps; length_m is the centreline length in metres."""

    track_id: str
    name: str
    length_m: float
    created_at: AwareDatetime


class TrackResponse(TrackSummary):
    """A track with its geometry for the replay canvas (contract Section 6).

    geometry is None when TRACKS_DIR has no centreline.csv for the track.
    """

    metadata_json: JsonObject
    geometry: TrackGeometry | None

    @classmethod
    def from_track(cls, track: Track, geometry: TrackGeometry | None) -> Self:
        """Build the response from a track row and its geometry."""
        return cls(
            track_id=track.track_id,
            name=track.name,
            length_m=track.length_m,
            created_at=track.created_at,
            metadata_json=track.metadata_json,
            geometry=geometry,
        )
