"""GET /api/tracks and GET /api/tracks/{track_id}."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.dependencies import SessionDep, SettingsDep, TrackIdPath
from backend.app.schemas.tracks import TrackResponse, TrackSummary
from backend.app.services import tracks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tracks", tags=["tracks"])


@router.get("")
def list_tracks(session: SessionDep) -> list[TrackSummary]:
    """List every track that has uploaded laps."""
    return [TrackSummary.model_validate(track) for track in tracks.list_tracks(session)]


@router.get("/{track_id}")
def get_track(track_id: TrackIdPath, session: SessionDep, settings: SettingsDep) -> TrackResponse:
    """Return a track with its centreline and edges, fetched once by the replay canvas."""
    track = tracks.get_track(session, track_id)
    if track is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    try:
        geometry = tracks.load_geometry(settings.tracks_dir, track_id)
    except ValueError as error:
        logger.error("centreline of track %s is invalid: %s", track_id, error)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"centreline of track {track_id} is invalid"
        ) from error
    return TrackResponse.from_track(track, geometry)
