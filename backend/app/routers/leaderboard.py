"""GET /api/leaderboard/{track_id}."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.dependencies import SessionDep, TrackIdPath
from backend.app.schemas.leaderboard import LeaderboardEntry, LeaderboardResponse
from backend.app.services import leaderboard, tracks
from lapzero.io.telemetry import DriverType

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])


@router.get("/{track_id}")
def get_leaderboard(
    track_id: TrackIdPath,
    session: SessionDep,
    driver_type: DriverType | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> LeaderboardResponse:
    """Rank drivers on a track by their fastest completed lap, optionally humans or models only."""
    if tracks.get_track(session, track_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    best = leaderboard.best_laps(session, track_id, driver_type=driver_type, limit=limit)
    return LeaderboardResponse(
        track_id=track_id,
        driver_type=driver_type,
        entries=[LeaderboardEntry.from_lap(rank, lap) for rank, lap in enumerate(best, start=1)],
    )
