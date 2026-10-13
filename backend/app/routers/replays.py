"""GET /api/replays/{lap_id}: the replay trajectory of a lap (contract Section 6)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.dependencies import LapIdPath, SessionDep
from backend.app.services import laps
from lapzero.io.replay import Replay

router = APIRouter(prefix="/api/replays", tags=["replays"])


@router.get("/{lap_id}")
def get_replay(lap_id: LapIdPath, session: SessionDep) -> Replay:
    """Return the 10 Hz replay of an uploaded lap."""
    replay = laps.get_replay(session, lap_id)
    if replay is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"lap {lap_id} not found")
    return replay
