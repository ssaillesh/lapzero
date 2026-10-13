"""Response of GET /api/leaderboard/{track_id}."""

from __future__ import annotations

from typing import Self

from pydantic import AwareDatetime, Field

from backend.app.models.domain import Lap
from backend.app.schemas.base import ResponseModel
from lapzero.io.telemetry import DriverType


class LeaderboardEntry(ResponseModel):
    """One driver's fastest completed lap; rank 1 is the fastest driver, lap_time_s in s."""

    rank: int = Field(ge=1)
    lap_id: str
    driver_type: DriverType
    driver_name: str
    driver_label: str
    model_version: str | None
    lap_time_s: float = Field(gt=0.0)
    created_at: AwareDatetime

    @classmethod
    def from_lap(cls, rank: int, lap: Lap) -> Self:
        """Build the entry at `rank` from a driver's best lap.

        Raises:
            ValueError: If the lap has no lap time (it did not complete).
        """
        if lap.lap_time_s is None:
            raise ValueError(f"lap {lap.lap_id} has no lap_time_s: only completed laps rank")
        return cls(
            rank=rank,
            lap_id=lap.lap_id,
            driver_type=lap.driver_type,
            driver_name=lap.driver_name,
            driver_label=lap.driver_label,
            model_version=lap.model_version,
            lap_time_s=lap.lap_time_s,
            created_at=lap.created_at,
        )


class LeaderboardResponse(ResponseModel):
    """Fastest drivers on a track, fastest first; driver_type is the filter applied, if any."""

    track_id: str
    driver_type: DriverType | None
    entries: list[LeaderboardEntry]
