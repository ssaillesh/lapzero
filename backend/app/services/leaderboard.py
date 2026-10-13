"""Leaderboard: each driver's fastest completed lap on a track."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.domain import Lap
from lapzero.io.telemetry import DriverType, LapStatus


def best_laps(
    session: Session, track_id: str, *, driver_type: DriverType | None = None, limit: int = 10
) -> list[Lap]:
    """Return each driver's fastest completed lap on a track, fastest first.

    A driver is a human's name, or a model's name and registry version, so every model version
    gets its own entry. Equal lap times rank the earlier-recorded lap first.

    Args:
        session: Open session.
        track_id: Track to rank.
        driver_type: Rank only humans or only models; None ranks both.
        limit: Maximum number of drivers returned.

    Returns:
        Up to `limit` laps, one per driver.
    """
    lap_order = (Lap.lap_time_s, Lap.created_at, Lap.lap_id)
    ranked = select(
        Lap.lap_id,
        func.row_number()
        .over(
            partition_by=(Lap.driver_type, Lap.driver_name, Lap.model_version), order_by=lap_order
        )
        .label("place"),
    ).where(Lap.track_id == track_id, Lap.status == LapStatus.COMPLETED)
    if driver_type is not None:
        ranked = ranked.where(Lap.driver_type == driver_type)
    best = ranked.subquery()
    query = (
        select(Lap)
        .join(best, Lap.lap_id == best.c.lap_id)
        .where(best.c.place == 1)
        .order_by(*lap_order)
        .limit(limit)
    )
    return list(session.scalars(query))
