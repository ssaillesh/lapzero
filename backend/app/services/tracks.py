"""Tracks: rows registered from uploaded laps, and geometry read from TRACKS_DIR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.domain import Track
from lapzero.io.centreline import TrackGeometry, read_track_geometry
from lapzero.io.telemetry import TrackInfo

logger = logging.getLogger(__name__)

CENTRELINE_FILE: Final = "centreline.csv"
"""File name of a track's centreline in TRACKS_DIR/<track_id>/ (CLAUDE.md section 3)."""


def list_tracks(session: Session) -> list[Track]:
    """Return every track, ordered by track ID."""
    return list(session.scalars(select(Track).order_by(Track.track_id)))


def get_track(session: Session, track_id: str) -> Track | None:
    """Return the track with `track_id`, or None if no lap was uploaded on it."""
    return session.get(Track, track_id)


def ensure_track(session: Session, info: TrackInfo) -> Track:
    """Return the track a lap was driven on, adding it on its first lap.

    An existing track is returned unchanged. A new one is named after its ID and flushed, so a
    lap row can reference it; the caller commits.

    Args:
        session: Open session.
        info: Track block of the lap telemetry (length_m in metres).

    Returns:
        The track row.
    """
    track = session.get(Track, info.track_id)
    if track is None:
        track = Track(track_id=info.track_id, name=info.track_id, length_m=info.length_m)
        session.add(track)
        session.flush()
        logger.info("registered track %s (%.1f m)", info.track_id, info.length_m)
    return track


def load_geometry(tracks_dir: Path, track_id: str) -> TrackGeometry | None:
    """Read a track's geometry from tracks_dir/<track_id>/centreline.csv.

    Args:
        tracks_dir: The TRACKS_DIR setting.
        track_id: Track ID; a validated ID cannot leave tracks_dir (no path separators).

    Returns:
        The geometry, or None if the track has no centreline file.

    Raises:
        ValueError: If the file does not hold a valid centreline.
    """
    path = tracks_dir / track_id / CENTRELINE_FILE
    if not path.is_file():
        return None
    return read_track_geometry(path)
