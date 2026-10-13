"""Unit tests for backend/app/services/tracks.py."""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from backend.app.services.tracks import (
    CENTRELINE_FILE,
    ensure_track,
    get_track,
    list_tracks,
    load_geometry,
)
from lapzero.io.centreline import Polyline
from lapzero.io.telemetry import TrackDirection, TrackInfo


def _info(track_id: str, length_m: float = 82.4) -> TrackInfo:
    return TrackInfo(track_id=track_id, direction=TrackDirection.CCW, length_m=length_m)


def _write_centreline(tracks_dir: Path, track_id: str, text: str) -> None:
    directory = tracks_dir / track_id
    directory.mkdir(parents=True)
    (directory / CENTRELINE_FILE).write_text(text)


def test_tracks_are_listed_by_track_id(session: Session) -> None:
    ensure_track(session, _info("zandvoort"))
    ensure_track(session, _info("oschersleben"))

    assert [track.track_id for track in list_tracks(session)] == ["oschersleben", "zandvoort"]


def test_unknown_track_is_none(session: Session) -> None:
    assert get_track(session, "nowhere") is None


def test_first_lap_registers_the_track_named_after_its_id(session: Session) -> None:
    track = ensure_track(session, _info("oschersleben"))

    assert get_track(session, "oschersleben") is track
    assert (track.name, track.length_m) == ("oschersleben", 82.4)


def test_later_laps_leave_the_registered_track_unchanged(session: Session) -> None:
    first = ensure_track(session, _info("oschersleben", 82.4))

    again = ensure_track(session, _info("oschersleben", 99.0))

    assert again is first
    assert again.length_m == 82.4
    assert len(list_tracks(session)) == 1


def test_geometry_is_read_from_the_tracks_centreline_file(tmp_path: Path) -> None:
    _write_centreline(tmp_path, "oschersleben", "# x_m, y_m\n0,0\n4,0\n4,3\n")

    geometry = load_geometry(tmp_path, "oschersleben")

    assert geometry is not None
    assert geometry.centreline == Polyline(x=[0.0, 4.0, 4.0], y=[0.0, 0.0, 3.0])


def test_track_without_a_centreline_file_has_no_geometry(tmp_path: Path) -> None:
    assert load_geometry(tmp_path, "oschersleben") is None


def test_invalid_centreline_file_raises_value_error(tmp_path: Path) -> None:
    _write_centreline(tmp_path, "oschersleben", "0,0\n1,0\n")

    with pytest.raises(ValueError, match="at least 3 distinct"):
        load_geometry(tmp_path, "oschersleben")
