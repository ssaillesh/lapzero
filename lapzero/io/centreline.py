"""Track geometry for the replay canvas (contract Section 6): the centreline and track edges.

Reads the f1tenth_racetracks centreline CSV (`# x_m, y_m, w_tr_right_m, w_tr_left_m`) kept at
tracks/<track_id>/centreline.csv. lapzero.sim.track reads the same file for the env; this reader
is separate because the backend imports lapzero.io and must never import lapzero.sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lapzero.contract import DECIMALS_POSITION

_WIDTH_COLUMNS = 4  # x, y, then the right and left track widths


class _Strict(BaseModel):
    """Base for geometry models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Polyline(_Strict):
    """Columnar vertices of a closed loop in the world (map) frame, metres.

    The last vertex joins the first.
    """

    x: list[float] = Field(min_length=3)
    y: list[float] = Field(min_length=3)

    @model_validator(mode="after")
    def _columns_have_equal_length(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError(f"x has {len(self.x)} vertices but y has {len(self.y)}")
        return self


class TrackGeometry(_Strict):
    """Track outline in driving order.

    Attributes:
        centreline: The racing centreline.
        left: Left track edge (seen in the driving direction); None if the CSV has no widths.
        right: Right track edge; None if the CSV has no widths.
    """

    centreline: Polyline
    left: Polyline | None
    right: Polyline | None


def read_track_geometry(path: Path) -> TrackGeometry:
    """Read a centreline CSV and return the track geometry.

    Lines that start with '#' are skipped. Columns are x, y and, optionally, the track widths to
    the right and left of the centreline, all in metres.

    Args:
        path: CSV file, e.g. tracks/<track_id>/centreline.csv.

    Returns:
        The geometry, positions rounded to DECIMALS_POSITION.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a row is not numeric or the rows do not form a valid track.
    """
    rows = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, dtype=np.float64)
    return track_geometry(rows)


def track_geometry(rows: NDArray[np.float64]) -> TrackGeometry:
    """Build the track geometry from centreline CSV rows.

    Each edge vertex is the centreline vertex moved sideways by the track width along the normal
    of the centreline's direction there (the chord from the previous to the next vertex).

    Args:
        rows: (N, 2) rows of x, y, or (N, 4+) rows of x, y, width right, width left (metres),
            in driving order. Consecutive duplicate vertices, including a last vertex repeating
            the first, are dropped.

    Returns:
        The geometry, positions rounded to DECIMALS_POSITION.

    Raises:
        ValueError: If rows has the wrong shape, a value is not finite, a width is negative, or
            fewer than 3 distinct vertices remain.
    """
    columns = rows.shape[1] if rows.ndim == 2 else 0
    if columns != 2 and columns < _WIDTH_COLUMNS:
        raise ValueError(f"centreline rows must have 2 or at least 4 columns, got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("centreline values must all be finite")
    to_next = np.roll(rows[:, :2], -1, axis=0) - rows[:, :2]
    rows = rows[np.linalg.norm(to_next, axis=1) > 0.0]
    if len(rows) < 3:
        raise ValueError(f"centreline needs at least 3 distinct vertices, got {len(rows)}")
    points = rows[:, :2]
    centreline = _polyline(points)
    if rows.shape[1] < _WIDTH_COLUMNS:
        return TrackGeometry(centreline=centreline, left=None, right=None)
    widths_right, widths_left = rows[:, 2], rows[:, 3]
    if (widths_right < 0.0).any() or (widths_left < 0.0).any():
        raise ValueError("track widths must not be negative")
    chords = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    lengths = np.linalg.norm(chords, axis=1)
    if (lengths == 0.0).any():
        raise ValueError("centreline doubles back on itself: a vertex has no direction")
    left_normals = np.column_stack((-chords[:, 1], chords[:, 0])) / lengths[:, None]
    return TrackGeometry(
        centreline=centreline,
        left=_polyline(points + widths_left[:, None] * left_normals),
        right=_polyline(points - widths_right[:, None] * left_normals),
    )


def _polyline(points: NDArray[np.float64]) -> Polyline:
    """Return (N, 2) points as a Polyline rounded to DECIMALS_POSITION."""
    return Polyline(
        x=[round(float(value), DECIMALS_POSITION) for value in points[:, 0]],
        y=[round(float(value), DECIMALS_POSITION) for value in points[:, 1]],
    )
