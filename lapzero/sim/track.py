"""Track geometry: the closed centreline that LapZeroEnv measures info["progress"] along."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)  # eq on array fields would raise; compare by identity
class Centreline:
    """Closed track centreline in the world (map) frame, as a polyline in driving order.

    Attributes:
        points: (N, 2) vertices x, y in metres with N >= 3 and consecutive duplicates removed
            (including a last vertex that repeats the first). The last vertex joins the first.
        length: Length of the closed loop, metres.
    """

    points: NDArray[np.float64]
    length: float = field(init=False)
    _segments: NDArray[np.float64] = field(init=False, repr=False)
    _segment_lengths: NDArray[np.float64] = field(init=False, repr=False)
    _starts: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Centreline.points must have shape (N, 2), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("Centreline.points must all be finite")
        to_next = np.roll(points, -1, axis=0) - points
        points = points[np.linalg.norm(to_next, axis=1) > 0.0]
        if len(points) < 3:
            raise ValueError(f"Centreline needs at least 3 distinct points, got {len(points)}")
        segments = np.roll(points, -1, axis=0) - points
        segment_lengths = np.linalg.norm(segments, axis=1)
        starts = np.concatenate(([0.0], np.cumsum(segment_lengths)[:-1]))
        for array in (points, segments, segment_lengths, starts):
            array.flags.writeable = False
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "length", float(segment_lengths.sum()))
        object.__setattr__(self, "_segments", segments)
        object.__setattr__(self, "_segment_lengths", segment_lengths)
        object.__setattr__(self, "_starts", starts)

    def project(self, x: float, y: float) -> float:
        """Return the arc length of the centreline point closest to (x, y).

        Args:
            x: World-frame x, metres.
            y: World-frame y, metres.

        Returns:
            Arc length from the first vertex in driving order, metres, in [0, length).
        """
        offsets = np.array([x, y]) - self.points
        along = np.einsum("ij,ij->i", offsets, self._segments) / self._segment_lengths**2
        along = np.clip(along, 0.0, 1.0)
        gaps = offsets - along[:, None] * self._segments
        nearest = int(np.argmin(np.einsum("ij,ij->i", gaps, gaps)))
        arc = self._starts[nearest] + along[nearest] * self._segment_lengths[nearest]
        return float(arc) % self.length


def load_centreline(path: Path) -> Centreline:
    """Read a centreline CSV whose first two columns are x and y in metres, in driving order.

    Reads the f1tenth_racetracks format (`# x_m, y_m, w_tr_right_m, w_tr_left_m`): lines that
    start with '#' are skipped and columns after the second are ignored.

    Args:
        path: CSV file, e.g. tracks/<track_id>/centreline.csv.

    Returns:
        The centreline.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a row is not numeric or the points do not form a valid centreline.
    """
    points = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1), ndmin=2)
    return Centreline(points)
