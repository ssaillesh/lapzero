"""Unit tests for lapzero/io/centreline.py: track geometry from a centreline CSV."""

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from lapzero.io.centreline import Polyline, read_track_geometry, track_geometry


def _circle(radius: float, count: int) -> NDArray[np.float64]:
    """`count` vertices of a counter-clockwise circle about the origin, metres."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _radii(line: Polyline | None) -> NDArray[np.float64]:
    assert line is not None
    return np.hypot(line.x, line.y)


def test_track_edges_lie_the_track_width_either_side_of_the_centreline() -> None:
    points = _circle(10.0, 40)
    widths = np.column_stack((np.full(40, 1.0), np.full(40, 1.5)))  # right, left

    geometry = track_geometry(np.hstack((points, widths)))

    # Driving counter-clockwise, the left edge is on the inside of the circle.
    assert _radii(geometry.left) == pytest.approx(np.full(40, 8.5), abs=2e-3)
    assert _radii(geometry.right) == pytest.approx(np.full(40, 11.0), abs=2e-3)
    assert _radii(geometry.centreline) == pytest.approx(np.full(40, 10.0), abs=2e-3)


def test_reads_the_f1tenth_racetracks_csv_format(tmp_path: Path) -> None:
    rows = np.hstack((_circle(5.0, 12), np.full((12, 2), 0.5)))
    csv = tmp_path / "centreline.csv"
    lines = [f"{x:.6f},{y:.6f},{wr},{wl}" for x, y, wr, wl in rows]
    csv.write_text("# x_m, y_m, w_tr_right_m, w_tr_left_m\n" + "\n".join(lines) + "\n")

    assert read_track_geometry(csv) == track_geometry(np.round(rows, 6))


def test_centreline_without_widths_has_no_edges(tmp_path: Path) -> None:
    csv = tmp_path / "centreline.csv"
    csv.write_text("0,0\n4,0\n4,3\n")

    geometry = read_track_geometry(csv)

    assert geometry.centreline == Polyline(x=[0.0, 4.0, 4.0], y=[0.0, 0.0, 3.0])
    assert geometry.left is None
    assert geometry.right is None


def test_positions_are_rounded_to_millimetres() -> None:
    geometry = track_geometry(np.array([[0.12345, 0.0], [1.0, 0.98765], [0.0, 1.0]]))

    assert geometry.centreline.x[0] == 0.123
    assert geometry.centreline.y[1] == 0.988


def test_duplicate_vertices_and_a_closing_vertex_are_dropped() -> None:
    rows = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    assert track_geometry(rows).centreline == Polyline(x=[0.0, 1.0, 1.0], y=[0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (np.zeros((4, 3)), "2 or at least 4 columns"),
        (np.zeros((4, 1)), "2 or at least 4 columns"),
        (np.zeros(4), "2 or at least 4 columns"),
        (np.array([[0.0, 0.0], [1.0, math.nan], [0.0, 1.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, math.inf], [0.0, 1.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), "at least 3 distinct"),
        (np.array([[0, 0, 1, 1], [1, 0, -1, 1], [0, 1, 1, 1]], dtype=float), "negative"),
        (np.array([[0, 0, 1, 1], [1, 0, 1, 1], [2, 0, 1, 1], [1, 0, 1, 1]], float), "doubles back"),
    ],
)
def test_invalid_centreline_rows_are_rejected(rows: NDArray[np.float64], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        track_geometry(rows)


def test_missing_csv_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_track_geometry(tmp_path / "missing.csv")


def test_non_numeric_csv_row_raises_value_error(tmp_path: Path) -> None:
    csv = tmp_path / "centreline.csv"
    csv.write_text("0,0\n1,zero\n0,1\n")

    with pytest.raises(ValueError):
        read_track_geometry(csv)


@pytest.mark.parametrize(
    ("x", "y"),
    [([0.0, 1.0], [0.0, 1.0]), ([0, 1, 2, 3], [0, 1, 2]), ([0, 1, math.nan], [0] * 3)],
)
def test_polyline_rejects_short_unequal_or_non_finite_columns(
    x: list[float], y: list[float]
) -> None:
    with pytest.raises(ValidationError):
        Polyline(x=x, y=y)
