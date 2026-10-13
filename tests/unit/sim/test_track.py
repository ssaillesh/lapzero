"""Unit tests for lapzero/sim/track.py: centreline validation, projection, cross-track offset, start
pose and CSV loading."""

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from lapzero.sim.track import Centreline, load_centreline, start_pose_from_centreline

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])  # CCW, 40 m loop


def test_centreline_length_is_the_closed_loop_perimeter() -> None:
    assert Centreline(SQUARE).length == pytest.approx(40.0)


def test_a_last_point_repeating_the_first_is_dropped() -> None:
    centreline = Centreline(np.vstack([SQUARE, SQUARE[:1]]))
    np.testing.assert_array_equal(centreline.points, SQUARE)
    assert centreline.length == pytest.approx(40.0)


def test_consecutive_duplicate_points_are_dropped() -> None:
    centreline = Centreline(np.repeat(SQUARE, 3, axis=0))
    np.testing.assert_array_equal(centreline.points, SQUARE)


def test_centreline_points_cannot_be_modified() -> None:
    centreline = Centreline(SQUARE)
    with pytest.raises(ValueError, match="read-only"):
        centreline.points[0, 0] = 5.0


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.zeros((4, 3)), "shape"),
        (np.array([0.0, 1.0, 2.0]), "shape"),
        (np.array([[0.0, 0.0], [1.0, np.nan], [1.0, 1.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), "3 distinct"),
        (np.ones((5, 2)), "3 distinct"),
    ],
)
def test_invalid_centreline_points_are_rejected(points: NDArray[np.float64], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Centreline(points)


@pytest.mark.parametrize(
    ("x", "y", "arc"),
    [
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 5.0),
        (5.0, -1.0, 5.0),  # outside the loop
        (10.0, 5.0, 15.0),
        (9.0, 5.0, 15.0),  # inside the loop
        (5.0, 10.0, 25.0),
        (0.0, 7.5, 32.5),
        (-3.0, -3.0, 0.0),  # beyond a corner, closest to the first vertex
    ],
)
def test_projection_is_the_arc_length_of_the_closest_centreline_point(
    x: float, y: float, arc: float
) -> None:
    assert Centreline(SQUARE).project(x, y) == pytest.approx(arc)


@given(st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
def test_projection_always_lies_on_the_loop(x: float, y: float) -> None:
    centreline = Centreline(SQUARE)
    assert 0.0 <= centreline.project(x, y) < centreline.length


@pytest.mark.parametrize(
    ("x", "y", "arc", "offset"),
    [
        (5.0, 0.0, 5.0, 0.0),  # on the centreline
        (5.0, 1.0, 5.0, 1.0),  # inside the CCW loop: left of the driving direction
        (5.0, -1.5, 5.0, -1.5),  # outside: right of the driving direction
        (10.5, 5.0, 15.0, -0.5),
        (2.0, 9.0, 28.0, 1.0),
        (-3.0, -4.0, 0.0, -5.0),  # beyond a corner, closest to the first vertex
    ],
)
def test_locate_gives_the_arc_and_the_signed_offset_positive_to_the_left(
    x: float, y: float, arc: float, offset: float
) -> None:
    assert Centreline(SQUARE).locate(x, y) == pytest.approx((arc, offset))


def test_the_offset_sign_follows_the_driving_direction() -> None:
    clockwise = Centreline(SQUARE[::-1])
    assert clockwise.locate(5.0, 1.0)[1] == pytest.approx(-1.0)  # inside a CW loop is the right


@given(st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
def test_locate_agrees_with_project_and_is_never_further_than_the_nearest_vertex(
    x: float, y: float
) -> None:
    centreline = Centreline(SQUARE)
    arc, offset = centreline.locate(x, y)
    assert arc == centreline.project(x, y)
    nearest_vertex = float(np.min(np.linalg.norm(SQUARE - [x, y], axis=1)))
    assert abs(offset) <= nearest_vertex + 1e-9


@pytest.mark.parametrize(
    ("points", "x", "y", "theta"),
    [
        (SQUARE, 0.0, 0.0, 0.0),
        ([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]], 0.0, 0.0, math.pi / 2),
        ([[0.0, 10.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], 0.0, 10.0, -math.pi / 2),
    ],
)
def test_start_pose_is_the_first_vertex_facing_the_second(
    points: NDArray[np.float64], x: float, y: float, theta: float
) -> None:
    pose = start_pose_from_centreline(Centreline(np.asarray(points)))
    assert (pose.x, pose.y, pose.theta) == pytest.approx((x, y, theta))


def test_load_centreline_reads_the_f1tenth_racetracks_csv_format(tmp_path: Path) -> None:
    path = tmp_path / "centreline.csv"
    rows = "\n".join(f"{x}, {y}, 1.1, 1.2" for x, y in SQUARE)
    path.write_text(f"# x_m, y_m, w_tr_right_m, w_tr_left_m\n{rows}\n")
    centreline = load_centreline(path)
    np.testing.assert_array_equal(centreline.points, SQUARE)
    assert centreline.length == pytest.approx(40.0)


def test_load_centreline_rejects_non_numeric_rows(tmp_path: Path) -> None:
    path = tmp_path / "centreline.csv"
    path.write_text("0.0, 0.0\nten, 0.0\n10.0, 10.0\n")
    with pytest.raises(ValueError):
        load_centreline(path)


def test_load_centreline_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_centreline(tmp_path / "missing.csv")
