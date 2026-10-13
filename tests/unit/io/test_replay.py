"""Unit tests for lapzero/io/replay.py: the contract Section 6 replay built from lap telemetry."""

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from lapzero.contract import CTRL_DT, DECIMALS_ANGLE, REPLAY_HZ, ContractViolationError
from lapzero.io.replay import Replay, ReplayPoints, build_replay, driver_label
from lapzero.io.telemetry import DriverType, LapTelemetry
from tests.unit.io.test_telemetry import contract_example


def _telemetry(**columns: list[float]) -> LapTelemetry:
    """The contract example lap with frames built from `columns`; unset columns are zeros."""
    count = len(columns["t"])
    document = contract_example()
    document["frames"] = {name: columns.get(name, [0.0] * count) for name in document["frames"]}
    document["result"]["sample_count"] = count
    return LapTelemetry.model_validate(document)


def _ctrl_times(start: float, count: int) -> list[float]:
    """`count` control-rate (25 Hz) frame times from `start`, rounded like telemetry t."""
    return [round(start + i * CTRL_DT, 2) for i in range(count)]


def _replay_document(**overrides: Any) -> dict[str, Any]:
    """The contract Section 6 example replay with top-level fields replaced."""
    document: dict[str, Any] = {
        "schema": "lapzero.replay",
        "schema_version": "1.0.0",
        "lap_id": "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
        "track_id": "track_02",
        "driver_label": "AI v7",
        "lap_time_s": 35.97,
        "hz": 10,
        "points": {
            "t": [0.0, 0.1, 0.2],
            "x": [0.000, 0.162, 0.412],
            "y": [0.000, 0.009, 0.031],
            "theta": [0.0000, 0.0110, 0.0240],
            "v": [0.00, 1.46, 2.81],
            "steer": [0.000, 0.071, 0.090],
        },
    }
    return document | overrides


def test_contract_example_replay_validates_and_round_trips() -> None:
    document = _replay_document()

    replay = Replay.model_validate(document)

    assert json.loads(replay.model_dump_json(by_alias=True)) == document


def test_contract_example_lap_builds_a_section_6_replay() -> None:
    replay = build_replay(LapTelemetry.model_validate(contract_example()))

    dumped = replay.model_dump(by_alias=True)
    assert list(dumped) == [
        "schema",
        "schema_version",
        "lap_id",
        "track_id",
        "driver_label",
        "lap_time_s",
        "hz",
        "points",
    ]
    assert dumped["schema"] == "lapzero.replay"
    assert replay.hz == REPLAY_HZ == 10
    assert replay.lap_id == "01J9ZK8V3Q7M2T6C1R4X5N8B0A"
    assert replay.track_id == "track_02"
    assert replay.driver_label == "AI v7"
    assert replay.lap_time_s == 35.97
    assert replay.points.t == [0.0]  # 0.08 s of frames hold one 10 Hz sample


def test_replay_samples_every_tenth_of_a_second_by_linear_interpolation() -> None:
    t = _ctrl_times(0.0, 26)  # 0.00 to 1.00 s
    telemetry = _telemetry(
        t=t,
        x=[2.0 * ti for ti in t],
        y=[-ti for ti in t],
        v_long=[3.0 * ti for ti in t],
        steer=[ti / 2.0 for ti in t],
    )

    points = build_replay(telemetry).points

    expected_t = [round(k / 10, 1) for k in range(11)]
    assert points.t == expected_t
    assert points.x == pytest.approx([2.0 * ti for ti in expected_t])
    assert points.y == pytest.approx([-ti for ti in expected_t])
    assert points.v == pytest.approx([3.0 * ti for ti in expected_t])
    assert points.steer == pytest.approx([ti / 2.0 for ti in expected_t])


def test_replay_time_restarts_at_zero_on_a_later_lap_of_the_episode() -> None:
    t = _ctrl_times(36.0, 11)  # second lap of an episode: 36.00 to 36.40 s
    telemetry = _telemetry(t=t, x=[ti - 36.0 for ti in t])

    points = build_replay(telemetry).points

    assert points.t == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert points.x == pytest.approx(points.t)


def test_last_sample_is_kept_when_float_error_shortens_the_lap() -> None:
    # 0.24 - 0.04 is 0.19999999999999998 in floating point, still a 0.2 s lap.
    telemetry = _telemetry(t=_ctrl_times(0.04, 6))

    assert build_replay(telemetry).points.t == [0.0, 0.1, 0.2]


def test_heading_crossing_pi_is_interpolated_the_short_way_round() -> None:
    telemetry = _telemetry(t=[0.0, 0.2], theta=[3.1, -3.1])

    theta = build_replay(telemetry).points.theta

    assert theta[0] == 3.1
    assert abs(theta[1]) == pytest.approx(math.pi, abs=10**-DECIMALS_ANGLE)
    assert theta[2] == -3.1


def test_heading_is_wrapped_to_minus_pi_exclusive_pi_inclusive() -> None:
    telemetry = _telemetry(t=[0.0, 0.1], theta=[-math.pi, -math.pi])

    assert build_replay(telemetry).points.theta == [round(math.pi, DECIMALS_ANGLE)] * 2


def test_single_frame_lap_replays_as_one_point() -> None:
    telemetry = _telemetry(t=[5.0], x=[1.25], v_long=[2.5])

    points = build_replay(telemetry).points

    assert (points.t, points.x, points.v) == ([0.0], [1.25], [2.5])


@pytest.mark.parametrize("t", [[0.0, 0.04, 0.04], [0.0, 0.08, 0.04]])
def test_frame_time_that_does_not_increase_is_a_contract_violation(t: list[float]) -> None:
    with pytest.raises(ContractViolationError, match=r"frames\.t\[2\]"):
        build_replay(_telemetry(t=t))


def test_incomplete_lap_replays_without_a_lap_time() -> None:
    document = contract_example()
    document["result"] |= {"status": "crashed", "lap_time_s": None, "collision": True}

    replay = build_replay(LapTelemetry.model_validate(document))

    assert replay.lap_time_s is None


def test_model_driver_is_labelled_with_its_registry_version() -> None:
    assert driver_label(DriverType.MODEL, "lapzero-ppo", "7") == "AI v7"


def test_human_driver_is_labelled_with_their_name() -> None:
    assert driver_label(DriverType.HUMAN, "sam", None) == "sam"


@pytest.mark.parametrize(
    "overrides",
    [
        {"hz": 25},
        {"schema": "lapzero.lap_telemetry"},
        {"lap_id": "not-a-ulid"},
        {"track_id": "../etc"},
        {"driver_label": ""},
        {"lap_time_s": 0.0},
        {"extra": 1},
    ],
)
def test_replay_rejects_fields_that_break_the_contract(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Replay.model_validate(_replay_document(**overrides))


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("t", [0.0, 0.1]),
        ("x", [0.0, math.nan, 0.4]),
        ("v", [0.0, math.inf, 2.8]),
        ("steer", [0.0, 1.5, 0.0]),
        ("t", [-0.1, 0.1, 0.2]),
    ],
)
def test_replay_points_reject_bad_columns(column: str, values: list[float]) -> None:
    points = _replay_document()["points"] | {column: values}

    with pytest.raises(ValidationError):
        ReplayPoints.model_validate(points)


_finite = st.floats(min_value=-1e3, max_value=1e3)


@st.composite
def _laps(draw: st.DrawFn) -> LapTelemetry:
    """Telemetry with strictly increasing t from any start, any heading and in-range actions."""
    start = draw(st.floats(min_value=0.0, max_value=100.0))
    steps = draw(st.lists(st.floats(min_value=0.001, max_value=0.5), min_size=0, max_size=60))
    t = np.cumsum([start, *steps]).tolist()
    count = len(t)
    columns = st.lists(_finite, min_size=count, max_size=count)
    return _telemetry(
        t=t,
        x=draw(columns),
        y=draw(columns),
        theta=draw(st.lists(st.floats(-10.0, 10.0), min_size=count, max_size=count)),
        v_long=draw(columns),
        steer=draw(st.lists(st.floats(-1.0, 1.0), min_size=count, max_size=count)),
    )


@given(_laps())
def test_any_valid_lap_replays_finite_points_on_the_10_hz_grid(telemetry: LapTelemetry) -> None:
    points = build_replay(telemetry).points

    elapsed = telemetry.frames.t[-1] - telemetry.frames.t[0]
    assert points.t == [round(k / REPLAY_HZ, 2) for k in range(len(points.t))]
    assert points.t[-1] <= elapsed + 1e-6 < points.t[-1] + 1 / REPLAY_HZ
    for column in (points.x, points.y, points.theta, points.v, points.steer):
        assert all(math.isfinite(value) for value in column)
    assert all(abs(theta) <= round(math.pi, DECIMALS_ANGLE) for theta in points.theta)
    assert all(-1.0 <= steer <= 1.0 for steer in points.steer)
