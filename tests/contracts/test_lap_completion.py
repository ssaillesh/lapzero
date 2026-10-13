"""Contract Section 8 / test_lap_completion: a lap needs 90% of the centreline driven (Section 3).

The fake simulator's lap_count stands in for F1TENTH's start gate, which counts any pass in either
direction. Each control step moves the car to a scripted point on a 40 m square centreline.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.simulator import Pose
from lapzero.sim.track import Centreline
from tests.conftest import FakeSimulator

SQUARE = Centreline(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))  # 40 m
ORIGIN = Pose(0.0, 0.0, 0.0)
DRIVE = np.array([0.0, 0.5], dtype=np.float32)  # half throttle, so the car is never stuck
_SIDES = [
    (0.0, 0.0, 1.0, 0.0),
    (10.0, 0.0, 0.0, 1.0),
    (10.0, 10.0, -1.0, 0.0),
    (0.0, 10.0, 0.0, -1.0),
]


def _pose_at(fraction: float) -> Pose:
    """Pose on SQUARE `fraction` of a lap from the origin in driving order."""
    side, along = divmod(fraction % 1.0 * 40.0, 10.0)
    x0, y0, dx, dy = _SIDES[int(side)]
    return Pose(x0 + dx * along, y0 + dy * along, 0.0)


def _fractions(start: float, end: float) -> Sequence[float]:
    """Lap fractions from `start` to `end` inclusive in 1 m steps, backwards if end < start."""
    return np.linspace(start, end, round(abs(end - start) * 40.0) + 1)[1:].tolist()


def _drive(
    env: LapZeroEnv, sim: FakeSimulator, fractions: Sequence[float], sim_laps: int
) -> tuple[bool, dict[str, Any]]:
    """Move the car through `fractions`; the simulator reports `sim_laps` on the last step."""
    terminated, info = False, dict[str, Any]()
    for i, fraction in enumerate(fractions):
        sim.pose = _pose_at(fraction)
        if i == len(fractions) - 1:
            sim.lap_count = sim_laps
        _, _, terminated, truncated, info = env.step(DRIVE)
        assert not truncated
    return terminated, info


@pytest.fixture
def env(fake_sim: FakeSimulator) -> LapZeroEnv:
    lap_zero_env = LapZeroEnv(fake_sim, ORIGIN, config=EnvConfig(), centreline=SQUARE)
    lap_zero_env.reset(seed=0)
    return lap_zero_env


def test_a_full_lap_through_the_start_line_counts_and_ends_a_one_lap_episode(
    env: LapZeroEnv, fake_sim: FakeSimulator
) -> None:
    terminated, info = _drive(env, fake_sim, _fractions(0.0, 1.0), sim_laps=1)
    assert terminated
    assert info["lap_count"] == 1


def test_a_start_line_pass_counts_only_after_90_percent_of_the_lap(
    env: LapZeroEnv, fake_sim: FakeSimulator
) -> None:
    terminated, info = _drive(env, fake_sim, _fractions(0.0, 0.875), sim_laps=1)
    assert not terminated
    assert info["lap_count"] == 0
    terminated, info = _drive(env, fake_sim, _fractions(0.875, 0.925), sim_laps=2)
    assert terminated
    assert info["lap_count"] == 1


def test_a_u_turn_back_through_the_start_line_is_not_a_lap(
    env: LapZeroEnv, fake_sim: FakeSimulator
) -> None:
    _drive(env, fake_sim, _fractions(0.0, 0.3), sim_laps=0)
    terminated, info = _drive(env, fake_sim, _fractions(0.3, 0.0), sim_laps=1)
    assert not terminated
    assert info["lap_count"] == 0
    assert info["lap_time"] == pytest.approx(info["t"])


def test_spinning_on_the_start_line_is_not_a_lap(env: LapZeroEnv, fake_sim: FakeSimulator) -> None:
    for sim_laps in (1, 2, 3):
        terminated, info = _drive(env, fake_sim, [0.0], sim_laps=sim_laps)
        assert not terminated
        assert info["lap_count"] == 0


def test_a_lap_driven_backwards_is_not_a_lap(env: LapZeroEnv, fake_sim: FakeSimulator) -> None:
    terminated, info = _drive(env, fake_sim, _fractions(1.0, 0.0), sim_laps=1)
    assert not terminated
    assert info["lap_count"] == 0


def test_a_full_lap_after_a_spin_on_the_start_line_counts_from_reset(
    env: LapZeroEnv, fake_sim: FakeSimulator
) -> None:
    _drive(env, fake_sim, [0.0], sim_laps=1)
    _, info = _drive(env, fake_sim, _fractions(0.0, 0.975), sim_laps=1)
    assert info["lap_time"] == pytest.approx(info["t"])  # the spin did not restart the lap clock
    terminated, info = _drive(env, fake_sim, [1.0], sim_laps=2)
    assert terminated
    assert info["lap_count"] == 1
