"""Contract Section 8 / test_action_space: action space and steering mapping (Sections 2.1, 2.2)."""

import math
from collections.abc import Callable

import numpy as np
import pytest
from gymnasium import spaces

from lapzero.sim.env import LapZeroEnv
from tests.conftest import FakeSimulator


def test_action_space_is_2d_float32_unit_box(make_env: Callable[..., LapZeroEnv]) -> None:
    space = make_env().action_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (2,)
    assert space.dtype == np.float32
    np.testing.assert_array_equal(space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(space.high, [1.0, 1.0])


@pytest.mark.parametrize(
    ("steer", "expected_rad"), [(1.0, 0.4189), (-1.0, -0.4189), (0.5, 0.20945)]
)
def test_steer_maps_linearly_to_steering_angle(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator, steer: float, expected_rad: float
) -> None:
    env = make_env(steer_rate_limit=False)
    env.reset(seed=0)
    *_, info = env.step(np.array([steer, 0.0], dtype=np.float32))
    assert info["steer_rad"] == pytest.approx(expected_rad)
    assert fake_sim.commands[-1][0] == pytest.approx(expected_rad)


def test_default_rate_limit_reaches_full_left_lock_in_three_control_steps(
    make_env: Callable[..., LapZeroEnv],
) -> None:
    env = make_env()
    env.reset(seed=0)
    angles = [env.step(np.array([1.0, 0.0], dtype=np.float32))[4]["steer_rad"] for _ in range(3)]
    assert angles == pytest.approx([0.15, 0.30, 0.4189])


@pytest.mark.parametrize(
    "action",
    [[5.0, -7.0], [-3.0, 2.5], [np.nan, np.inf], [-np.inf, np.nan], [1e30, -1e30]],
)
def test_out_of_range_actions_are_clipped_and_never_raise(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator, action: list[float]
) -> None:
    env = make_env(steer_rate_limit=False)
    env.reset(seed=0)
    obs, _, _, _, info = env.step(np.array(action))
    assert -0.4189 <= info["steer_rad"] <= 0.4189
    assert 0.0 <= info["v_target"] <= 8.0
    assert -1.0 <= obs[111] <= 1.0
    assert -1.0 <= obs[112] <= 1.0
    assert all(math.isfinite(value) for command in fake_sim.commands for value in command)
