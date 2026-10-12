"""Contract Section 8 / test_obs_space: observation shape, dtype and bounds (Section 1.2).

Values are written as literals from docs/contracts/state_space.md on purpose, so a silent change
to lapzero/contract.py fails here.
"""

from collections.abc import Callable

import numpy as np
from gymnasium import spaces

from lapzero import contract
from lapzero.sim.env import LapZeroEnv


def test_observation_space_is_flat_113_float32_box(make_env: Callable[..., LapZeroEnv]) -> None:
    space = make_env().observation_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (113,)
    assert space.dtype == np.float32


def test_observation_bounds_are_unit_lidar_then_symmetric_unit_state(
    make_env: Callable[..., LapZeroEnv],
) -> None:
    space = make_env().observation_space
    assert isinstance(space, spaces.Box)
    np.testing.assert_array_equal(space.low[0:108], np.zeros(108))
    np.testing.assert_array_equal(space.high[0:108], np.ones(108))
    np.testing.assert_array_equal(space.low[108:113], -np.ones(5))
    np.testing.assert_array_equal(space.high[108:113], np.ones(5))


def test_contract_module_exports_documented_layout() -> None:
    assert contract.OBS_DIM == 113
    assert (contract.LIDAR_SLICE.start, contract.LIDAR_SLICE.stop) == (0, 108)
    indices = (
        contract.V_LONG,
        contract.V_LAT,
        contract.YAW_RATE,
        contract.PREV_STEER,
        contract.PREV_THROTTLE,
    )
    assert indices == (108, 109, 110, 111, 112)


def test_reset_observation_is_in_space_with_zero_previous_action(
    make_env: Callable[..., LapZeroEnv],
) -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (113,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert obs[111] == 0.0
    assert obs[112] == 0.0


def test_step_observation_is_in_space_and_carries_previous_action(
    make_env: Callable[..., LapZeroEnv],
) -> None:
    env = make_env()
    env.reset(seed=0)
    obs, *_ = env.step(np.array([0.5, -0.25], dtype=np.float32))
    assert obs.shape == (113,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert obs[111] == np.float32(0.5)
    assert obs[112] == np.float32(-0.25)
