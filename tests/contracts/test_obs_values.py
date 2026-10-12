"""Contract Section 8 / test_obs_values: 500 random steps give finite, in-bounds observations.

The fake simulator feeds NaN, +/-inf, negative and huge readings on every physics step, so the
whole sanitizing path (Sections 0 and 1.3) is exercised.
"""

import math
from collections.abc import Callable

import numpy as np

from lapzero.sim.env import LapZeroEnv
from tests.conftest import FakeSimulator


def test_500_random_steps_give_finite_observations_inside_bounds(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator
) -> None:
    fake_sim.rng = np.random.default_rng(0)
    env = make_env()
    env.action_space.seed(0)
    low, high = np.zeros(113), np.ones(113)
    low[108:113] = -1.0
    obs, _ = env.reset(seed=0)

    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (113,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert (obs >= low).all()
        assert (obs <= high).all()
        assert math.isfinite(reward)
        numbers = [value for value in info.values() if isinstance(value, float)]
        assert all(math.isfinite(value) for value in numbers)
        if terminated or truncated:
            env.reset()
