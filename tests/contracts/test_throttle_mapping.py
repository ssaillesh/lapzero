"""Contract Section 8 / test_throttle_mapping: throttle/brake to target velocity (Section 2.3)."""

from collections.abc import Callable

import numpy as np
import pytest

from lapzero.sim.env import LapZeroEnv, throttle_to_v_target
from tests.conftest import FakeSimulator


def test_full_throttle_raises_v_target_by_accel_step() -> None:
    assert throttle_to_v_target(1.0, v_long=3.0) == pytest.approx(3.0 + 0.6)


def test_full_brake_lowers_v_target_by_brake_step() -> None:
    assert throttle_to_v_target(-1.0, v_long=3.0) == pytest.approx(3.0 - 1.2)


def test_zero_throttle_holds_current_speed() -> None:
    assert throttle_to_v_target(0.0, v_long=3.0) == pytest.approx(3.0)


def test_brake_never_drops_v_target_below_zero() -> None:
    assert throttle_to_v_target(-1.0, v_long=0.5) == 0.0


def test_throttle_never_raises_v_target_above_v_max_train() -> None:
    assert throttle_to_v_target(1.0, v_long=7.9) == 8.0


@pytest.mark.parametrize(
    "throttle", [-1e9, -1.0, -0.3, 0.0, 0.3, 1.0, 1e9, np.nan, np.inf, -np.inf]
)
@pytest.mark.parametrize("v_long", [-10.0, 0.0, 4.0, 8.0, 50.0])
def test_v_target_always_stays_between_zero_and_v_max_train(throttle: float, v_long: float) -> None:
    assert 0.0 <= throttle_to_v_target(throttle, v_long) <= 8.0


def test_env_sends_mapped_target_velocity_for_every_repeated_physics_step(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator
) -> None:
    fake_sim.follow_target = False
    fake_sim.v_long = 3.0
    env = make_env()
    env.reset(seed=0)
    *_, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert info["v_target"] == pytest.approx(3.6)
    assert [v_target for _, v_target in fake_sim.commands] == pytest.approx([3.6] * 4)
