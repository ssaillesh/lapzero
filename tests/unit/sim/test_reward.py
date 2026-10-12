"""Unit tests for lapzero/sim/reward.py: reward terms, weights and episode ending.

RacingReward wraps a LapZeroEnv with a centreline on the scripted FakeSimulator; tests move the
car by setting the fake's pose before a step, and the env reports info["progress"].
"""

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.reward import MAX_PROGRESS_STEP, RacingReward, RewardConfig
from lapzero.sim.simulator import Pose, SimState
from lapzero.sim.track import Centreline
from tests.conftest import FakeSimulator

SQUARE = Centreline(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))  # 40 m
ORIGIN = Pose(0.0, 0.0, 0.0)
ONLY_PROGRESS = RewardConfig(
    progress_weight=2.0, speed_weight=0.0, collision_penalty=0.0, steer_jerk_weight=0.0
)
ONLY_SPEED = RewardConfig(
    progress_weight=0.0, speed_weight=0.5, collision_penalty=0.0, steer_jerk_weight=0.0
)
ONLY_JERK = RewardConfig(
    progress_weight=0.0, speed_weight=0.0, collision_penalty=0.0, steer_jerk_weight=0.1
)
ONLY_COLLISION = RewardConfig(
    progress_weight=0.0, speed_weight=0.0, collision_penalty=50.0, steer_jerk_weight=0.0
)
COAST = np.array([0.0, 0.0], dtype=np.float32)


def _wrapped(sim: FakeSimulator, config: RewardConfig, **env_config: Any) -> RacingReward:
    env = LapZeroEnv(sim, ORIGIN, config=EnvConfig(**env_config), centreline=SQUARE)
    return RacingReward(env, config)


def _action(steer: float, throttle: float = 0.0) -> NDArray[np.float32]:
    return np.array([steer, throttle], dtype=np.float32)


def _drive_to(env: RacingReward, sim: FakeSimulator, x: float, y: float) -> float:
    """Place the car at (x, y) and coast one control step; return the reward."""
    sim.pose = Pose(x, y, 0.0)
    _, reward, *_ = env.step(COAST)
    return reward


# RewardConfig


@pytest.mark.parametrize(
    "name", ["progress_weight", "speed_weight", "collision_penalty", "steer_jerk_weight"]
)
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_reward_weights_must_be_finite_and_non_negative(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        RewardConfig(**{name: value})


# RacingReward


def test_an_env_without_a_centreline_is_rejected(fake_sim: FakeSimulator) -> None:
    with pytest.raises(ValueError, match="centreline"):
        RacingReward(LapZeroEnv(fake_sim, ORIGIN))


def test_forward_progress_along_the_centreline_is_rewarded_per_metre(
    fake_sim: FakeSimulator,
) -> None:
    env = _wrapped(fake_sim, ONLY_PROGRESS)
    env.reset(seed=0)
    assert _drive_to(env, fake_sim, 0.3, 0.0) == pytest.approx(2.0 * 0.3)


def test_driving_backwards_along_the_centreline_is_penalized(fake_sim: FakeSimulator) -> None:
    env = _wrapped(fake_sim, ONLY_PROGRESS)
    env.reset(seed=0)
    assert _drive_to(env, fake_sim, 0.0, 0.3) == pytest.approx(-2.0 * 0.3)


def test_crossing_the_start_line_forwards_counts_as_forward_progress(
    fake_sim: FakeSimulator,
) -> None:
    env = _wrapped(fake_sim, ONLY_PROGRESS)
    env.reset(seed=0)
    _drive_to(env, fake_sim, 0.0, 0.2)  # 0.2 m behind the start line: progress 0.995
    assert _drive_to(env, fake_sim, 0.2, 0.0) == pytest.approx(2.0 * 0.4)  # progress 0.005


@pytest.mark.parametrize(("x", "y", "sign"), [(5.0, 0.0, 1.0), (0.0, 5.0, -1.0)])
def test_progress_per_step_is_capped_at_the_furthest_a_car_can_travel(
    fake_sim: FakeSimulator, x: float, y: float, sign: float
) -> None:
    env = _wrapped(fake_sim, ONLY_PROGRESS)
    env.reset(seed=0)
    assert _drive_to(env, fake_sim, x, y) == pytest.approx(sign * 2.0 * MAX_PROGRESS_STEP)


def test_speed_bonus_is_proportional_to_forward_speed(fake_sim: FakeSimulator) -> None:
    env = _wrapped(fake_sim, ONLY_SPEED)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(_action(0.0, 1.0))
    assert info["v_long"] == pytest.approx(0.6)
    assert reward == pytest.approx(0.5 * 0.6)


def test_rolling_backwards_earns_no_speed_bonus(fake_sim: FakeSimulator) -> None:
    fake_sim.follow_target = False
    fake_sim.v_long = -1.0
    env = _wrapped(fake_sim, ONLY_SPEED)
    env.reset(seed=0)
    _, reward, *_ = env.step(COAST)
    assert reward == 0.0


def test_a_collision_is_penalized_and_ends_the_episode(fake_sim: FakeSimulator) -> None:
    fake_sim.collision_at_step = 1
    env = _wrapped(fake_sim, ONLY_COLLISION)
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(COAST)
    assert reward == pytest.approx(-50.0)
    assert terminated
    assert not truncated


def test_a_steering_change_costs_its_square_and_steady_steering_costs_nothing(
    fake_sim: FakeSimulator,
) -> None:
    env = _wrapped(fake_sim, ONLY_JERK)
    env.reset(seed=0)
    rewards = [env.step(_action(steer))[1] for steer in (0.5, 0.5, -0.5, 1.0)]
    assert rewards == pytest.approx([-0.1 * 0.25, 0.0, -0.1 * 1.0, -0.1 * 2.25])


def test_a_nan_steering_action_counts_as_straight(fake_sim: FakeSimulator) -> None:
    env = _wrapped(fake_sim, ONLY_JERK)
    env.reset(seed=0)
    env.step(_action(0.5))
    _, reward, *_ = env.step(_action(float("nan")))
    assert reward == pytest.approx(-0.1 * 0.25)


def test_reset_restarts_the_progress_and_steering_references(fake_sim: FakeSimulator) -> None:
    config = RewardConfig(
        progress_weight=1.0, speed_weight=0.0, collision_penalty=0.0, steer_jerk_weight=0.1
    )
    env = _wrapped(fake_sim, config)
    env.reset(seed=0)
    fake_sim.pose = Pose(0.5, 0.0, 0.0)
    env.step(_action(1.0))
    env.reset(seed=1)  # car back at the start line, previous steering 0
    _, reward, *_ = env.step(COAST)
    assert reward == 0.0


def test_the_wrapped_envs_own_reward_is_replaced(fake_sim: FakeSimulator) -> None:
    def huge(prev: SimState, curr: SimState) -> float:
        return 100.0

    env = RacingReward(
        LapZeroEnv(fake_sim, ORIGIN, reward_fn=huge, centreline=SQUARE), ONLY_PROGRESS
    )
    env.reset(seed=0)
    _, reward, *_ = env.step(COAST)
    assert reward == 0.0


def test_defaults_to_the_default_reward_weights(fake_sim: FakeSimulator) -> None:
    env = RacingReward(LapZeroEnv(fake_sim, ORIGIN, centreline=SQUARE))
    assert env.config == RewardConfig()


@given(
    steer=st.floats(allow_nan=True, allow_infinity=True),
    throttle=st.floats(allow_nan=True, allow_infinity=True),
    x=st.floats(-1e3, 1e3),
    y=st.floats(-1e3, 1e3),
    collide=st.booleans(),
)
def test_reward_is_finite_for_any_action_and_position(
    steer: float, throttle: float, x: float, y: float, collide: bool
) -> None:
    sim = FakeSimulator()
    sim.collision_at_step = 1 if collide else None
    env = _wrapped(sim, RewardConfig())
    env.reset(seed=0)
    sim.pose = Pose(x, y, 0.0)
    _, reward, terminated, *_ = env.step(np.array([steer, throttle]))  # float64: no cast overflow
    assert np.isfinite(reward)
    assert terminated is collide
