"""Unit tests for lapzero/sim/env.py: episode rules, info dict, config and action mapping.

None of these start the simulator; LapZeroEnv runs on the scripted FakeSimulator.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from lapzero.contract import SIM_DT, STEER_MAX, STEER_RATE_MAX, V_MAX_TRAIN
from lapzero.sim.env import (
    EnvConfig,
    LapZeroEnv,
    sanitize_action,
    steer_to_rad,
    throttle_to_v_target,
    wrap_angle,
    zero_reward,
)
from lapzero.sim.simulator import Pose
from lapzero.sim.track import Centreline
from tests.conftest import START_POSE, FakeSimulator, make_state

COAST = np.array([0.0, 0.0], dtype=np.float32)
INFO_KEYS = {
    "t",
    "x",
    "y",
    "theta",
    "v_long",
    "steer_rad",
    "v_target",
    "lap_count",
    "lap_time",
    "collision",
    "progress",
}
ANY_FLOAT = st.floats(allow_nan=True, allow_infinity=True)
UNIT = st.floats(-1.0, 1.0)

EnvFactory = Callable[..., LapZeroEnv]


# Episode lifecycle


def test_reset_starts_at_time_zero_with_contract_info_keys(make_env: EnvFactory) -> None:
    _, info = make_env().reset(seed=0)
    assert set(info) == INFO_KEYS
    assert info["t"] == 0.0
    assert info["lap_time"] == 0.0
    assert info["progress"] == 0.0


def test_step_maps_once_and_repeats_the_command_for_four_physics_steps(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env()
    env.reset(seed=0)
    *_, info = env.step(np.array([0.2, 1.0], dtype=np.float32))
    assert len(fake_sim.commands) == 4
    assert len(set(fake_sim.commands)) == 1
    assert info["t"] == pytest.approx(4 * SIM_DT)


def test_collision_terminates_and_stops_physics_steps_early(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.collision_at_step = 2
    env = make_env()
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(COAST)
    assert terminated
    assert not truncated
    assert info["collision"] is True
    assert len(fake_sim.commands) == 2
    assert info["t"] == pytest.approx(2 * SIM_DT)


def test_reaching_laps_per_episode_terminates(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env(laps_per_episode=2)
    env.reset(seed=0)
    fake_sim.lap_count = 1
    assert env.step(COAST)[2] is False
    fake_sim.lap_count = 2
    assert env.step(COAST)[2] is True


def test_lap_time_restarts_when_a_lap_completes(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env(laps_per_episode=3)
    env.reset(seed=0)
    env.step(COAST)
    fake_sim.lap_count = 1
    *_, info = env.step(COAST)
    assert info["lap_time"] == 0.0
    *_, info = env.step(COAST)
    assert info["lap_time"] == pytest.approx(4 * SIM_DT)
    assert info["t"] == pytest.approx(12 * SIM_DT)


def test_episode_truncates_at_max_episode_time(make_env: EnvFactory) -> None:
    env = make_env(max_episode_s=0.2)
    env.reset(seed=0)
    truncations = [env.step(np.array([0.0, 1.0], dtype=np.float32))[3] for _ in range(5)]
    assert truncations == [False, False, False, False, True]


def test_car_slower_than_stuck_speed_for_three_seconds_truncates(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.follow_target = False
    fake_sim.v_long = 0.1
    env = make_env()
    env.reset(seed=0)
    truncations = [env.step(COAST)[3] for _ in range(75)]  # 75 x 0.04 s = 3 s
    assert not any(truncations[:74])
    assert truncations[74]


def test_moving_resets_the_stuck_timer(make_env: EnvFactory, fake_sim: FakeSimulator) -> None:
    fake_sim.follow_target = False
    env = make_env()
    env.reset(seed=0)
    for _ in range(70):
        env.step(COAST)
    fake_sim.v_long = 1.0
    env.step(COAST)
    fake_sim.v_long = 0.0
    assert not any(env.step(COAST)[3] for _ in range(70))


def test_reward_is_summed_over_the_repeated_physics_steps(make_env: EnvFactory) -> None:
    env = make_env(reward_fn=lambda prev, curr: 0.25)
    env.reset(seed=0)
    assert env.step(COAST)[1] == pytest.approx(1.0)


def test_placeholder_reward_is_zero(make_env: EnvFactory) -> None:
    env = make_env()
    env.reset(seed=0)
    assert env.step(COAST)[1] == 0.0
    assert zero_reward(make_state(), make_state()) == 0.0


def test_non_finite_reward_raises_value_error(make_env: EnvFactory) -> None:
    env = make_env(reward_fn=lambda prev, curr: math.nan)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="non-finite reward"):
        env.step(COAST)


def test_step_before_reset_raises_runtime_error(make_env: EnvFactory) -> None:
    with pytest.raises(RuntimeError, match="reset"):
        make_env().step(COAST)


def test_step_after_episode_end_raises_until_reset(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.collision_at_step = 1
    env = make_env()
    env.reset(seed=0)
    env.step(COAST)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(COAST)
    fake_sim.collision_at_step = None
    env.reset(seed=1)
    env.step(COAST)


def test_reset_clears_previous_action_and_steering(make_env: EnvFactory) -> None:
    env = make_env()
    env.reset(seed=0)
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    obs, info = env.reset(seed=1)
    assert obs[111] == 0.0
    assert obs[112] == 0.0
    assert info["steer_rad"] == 0.0
    assert info["v_target"] == 0.0


def test_wrong_action_shape_raises_value_error(make_env: EnvFactory) -> None:
    env = make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(3))


def test_info_reports_world_pose_with_wrapped_heading(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env()
    env.reset(seed=0)
    fake_sim.pose = Pose(4.0, -2.0, 1.5 * math.pi)
    *_, info = env.step(COAST)
    assert (info["x"], info["y"]) == (4.0, -2.0)
    assert info["theta"] == pytest.approx(-0.5 * math.pi)


def test_world_pose_never_enters_the_observation(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env()
    obs_a, _ = env.reset(seed=0)
    fake_sim.pose = Pose(50.0, -50.0, 3.0)
    obs_b, *_ = env.step(COAST)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_start_pose_is_used_unchanged_by_default(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env()
    env.reset(seed=0)
    env.reset(seed=1)
    assert fake_sim.reset_poses == [START_POSE, START_POSE]


def test_randomized_start_stays_within_contract_offsets_and_is_reproducible(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env(randomize_start=True)
    for seed in range(20):
        env.reset(seed=seed)
    env.reset(seed=3)
    poses = fake_sim.reset_poses
    assert poses[-1] == poses[3]
    assert len(set(poses[:20])) == 20
    for pose in poses:
        assert abs(pose.x - START_POSE.x) <= 0.3
        assert abs(pose.y - START_POSE.y) <= 0.3
        assert abs(pose.theta - START_POSE.theta) <= 0.1


def test_without_a_friction_range_the_env_reports_the_simulators_own_friction(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.friction = 0.8
    env = make_env()
    env.reset(seed=0)
    env.reset()
    assert fake_sim.friction == 0.8
    assert env.friction == 0.8


def test_friction_range_draws_a_new_friction_on_every_reset_reproducibly(
    make_env: EnvFactory, fake_sim: FakeSimulator
) -> None:
    env = make_env(friction_range=(0.5, 1.05))
    frictions = []
    for seed in range(20):
        env.reset(seed=seed)
        frictions.append(env.friction)
        assert fake_sim.friction == env.friction
    unseeded = []
    for _ in range(5):  # Gymnasium vector envs seed only the first reset
        env.reset()
        unseeded.append(env.friction)
    env.reset(seed=3)
    assert env.friction == frictions[3]
    assert len(set(frictions + unseeded)) == 25
    assert all(0.5 <= mu <= 1.05 for mu in frictions + unseeded)


def test_friction_range_keeps_the_seeds_start_jitter_and_the_info_keys() -> None:
    plain, randomized = FakeSimulator(), FakeSimulator()
    LapZeroEnv(plain, START_POSE, config=EnvConfig(randomize_start=True)).reset(seed=4)
    config = EnvConfig(randomize_start=True, friction_range=(0.5, 1.05))
    _, info = LapZeroEnv(randomized, START_POSE, config=config).reset(seed=4)
    assert randomized.reset_poses == plain.reset_poses
    assert set(info) == INFO_KEYS


def test_close_closes_the_simulator(make_env: EnvFactory, fake_sim: FakeSimulator) -> None:
    make_env().close()
    assert fake_sim.closed


# Progress along the centreline (info["progress"])

SQUARE = Centreline(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))  # 40 m


def _progress_after_moving_to(env: LapZeroEnv, sim: FakeSimulator, x: float, y: float) -> float:
    sim.pose = Pose(x, y, 0.0)
    _, _, _, _, info = env.step(COAST)
    progress: float = info["progress"]
    return progress


def test_progress_is_the_lap_fraction_driven_along_the_centreline(
    fake_sim: FakeSimulator,
) -> None:
    env = LapZeroEnv(fake_sim, Pose(0.0, 0.0, 0.0), centreline=SQUARE)
    _, info = env.reset(seed=0)
    assert info["progress"] == 0.0
    assert _progress_after_moving_to(env, fake_sim, 5.0, 0.5) == pytest.approx(0.125)
    assert _progress_after_moving_to(env, fake_sim, 10.0, 10.0) == pytest.approx(0.5)


def test_progress_is_measured_from_the_start_pose_and_wraps_each_lap(
    fake_sim: FakeSimulator,
) -> None:
    env = LapZeroEnv(fake_sim, Pose(10.0, 5.0, 1.57), centreline=SQUARE)  # 15 m along the line
    _, info = env.reset(seed=0)
    assert info["progress"] == 0.0
    assert _progress_after_moving_to(env, fake_sim, 10.0, 9.0) == pytest.approx(0.1)
    assert _progress_after_moving_to(env, fake_sim, 10.0, 4.0) == pytest.approx(0.975)


def test_progress_reference_is_the_fixed_start_pose_not_the_jittered_one(
    fake_sim: FakeSimulator,
) -> None:
    config = EnvConfig(randomize_start=True)
    env = LapZeroEnv(fake_sim, Pose(5.0, 0.0, 0.0), config=config, centreline=SQUARE)
    _, info = env.reset(seed=0)
    jittered = fake_sim.reset_poses[-1]
    assert info["progress"] == pytest.approx((jittered.x - 5.0) / 40.0 % 1.0)


def test_without_a_centreline_progress_stays_zero(fake_sim: FakeSimulator) -> None:
    env = LapZeroEnv(fake_sim, START_POSE)
    env.reset(seed=0)
    assert env.centreline is None
    assert _progress_after_moving_to(env, fake_sim, 25.0, -3.0) == 0.0


# Lap completion (contract Section 3; the edge cases are in tests/contracts/test_lap_completion.py)


def test_without_a_centreline_every_lap_the_simulator_counts_is_a_lap(
    fake_sim: FakeSimulator,
) -> None:
    env = LapZeroEnv(fake_sim, START_POSE)
    env.reset(seed=0)
    fake_sim.lap_count = 1
    _, _, terminated, _, info = env.step(COAST)
    assert terminated
    assert info["lap_count"] == 1


def test_each_lap_needs_its_own_90_percent_of_the_centreline(fake_sim: FakeSimulator) -> None:
    config = EnvConfig(laps_per_episode=2)
    env = LapZeroEnv(fake_sim, Pose(0.0, 0.0, 0.0), config=config, centreline=SQUARE)
    env.reset(seed=0)
    for x, y in [(5.0, 0.0), (10.0, 5.0), (5.0, 10.0), (0.0, 5.0)]:  # round the 40 m square
        _progress_after_moving_to(env, fake_sim, x, y)
    fake_sim.lap_count = 1
    _progress_after_moving_to(env, fake_sim, 0.0, 0.0)
    fake_sim.lap_count = 2  # the simulator counts a second pass right after the first lap
    _, _, terminated, _, info = env.step(COAST)
    assert not terminated
    assert info["lap_count"] == 1


def test_the_env_exposes_the_config_it_runs_with(fake_sim: FakeSimulator) -> None:
    config = EnvConfig(laps_per_episode=2, max_episode_s=30.0)
    assert LapZeroEnv(fake_sim, START_POSE, config=config).config is config


def test_the_env_exposes_the_centreline_it_measures_progress_along(
    fake_sim: FakeSimulator,
) -> None:
    assert LapZeroEnv(fake_sim, START_POSE, centreline=SQUARE).centreline is SQUARE


@given(st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
def test_progress_is_always_a_finite_lap_fraction(x: float, y: float) -> None:
    sim = FakeSimulator()
    env = LapZeroEnv(sim, START_POSE, centreline=SQUARE)
    env.reset(seed=0)
    progress = _progress_after_moving_to(env, sim, x, y)
    assert math.isfinite(progress)
    assert 0.0 <= progress <= 1.0


# Config validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"laps_per_episode": 0},
        {"max_episode_s": 0.0},
        {"max_episode_s": math.inf},
        {"accel_step": -0.6},
        {"brake_step": math.nan},
        {"v_max": 0.0},
        {"v_max": 20.5},
        {"friction_range": (0.0, 1.0)},
        {"friction_range": (1.05, 0.5)},
        {"friction_range": (0.5, math.inf)},
        {"friction_range": (math.nan, 1.0)},
    ],
)
def test_env_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=rf"EnvConfig\.{next(iter(overrides))}"):
        EnvConfig(**overrides)


# Action mapping: properties over any input, including NaN, inf and out-of-range values


@given(steer=ANY_FLOAT, prev=st.floats(-STEER_MAX, STEER_MAX), rate_limited=st.booleans())
def test_steering_angle_is_finite_within_steer_max_and_rate_limited(
    steer: float, prev: float, rate_limited: bool
) -> None:
    rate_max = STEER_RATE_MAX if rate_limited else None
    steer_rad = steer_to_rad(steer, prev, rate_max)
    assert math.isfinite(steer_rad)
    assert -STEER_MAX <= steer_rad <= STEER_MAX
    if rate_max is not None:
        assert abs(steer_rad - prev) <= rate_max + 1e-12


@given(throttle=ANY_FLOAT, v_long=ANY_FLOAT)
@example(throttle=-1.0, v_long=math.nan)
@example(throttle=1.0, v_long=math.inf)
def test_v_target_is_within_zero_and_v_max_train_for_any_input(
    throttle: float, v_long: float
) -> None:
    assert 0.0 <= throttle_to_v_target(throttle, v_long) <= V_MAX_TRAIN


@given(v_long=st.floats(-10.0, 30.0), a=UNIT, b=UNIT)
def test_more_throttle_never_lowers_v_target(v_long: float, a: float, b: float) -> None:
    low, high = sorted((a, b))
    assert throttle_to_v_target(low, v_long) <= throttle_to_v_target(high, v_long)


@given(st.lists(ANY_FLOAT, min_size=2, max_size=2))
@example([math.nan, math.inf])
def test_sanitized_action_is_finite_float32_in_unit_range(action: list[float]) -> None:
    result = sanitize_action(action)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert (np.abs(result) <= 1.0).all()


@given(st.lists(UNIT, min_size=2, max_size=2))
def test_in_range_actions_pass_through_unchanged(action: list[float]) -> None:
    np.testing.assert_array_equal(sanitize_action(action), np.array(action, dtype=np.float32))


def test_nan_action_component_becomes_zero() -> None:
    np.testing.assert_array_equal(sanitize_action([math.nan, 0.5]), [0.0, 0.5])


# Heading wrap


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3 * math.pi, math.pi),
        (1.5 * math.pi, -0.5 * math.pi),
        (-0.25 * math.pi, -0.25 * math.pi),
    ],
)
def test_wrap_angle_maps_into_half_open_pi_interval(theta: float, expected: float) -> None:
    assert wrap_angle(theta) == pytest.approx(expected)


@given(st.floats(-1e6, 1e6))
@example(math.pi + 4.4e-16)
def test_wrapped_angle_is_in_half_open_pi_interval_and_points_the_same_way(theta: float) -> None:
    wrapped = wrap_angle(theta)
    assert -math.pi < wrapped <= math.pi
    assert math.cos(wrapped) == pytest.approx(math.cos(theta), abs=1e-6)
    assert math.sin(wrapped) == pytest.approx(math.sin(theta), abs=1e-6)
