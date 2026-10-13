"""Unit tests for lapzero/eval/runner.py: episode outcomes, lap timing, traces, observation noise
and summary statistics.

Episodes run on the FakeSimulator; CircuitSimulator moves the car round a 40 m square at a fixed
distance per physics step and counts laps, so a lap takes a known time.
"""

import dataclasses

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from lapzero.contract import LIDAR_SLICE, STEER_RATE_MAX
from lapzero.eval.perturb import ObservationNoise
from lapzero.eval.runner import (
    EpisodeResult,
    EpisodeTrace,
    EvalSummary,
    evaluate_policy,
    run_episode,
)
from lapzero.io.telemetry import LapStatus
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.simulator import Pose, SimState
from lapzero.sim.track import Centreline
from tests.conftest import FakeSimulator

SQUARE = Centreline(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))  # 40 m
ORIGIN = Pose(0.0, 0.0, 0.0)


class ConstantPolicy(nn.Module):
    """Always returns the same [steer, throttle] action."""

    def __init__(self, steer: float = 0.0, throttle: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("action", torch.tensor([steer, throttle]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("action").clone()


class CircuitSimulator(FakeSimulator):
    """Drives the car round SQUARE, 0.125 m per physics step: one lap is 80 control steps."""

    METRES_PER_STEP = 0.125

    def step(self, steer_rad: float, v_target: float) -> object:
        distance = (self.steps + 1) * self.METRES_PER_STEP
        side, along = divmod(distance % 40.0, 10.0)
        corners = [(0.0, 0.0, 1.0, 0.0), (10.0, 0.0, 0.0, 1.0), (10.0, 10.0, -1.0, 0.0)]
        x0, y0, dx, dy = (*corners, (0.0, 10.0, 0.0, -1.0))[int(side)]
        self.pose = Pose(x0 + dx * along, y0 + dy * along, 0.0)
        self.lap_count = int(distance // 40.0)
        return super().step(steer_rad, v_target)


def _env(sim: FakeSimulator, **config: object) -> LapZeroEnv:
    return LapZeroEnv(sim, ORIGIN, config=EnvConfig(**config), centreline=SQUARE)  # type: ignore[arg-type]


def test_a_completed_lap_reports_its_lap_time_and_one_full_lap_driven() -> None:
    result = run_episode(_env(CircuitSimulator()), ConstantPolicy(throttle=0.5), seed=3)
    assert result.status is LapStatus.COMPLETED
    assert result.lap_time_s == pytest.approx(3.2)  # 80 control steps of 0.04 s
    assert result.lap_fraction == pytest.approx(1.0, abs=1e-3)
    assert (result.seed, result.steps) == (3, 80)


def test_a_collision_is_a_crash_without_a_lap_time(fake_sim: FakeSimulator) -> None:
    fake_sim.collision_at_step = 10
    result = run_episode(_env(fake_sim), ConstantPolicy(throttle=1.0), seed=0)
    assert result.status is LapStatus.CRASHED
    assert result.lap_time_s is None
    assert result.steps == 3


def test_running_out_of_time_is_a_timeout(fake_sim: FakeSimulator) -> None:
    result = run_episode(_env(fake_sim, max_episode_s=0.2), ConstantPolicy(throttle=1.0), seed=0)
    assert result.status is LapStatus.TIMEOUT
    assert result.lap_time_s is None
    assert result.mean_speed_mps == pytest.approx(np.mean([0.6, 1.2, 1.8, 2.4, 3.0]))


def test_standing_still_is_stuck(fake_sim: FakeSimulator) -> None:
    result = run_episode(_env(fake_sim), ConstantPolicy(throttle=0.0), seed=0)
    assert result.status is LapStatus.STUCK
    assert result.steps == 75  # contract: 3 s below STUCK_SPEED


def test_without_a_centreline_no_lap_fraction_is_measured(fake_sim: FakeSimulator) -> None:
    env = LapZeroEnv(fake_sim, ORIGIN, config=EnvConfig(max_episode_s=0.2))
    fake_sim.pose = Pose(5.0, 0.0, 0.0)
    assert run_episode(env, ConstantPolicy(throttle=1.0), seed=0).lap_fraction == 0.0


def test_the_policy_is_run_in_eval_mode(fake_sim: FakeSimulator) -> None:
    policy = ConstantPolicy(throttle=1.0)
    run_episode(_env(fake_sim, max_episode_s=0.2), policy, seed=0)
    assert not policy.training


def test_evaluation_seeds_episode_i_with_seed_plus_i() -> None:
    summary = evaluate_policy(
        _env(CircuitSimulator()), ConstantPolicy(throttle=0.5), episodes=3, seed=5
    )
    assert [e.seed for e in summary.episodes] == [5, 6, 7]
    assert summary.completion_rate == 1.0


@pytest.mark.parametrize(("episodes", "laps", "message"), [(0, 1, "episodes"), (2, 3, "laps")])
def test_evaluation_needs_episodes_and_exactly_one_lap_each(
    fake_sim: FakeSimulator, episodes: int, laps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_policy(
            _env(fake_sim, laps_per_episode=laps), ConstantPolicy(), episodes=episodes, seed=0
        )


def _result(status: LapStatus, lap_time_s: float | None = None) -> EpisodeResult:
    return EpisodeResult(0, status, lap_time_s, 1.0, 5.0, 100)


def test_summary_rates_and_mean_lap_time_count_only_completed_laps() -> None:
    summary = EvalSummary(
        (
            _result(LapStatus.COMPLETED, 50.0),
            _result(LapStatus.COMPLETED, 54.0),
            _result(LapStatus.CRASHED),
            _result(LapStatus.STUCK),
        )
    )
    assert summary.completion_rate == 0.5
    assert summary.crash_rate == 0.25
    assert summary.lap_times_s == (50.0, 54.0)
    assert summary.mean_lap_time_s == pytest.approx(52.0)


def test_summary_without_completed_laps_has_no_mean_lap_time() -> None:
    assert EvalSummary((_result(LapStatus.CRASHED),)).mean_lap_time_s is None


class RecordingPolicy(ConstantPolicy):
    """ConstantPolicy that keeps a copy of every observation it is given."""

    def __init__(self, steer: float = 0.0, throttle: float = 0.0) -> None:
        super().__init__(steer, throttle)
        self.seen: list[NDArray[np.float32]] = []

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self.seen.append(obs.numpy().copy())
        return super().forward(obs)


def test_the_trace_has_one_row_per_control_step_ending_at_the_episode_end() -> None:
    result = run_episode(_env(CircuitSimulator()), ConstantPolicy(throttle=0.5), seed=0)
    trace = result.trace
    assert trace is not None
    assert len(trace.t) == result.steps == 80
    assert trace.t[0] == pytest.approx(0.04)
    assert trace.t[-1] == pytest.approx(3.2)
    np.testing.assert_allclose(trace.x[:4], [0.5, 1.0, 1.5, 2.0])  # 0.5 m per control step
    assert not trace.collision.any()


def test_the_trace_records_the_applied_action_and_the_command_it_produced() -> None:
    result = run_episode(_env(CircuitSimulator()), ConstantPolicy(steer=3.0, throttle=0.5), seed=0)
    trace = result.trace
    assert trace is not None
    assert (trace.steer == 1.0).all()  # clipped like the env clips it
    assert (trace.throttle == 0.5).all()
    assert trace.steer_rad[0] == pytest.approx(STEER_RATE_MAX)  # rate limited on the first step
    assert trace.v_target[0] == pytest.approx(0.3)  # 0 m/s + 0.5 * ACCEL_STEP


def test_a_crash_is_the_last_row_of_the_trace(fake_sim: FakeSimulator) -> None:
    fake_sim.collision_at_step = 10
    trace = run_episode(_env(fake_sim), ConstantPolicy(throttle=1.0), seed=0).trace
    assert trace is not None
    np.testing.assert_array_equal(trace.collision, [False, False, True])


class ParkedSimulator(FakeSimulator):
    """Moves the car to a fixed pose on its first step and keeps it there."""

    def __init__(self, pose: Pose) -> None:
        super().__init__()
        self.parked = pose

    def step(self, steer_rad: float, v_target: float) -> SimState:
        self.pose = self.parked
        return super().step(steer_rad, v_target)


def test_cross_track_error_is_the_signed_offset_from_the_centreline() -> None:
    sim = ParkedSimulator(Pose(5.0, -1.5, 0.0))  # outside the CCW square: right of the centreline
    trace = run_episode(_env(sim, max_episode_s=0.2), ConstantPolicy(), seed=0).trace
    assert trace is not None
    np.testing.assert_allclose(trace.cte_m, -1.5)
    assert trace.mean_abs_cte_m == trace.max_abs_cte_m == trace.rms_cte_m == pytest.approx(1.5)


def test_without_a_centreline_the_cross_track_error_is_zero() -> None:
    sim = ParkedSimulator(Pose(5.0, -1.5, 0.0))
    env = LapZeroEnv(sim, ORIGIN, config=EnvConfig(max_episode_s=0.2))
    trace = run_episode(env, ConstantPolicy(throttle=1.0), seed=0).trace
    assert trace is not None
    assert (trace.cte_m == 0.0).all()


def test_the_trace_is_left_out_of_the_logged_result() -> None:
    result = run_episode(_env(CircuitSimulator()), ConstantPolicy(throttle=0.5), seed=0)
    assert "trace" not in repr(result)


def test_without_noise_the_policy_sees_the_clean_observation(fake_sim: FakeSimulator) -> None:
    policy = RecordingPolicy(throttle=1.0)
    run_episode(_env(fake_sim, max_episode_s=0.2), policy, seed=0)
    assert all((obs[LIDAR_SLICE] == np.float32(10.0 / 30.0)).all() for obs in policy.seen)


def test_observation_noise_reaches_the_policy(fake_sim: FakeSimulator) -> None:
    policy = RecordingPolicy(throttle=1.0)
    noise = ObservationNoise(lidar_std_m=0.3)
    evaluate_policy(_env(fake_sim, max_episode_s=0.2), policy, episodes=1, seed=0, obs_noise=noise)
    beams = np.array([obs[LIDAR_SLICE] for obs in policy.seen])  # walls 10 m away, 5 steps
    assert (beams - 10.0 / 30.0).std() * 30.0 == pytest.approx(0.3, rel=0.15)


def test_observation_noise_is_seeded_by_the_episode_seed() -> None:
    noise = ObservationNoise(speed_std_mps=0.5)
    runs = []
    for seed in (4, 4, 5):
        policy = RecordingPolicy(throttle=1.0)
        env = _env(FakeSimulator(), max_episode_s=0.2)
        run_episode(env, policy, seed=seed, obs_noise=noise)
        runs.append(np.array(policy.seen))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def _trace(cte_m: list[float], v_long: list[float] | None = None) -> EpisodeTrace:
    n = len(cte_m)
    zeros = np.zeros(n)
    return EpisodeTrace(
        t=np.arange(1, n + 1) * 0.04,
        x=zeros,
        y=zeros,
        theta=zeros,
        v_long=np.array(v_long) if v_long is not None else zeros,
        steer=zeros,
        throttle=zeros,
        steer_rad=zeros,
        v_target=zeros,
        progress=zeros,
        cte_m=np.array(cte_m),
        collision=np.zeros(n, dtype=bool),
    )


def test_trace_statistics_use_the_magnitude_of_the_cross_track_error() -> None:
    trace = _trace([0.6, -0.8, 0.0, 0.0], v_long=[1.0, 4.0, 3.0, 2.0])
    assert trace.mean_abs_cte_m == pytest.approx(0.35)
    assert trace.max_abs_cte_m == pytest.approx(0.8)
    assert trace.rms_cte_m == pytest.approx(0.5)
    assert trace.max_speed_mps == pytest.approx(4.0)


@pytest.mark.parametrize("cte_m", [[], [0.0, 1.0]])
def test_every_trace_column_has_the_same_length(cte_m: list[float]) -> None:
    with pytest.raises(ValueError, match="cte_m"):
        dataclasses.replace(_trace([0.0]), cte_m=np.array(cte_m))


def test_a_trace_has_at_least_one_row() -> None:
    with pytest.raises(ValueError, match="row"):
        _trace([])
