"""Unit tests for lapzero/inference/engine.py: policy sources, step-by-step driving, episode
outcomes and their telemetry.

The engine never starts F1TENTH here: F1TenthSimulator is replaced with a scripted fake.
CircuitSimulator moves the car round a 40 m square at a fixed distance per physics step and
counts start-line passes, so a lap takes a known number of control steps. Policies are real
save_policy checkpoints whose output layer is pinned to a constant action.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import lapzero.inference.engine as engine_module
from lapzero.contract import ACT_DIM, CTRL_DT, MAX_EPISODE_S, OBS_DIM, STUCK_S, VY_NORM
from lapzero.eval.gate import NOMINAL_MU
from lapzero.inference.engine import DEFAULT_MAX_STEPS, InferenceEngine, load_policy_source
from lapzero.io.telemetry import LapStatus
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.reward import RacingReward
from lapzero.sim.simulator import Pose
from lapzero.sim.track import Centreline
from lapzero.training.policy import BCPolicy, save_policy
from lapzero.training.tracking import mlflow_client, mlflow_run
from tests.conftest import FakeSimulator

SQUARE_POINTS = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]  # 40 m, start at origin
START = Pose(0.0, 0.0, 0.0)  # first centreline point facing the second
LAP_STEPS = 80  # control steps per lap on CircuitSimulator
MODEL = "test-policy"

EngineFactory = Callable[..., InferenceEngine]


class CircuitSimulator(FakeSimulator):
    """Drives the car round the square, 0.125 m per physics step: one lap is LAP_STEPS steps."""

    METRES_PER_STEP = 0.125

    def step(self, steer_rad: float, v_target: float) -> Any:
        distance = (self.steps + 1) * self.METRES_PER_STEP
        side, along = divmod(distance % 40.0, 10.0)
        corners = [(0.0, 0.0, 1.0, 0.0), (10.0, 0.0, 0.0, 1.0), (10.0, 10.0, -1.0, 0.0)]
        x0, y0, dx, dy = (*corners, (0.0, 10.0, 0.0, -1.0))[int(side)]
        self.pose = Pose(x0 + dx * along, y0 + dy * along, 0.0)
        self.lap_count = int(distance // 40.0)
        return super().step(steer_rad, v_target)


def constant_policy(directory: Path, steer: float, throttle: float) -> Path:
    """Save a BCPolicy whose action is always [steer, throttle] (each strictly inside +/-1)."""
    policy = BCPolicy(8)
    output = policy.net[-2]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(torch.atanh(torch.tensor([steer, throttle])))
    path = directory / f"constant_{steer}_{throttle}.pt"
    save_policy(policy, path)
    return path


@pytest.fixture
def simulator_calls() -> list[tuple[Path, int]]:
    return []


@pytest.fixture
def make_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulator_calls: list[tuple[Path, int]]
) -> EngineFactory:
    """Factory for an engine on `sim` driving a constant policy; keywords go to the engine."""
    centreline = tmp_path / "centreline.csv"
    np.savetxt(centreline, SQUARE_POINTS, delimiter=",", header="x_m, y_m")

    def _make(
        sim: FakeSimulator, *, steer: float = 0.0, throttle: float = 0.5, **kwargs: Any
    ) -> InferenceEngine:
        def start(map_yaml: Path, *, seed: int) -> FakeSimulator:
            simulator_calls.append((map_yaml, seed))
            return sim

        monkeypatch.setattr(engine_module, "F1TenthSimulator", start)
        source = str(constant_policy(tmp_path, steer, throttle))
        return InferenceEngine(source, str(tmp_path / "track.yaml"), str(centreline), **kwargs)

    return _make


# Construction


def test_engine_starts_the_simulator_on_the_map_with_its_seed(
    make_engine: EngineFactory, simulator_calls: list[tuple[Path, int]], tmp_path: Path
) -> None:
    make_engine(FakeSimulator(), seed=7)
    assert simulator_calls == [(tmp_path / "track.yaml", 7)]


def test_invalid_friction_is_rejected_before_the_simulator_starts(
    make_engine: EngineFactory, simulator_calls: list[tuple[Path, int]]
) -> None:
    with pytest.raises(ValueError, match="friction_range"):
        make_engine(FakeSimulator(), friction=0.0)
    assert simulator_calls == []


def test_close_and_the_context_manager_stop_the_simulator(make_engine: EngineFactory) -> None:
    sim = FakeSimulator()
    make_engine(sim).close()
    assert sim.closed
    other = FakeSimulator()
    with make_engine(other):
        assert not other.closed
    assert other.closed


# Step by step


def test_reset_returns_the_car_at_rest_on_the_start_line_as_step_zero(
    make_engine: EngineFactory,
) -> None:
    frame = make_engine(CircuitSimulator()).reset()
    assert (frame.step, frame.t, frame.x, frame.y, frame.theta) == (0, 0.0, 0.0, 0.0, 0.0)
    assert (frame.steer, frame.throttle, frame.reward) == (0.0, 0.0, 0.0)
    assert (frame.progress, frame.lap_fraction, frame.lap_count) == (0.0, 0.0, 0)
    assert frame.friction_mu == NOMINAL_MU


def test_every_episode_runs_at_the_engines_friction(make_engine: EngineFactory) -> None:
    sim = FakeSimulator()
    engine = make_engine(sim, friction=0.6)
    assert engine.reset().friction_mu == 0.6
    assert sim.friction == 0.6


def test_step_without_an_action_drives_the_policys_action(make_engine: EngineFactory) -> None:
    sim = CircuitSimulator()
    engine = make_engine(sim, steer=0.25, throttle=0.5)
    engine.reset()
    frame, reward, terminated, truncated, info = engine.step()
    assert frame.steer == pytest.approx(0.25, abs=1e-6)
    assert frame.throttle == pytest.approx(0.5, abs=1e-6)
    assert (frame.step, frame.t) == (1, pytest.approx(CTRL_DT))
    assert frame.reward == reward
    assert not (terminated or truncated)
    assert info["steer_rad"] == frame.steer_rad


def test_predict_returns_the_policys_float32_action(make_engine: EngineFactory) -> None:
    engine = make_engine(FakeSimulator(), steer=-0.3, throttle=0.7)
    action = engine.predict(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (ACT_DIM,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, [-0.3, 0.7], atol=1e-6)


def test_step_with_an_action_drives_it_instead_of_the_policy(make_engine: EngineFactory) -> None:
    engine = make_engine(FakeSimulator(), steer=0.25, throttle=0.5)
    engine.reset()
    frame, *_ = engine.step(np.array([-0.5, 0.0]))
    assert (frame.steer, frame.throttle) == (-0.5, 0.0)


def test_out_of_range_actions_are_clipped_by_the_env(make_engine: EngineFactory) -> None:
    engine = make_engine(FakeSimulator())
    engine.reset()
    frame, *_ = engine.step([3.0, -3.0])
    assert (frame.steer, frame.throttle) == (1.0, -1.0)


def test_step_reports_the_raw_lateral_speed_beyond_the_observation_clip(
    make_engine: EngineFactory,
) -> None:
    sim = FakeSimulator()
    engine = make_engine(sim)
    engine.reset()
    sim.v_lat = 1.4 * VY_NORM
    frame, *_ = engine.step()
    assert frame.v_lat == 1.4 * VY_NORM
    assert frame.speed == math.hypot(frame.v_long, 1.4 * VY_NORM)


def test_step_before_reset_raises(make_engine: EngineFactory) -> None:
    with pytest.raises(RuntimeError, match="reset"):
        make_engine(FakeSimulator()).step()


def test_reward_is_the_training_racing_reward(make_engine: EngineFactory) -> None:
    frames = make_engine(CircuitSimulator(), steer=0.1).run_episode(max_steps=20).frames
    config = EnvConfig(friction_range=(NOMINAL_MU, NOMINAL_MU))
    reference = RacingReward(
        LapZeroEnv(CircuitSimulator(), START, config=config, centreline=Centreline(SQUARE_POINTS))
    )
    reference.reset(seed=0)
    for frame in frames[1:]:
        _, reward, *_ = reference.step(np.array([frame.steer, frame.throttle], dtype=np.float32))
        assert frame.reward == reward
    assert all(frame.reward > 0.0 for frame in frames[1:])


# Episodes


def test_a_full_lap_completes_the_episode(make_engine: EngineFactory) -> None:
    result = make_engine(CircuitSimulator()).run_episode()
    assert result.status is LapStatus.COMPLETED
    assert result.success
    assert (result.steps, result.lap_count, len(result.frames)) == (LAP_STEPS, 1, LAP_STEPS + 1)
    assert result.lap_time_s == pytest.approx(LAP_STEPS * CTRL_DT)
    assert not result.collision
    assert [frame.lap_count for frame in result.frames[:-1]] == [0] * LAP_STEPS
    assert result.frames[-1].lap_fraction == pytest.approx(1.0)


def test_a_start_line_pass_short_of_90_percent_of_the_lap_is_not_a_lap(
    make_engine: EngineFactory,
) -> None:
    sim = FakeSimulator()  # the car stays on the start line
    engine = make_engine(sim)
    engine.reset()
    sim.lap_count = 1  # F1TENTH reports a pass, e.g. after a spin back through the line
    frame, _, terminated, truncated, _ = engine.step()
    assert frame.lap_count == 0
    assert not (terminated or truncated)


def test_a_collision_ends_the_episode_as_crashed(make_engine: EngineFactory) -> None:
    sim = CircuitSimulator()
    sim.collision_at_step = 10  # physics step 10 falls in control step 3
    result = make_engine(sim).run_episode()
    assert result.status is LapStatus.CRASHED
    assert (result.collision, result.steps, result.lap_time_s) == (True, 3, None)
    assert not result.success


def test_a_car_that_stops_is_reported_stuck(make_engine: EngineFactory) -> None:
    sim = FakeSimulator()
    sim.follow_target = False  # the car never moves off the line
    result = make_engine(sim).run_episode()
    assert result.status is LapStatus.STUCK
    assert result.episode_time_s == pytest.approx(STUCK_S)


def test_the_envs_time_limit_ends_the_episode_as_a_timeout(make_engine: EngineFactory) -> None:
    result = make_engine(FakeSimulator()).run_episode()  # drives on the spot, never stuck
    assert result.status is LapStatus.TIMEOUT
    assert result.episode_time_s == pytest.approx(MAX_EPISODE_S)
    assert result.steps < DEFAULT_MAX_STEPS


def test_max_steps_cuts_the_episode_short_as_a_timeout(make_engine: EngineFactory) -> None:
    engine = make_engine(CircuitSimulator())
    result = engine.run_episode(max_steps=10)
    assert result.status is LapStatus.TIMEOUT
    assert (result.steps, len(result.frames), result.lap_time_s) == (10, 11, None)
    with pytest.raises(RuntimeError, match="episode ends"):
        engine.step()


def test_stream_episode_yields_every_frame_and_returns_the_result(
    make_engine: EngineFactory,
) -> None:
    episode = make_engine(CircuitSimulator()).stream_episode(max_steps=5)
    yielded = []
    while True:
        try:
            yielded.append(next(episode))
        except StopIteration as end:
            result = end.value
            break
    assert [frame.step for frame in yielded] == [0, 1, 2, 3, 4, 5]
    assert tuple(yielded) == result.frames


def test_stream_episode_rejects_max_steps_below_one_before_the_episode_starts(
    make_engine: EngineFactory,
) -> None:
    sim = FakeSimulator()
    engine = make_engine(sim)
    with pytest.raises(ValueError, match="max_steps must be >= 1, got 0"):
        engine.stream_episode(max_steps=0)
    assert sim.reset_poses == []


def test_episodes_are_seeded_in_order_from_the_engine_seed(make_engine: EngineFactory) -> None:
    sim = FakeSimulator()
    engine = make_engine(sim, seed=7)
    engine.reset(randomize_start=True)  # seed 7
    engine.reset(randomize_start=True)  # seed 8
    engine.reset(randomize_start=True, seed=7)
    first, second, again = sim.reset_poses
    assert first != second
    assert again == first
    assert first != START


def test_randomize_start_can_be_turned_back_off(make_engine: EngineFactory) -> None:
    sim = FakeSimulator()
    engine = make_engine(sim)
    engine.run_episode(max_steps=1, randomize_start=True)
    engine.run_episode(max_steps=1)
    jittered, fixed = sim.reset_poses
    assert jittered != START
    assert fixed == START


# Policy sources


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    return (tmp_path / "mlruns").as_uri()


@pytest.fixture
def registered(tracking_uri: str, tmp_path: Path) -> list[tuple[str, Path]]:
    """Two registered versions of MODEL, champion then challenger, as (run_id, checkpoint)."""
    runs = []
    for steer in (-0.2, 0.2):
        checkpoint = constant_policy(tmp_path, steer, 0.5)
        with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
            tracker.register_policy(checkpoint, MODEL, {"algorithm": "bc"})
        runs.append((tracker.run_id, checkpoint))
    return runs


def _steer(policy: torch.nn.Module) -> float:
    with torch.no_grad():
        return float(policy(torch.zeros(OBS_DIM))[0])


def test_a_local_checkpoint_loads_from_its_path(tmp_path: Path) -> None:
    path = constant_policy(tmp_path, 0.4, 0.5)
    assert _steer(load_policy_source(str(path))) == pytest.approx(0.4, abs=1e-6)


def test_a_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.pt"):
        load_policy_source(str(tmp_path / "missing.pt"))


@pytest.mark.parametrize(("alias", "steer"), [("champion", -0.2), ("challenger", 0.2)])
def test_a_bare_alias_loads_that_version_of_the_registered_model(
    registered: list[tuple[str, Path]], tracking_uri: str, alias: str, steer: float
) -> None:
    policy = load_policy_source(alias, tracking_uri=tracking_uri, model_name=MODEL)
    assert _steer(policy) == pytest.approx(steer, abs=1e-6)


def test_a_models_uri_names_the_registered_model_and_alias(
    registered: list[tuple[str, Path]], tracking_uri: str
) -> None:
    policy = load_policy_source(f"models:/{MODEL}@challenger", tracking_uri=tracking_uri)
    assert _steer(policy) == pytest.approx(0.2, abs=1e-6)


def test_a_run_id_loads_the_version_that_run_registered(
    registered: list[tuple[str, Path]], tracking_uri: str
) -> None:
    run_id = registered[0][0]
    policy = load_policy_source(run_id, tracking_uri=tracking_uri, model_name=MODEL)
    assert _steer(policy) == pytest.approx(-0.2, abs=1e-6)


def test_a_runs_uri_loads_the_logged_policy_model(
    registered: list[tuple[str, Path]], tracking_uri: str
) -> None:
    run_id = registered[1][0]
    policy = load_policy_source(f"runs:/{run_id}/policy", tracking_uri=tracking_uri)
    assert _steer(policy) == pytest.approx(0.2, abs=1e-6)


def test_a_run_that_registered_no_policy_raises(tracking_uri: str) -> None:
    with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
        pass
    with pytest.raises(ValueError, match=f"run {tracker.run_id} registered no version"):
        load_policy_source(tracker.run_id, tracking_uri=tracking_uri, model_name=MODEL)


def test_a_runs_uri_that_is_not_a_policy_model_raises(tracking_uri: str, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "MLmodel").write_text("flavors:\n  python_function: {}\n")
    with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
        mlflow_client(tracking_uri).log_artifacts(tracker.run_id, str(other), "other")
    with pytest.raises(ValueError, match="is not a lapzero_policy model"):
        load_policy_source(f"runs:/{tracker.run_id}/other", tracking_uri=tracking_uri)


def test_registry_sources_default_to_the_repo_tracking_store(
    registered: list[tuple[str, Path]], tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "get_tracking_uri", lambda: tracking_uri)
    assert _steer(load_policy_source("champion", model_name=MODEL)) == pytest.approx(-0.2, abs=1e-6)


@pytest.mark.parametrize(
    "source", ["best", "champion2", "models:/lapzero-policy", "runs:/abc/policy", "z" * 32]
)
def test_an_unrecognised_source_raises(source: str) -> None:
    with pytest.raises(ValueError, match=r"is not a \.pt path"):
        load_policy_source(source, tracking_uri="file:///unused")
