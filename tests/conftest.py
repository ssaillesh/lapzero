"""Shared test setup: a scripted fake simulator so contract and unit tests never start F1TENTH.

Hypothesis runs derandomized (fixed examples, no example database) to keep tests deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import settings
from numpy.typing import NDArray

from lapzero.contract import CONTRACT_VERSION, CTRL_DT, RAW_BEAMS
from lapzero.eval.sweep import SWEEP_FILE, LevelReport, SweepReport, SweepRun, friction_grid
from lapzero.io.telemetry import (
    DriverInfo,
    DriverType,
    InputDevice,
    TrackDirection,
    TrackInfo,
)
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.record.ulid import new_ulid
from lapzero.sim.env import EnvConfig, LapZeroEnv, RewardFn, zero_reward
from lapzero.sim.simulator import Pose, SimState
from lapzero.training.tracking import mlflow_client

settings.register_profile("lapzero", derandomize=True, database=None, deadline=None)
settings.load_profile("lapzero")

START_POSE = Pose(1.0, 2.0, 0.5)
HUMAN_DRIVER = DriverInfo(
    type=DriverType.HUMAN, name="tester", human_input_device=InputDevice.KEYBOARD
)

_WILD_RANGES = np.array([np.nan, np.inf, -np.inf, -5.0, 0.0, 1e6])


class FakeSimulator:
    """Scripted stand-in for F1TENTH Gym implementing the Simulator protocol.

    Tests set the public attributes to script the next readings, and read `commands` to see the
    (steer_rad, v_target) pairs the env sent. With `rng` set, every reading is randomized with
    extreme values (NaN/inf ranges, huge speeds) to stress the observation pipeline.
    """

    def __init__(self) -> None:
        self.scan: NDArray[np.float64] = np.full(RAW_BEAMS, 10.0)
        self.v_long = 0.0
        self.v_lat = 0.0
        self.yaw_rate = 0.0
        self.lap_count = 0
        self.friction = 1.0
        self.follow_target = True  # car reaches the commanded speed every physics step
        self.collision_at_step: int | None = None  # physics step (since reset) that collides
        self.rng: np.random.Generator | None = None
        self.pose = START_POSE
        self.steps = 0
        self.commands: list[tuple[float, float]] = []
        self.reset_poses: list[Pose] = []
        self.closed = False

    def reset(self, pose: Pose) -> SimState:
        self.reset_poses.append(pose)
        self.pose = pose
        self.steps = 0
        return self._reading()

    def step(self, steer_rad: float, v_target: float) -> SimState:
        self.commands.append((steer_rad, v_target))
        self.steps += 1
        if self.follow_target:
            self.v_long = v_target
        return self._reading()

    def set_friction(self, mu: float) -> None:
        self.friction = mu

    def close(self) -> None:
        self.closed = True

    def _reading(self) -> SimState:
        collision = self.collision_at_step is not None and self.steps >= self.collision_at_step
        if self.rng is not None:
            wild = self.rng.choice(_WILD_RANGES, size=RAW_BEAMS)
            scan = np.where(
                self.rng.random(RAW_BEAMS) < 0.3, wild, self.rng.uniform(0, 60, RAW_BEAMS)
            )
            v_long, v_lat, yaw_rate = self.rng.uniform(-1e3, 1e3, size=3)
            return SimState(
                scan, self.pose, float(v_long), float(v_lat), float(yaw_rate), collision, 0
            )
        return SimState(
            scan=self.scan.copy(),
            pose=self.pose,
            v_long=self.v_long,
            v_lat=self.v_lat,
            yaw_rate=self.yaw_rate,
            collision=collision,
            lap_count=self.lap_count,
        )


def make_state(**overrides: Any) -> SimState:
    """A plausible SimState (car at rest at the origin, walls 10 m away) with field overrides."""
    fields: dict[str, Any] = {
        "scan": np.full(RAW_BEAMS, 10.0),
        "pose": Pose(0.0, 0.0, 0.0),
        "v_long": 0.0,
        "v_lat": 0.0,
        "yaw_rate": 0.0,
        "collision": False,
        "lap_count": 0,
    }
    return SimState(**(fields | overrides))


def make_sweep_report(
    policy_version: str | None,
    *,
    alias: str = "champion",
    sweep_id: str | None = None,
    track_id: str = "oschersleben",
    nominal_lap_s: float = 46.56,
) -> SweepReport:
    """A release-gate friction sweep of lapzero-policy@`alias` at `policy_version`.

    policy_version None means the sweep evaluated a local checkpoint. Every episode on the 0.50 to
    1.05 grid (10 per level) completes; laps at mu 1.05 take `nominal_lap_s`, elsewhere 48.0 s.
    """
    levels = tuple(
        LevelReport(
            friction_mu=mu,
            episodes=10,
            completed=10,
            crashed=0,
            completion_rate=1.0,
            mean_lap_time_s=nominal_lap_s if math.isclose(mu, 1.05) else 48.0,
            eval_dir=f"mu_{mu:g}",
        )
        for mu in friction_grid(0.50, 1.05, 0.05)
    )
    run = SweepRun(
        sweep_id=sweep_id or new_ulid(),
        git_commit="abc123",
        policy=f"lapzero-policy@{alias}" if policy_version is not None else "policy.pt",
        policy_version=policy_version,
        track_id=track_id,
        map_yaml=f"data/tracks/{track_id}/{track_id}.yaml",
        start_pose=(0.0, 0.0, 2.857),
        seed=0,
        episodes_per_level=10,
        randomize_start=True,
    )
    return SweepReport(run=run, levels=levels, crashes=())


def register_model_versions(
    tracking_uri: str, model_name: str, versions: int, aliases: Mapping[str, str]
) -> None:
    """Register `versions` bare model versions (no weights) tagged algorithm=ppo, then set
    `aliases` (alias to version number)."""
    client = mlflow_client(tracking_uri)
    client.create_registered_model(model_name)
    for i in range(versions):
        client.create_model_version(model_name, f"file:///weights/{i}", tags={"algorithm": "ppo"})
    for alias, version in aliases.items():
        client.set_registered_model_alias(model_name, alias, version)


def write_sweep_report(root: Path, report: SweepReport) -> Path:
    """Write `report` as sweep.json in root/<track_id>/<sweep_id>/, like write_sweep; return it."""
    directory = root / report.run.track_id / report.run.sweep_id
    directory.mkdir(parents=True)
    (directory / SWEEP_FILE).write_text(report.model_dump_json(indent=2) + "\n")
    return directory


def record_lap(
    root: Path,
    *,
    track_id: str = "track_a",
    driver: DriverInfo = HUMAN_DRIVER,
    steps: int = 5,
    crash: bool = False,
    seed: int = 0,
    policy: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
) -> LapFiles:
    """Record one lap on a fresh FakeSimulator via LapRecorder.

    The scan is a seeded random (constant) wall layout. Actions come from `policy(obs)`, or are
    seeded uniform noise when no policy is given. The last of `steps` control steps crosses the
    start line, or collides when `crash` is set, so exactly one completed (or crashed) lap is
    written under root/<track_id>/.
    """
    rng = np.random.default_rng(seed)
    sim = FakeSimulator()
    sim.scan = rng.uniform(2.0, 30.0, RAW_BEAMS)
    config = EnvConfig(laps_per_episode=1)
    env = LapZeroEnv(sim, START_POSE, config=config)
    track = TrackInfo(track_id=track_id, direction=TrackDirection.CCW, length_m=50.0)
    recorder = LapRecorder(track, driver, sim_version="test", env_config=config, root=root)
    obs, info = env.reset(seed=seed)
    recorder.start_episode(obs, info, seed=seed)
    written: list[LapFiles] = []
    for step in range(steps):
        if step == steps - 1 and crash:
            sim.collision_at_step = sim.steps + 1
        elif step == steps - 1:
            sim.lap_count = 1
        if policy is None:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        else:
            action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        written += recorder.record_step(action, obs, reward, terminated, truncated, info)
    (files,) = written
    return files


def fixed_ulid(n: int) -> str:
    """A constant, valid ULID numbered `n` (0 to 99), for deterministic lap IDs."""
    return f"01J9ZK8V3Q7M2T6C1R4X5N8B{n:02d}"


def model_driver_json(version: str = "7", name: str = "lapzero-ppo") -> dict[str, Any]:
    """The telemetry driver block of a registered model version."""
    return {
        "type": "model",
        "name": name,
        "model_version": version,
        "mlflow_run_id": "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
        "human_input_device": None,
    }


def human_driver_json(name: str = "sam") -> dict[str, Any]:
    """The telemetry driver block of a human on a keyboard."""
    return {
        "type": "human",
        "name": name,
        "model_version": None,
        "mlflow_run_id": None,
        "human_input_device": "keyboard",
    }


def lap_telemetry_document(
    lap_id: str = "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
    *,
    track_id: str = "track_02",
    driver: dict[str, Any] | None = None,
    lap_time_s: float | None = 35.97,
    created_at: str = "2026-10-20T14:32:05Z",
    contract_version: str = CONTRACT_VERSION,
    frames: int = 26,
) -> dict[str, Any]:
    """A valid contract Section 5 lap telemetry JSON document, as a lap upload body.

    The car drives along x at 2 m/s for `frames` 25 Hz samples. lap_time_s None makes the lap
    a crash; driver defaults to model version 7.
    """
    t = [round(i * CTRL_DT, 2) for i in range(frames)]
    zeros = [0.0] * frames
    return {
        "schema": "lapzero.lap_telemetry",
        "schema_version": "1.0.0",
        "contract_version": contract_version,
        "lap_id": lap_id,
        "episode_id": "01J9ZK8T9E0Y3H2P6W1D7F4G5S",
        "lap_index": 1,
        "created_at": created_at,
        "track": {"track_id": track_id, "direction": "ccw", "length_m": 82.4},
        "driver": driver or model_driver_json(),
        "sim": {
            "simulator": "f1tenth_gym",
            "sim_version": "0.2.1",
            "sim_dt": 0.01,
            "action_repeat": 4,
            "ctrl_hz": 25,
            "v_max_train": 8.0,
            "seed": 42,
        },
        "result": {
            "status": "completed" if lap_time_s is not None else "crashed",
            "lap_time_s": lap_time_s,
            "collision": lap_time_s is None,
            "off_track_events": 0,
            "max_speed_mps": 2.0,
            "mean_speed_mps": 2.0,
            "sample_count": frames,
        },
        "frames": {
            "t": t,
            "x": [round(2.0 * ti, 3) for ti in t],
            "y": zeros,
            "theta": zeros,
            "v_long": [2.0] * frames,
            "steer": zeros,
            "throttle": [0.5] * frames,
            "steer_rad": zeros,
            "v_target": [2.0] * frames,
            "progress": [round(i / frames, 4) for i in range(frames)],
        },
    }


@pytest.fixture
def fake_sim() -> FakeSimulator:
    return FakeSimulator()


@pytest.fixture
def make_env(fake_sim: FakeSimulator) -> Callable[..., LapZeroEnv]:
    """Factory for a LapZeroEnv on `fake_sim`; keyword arguments go to EnvConfig."""

    def _make(reward_fn: RewardFn = zero_reward, **config: Any) -> LapZeroEnv:
        return LapZeroEnv(fake_sim, START_POSE, reward_fn=reward_fn, config=EnvConfig(**config))

    return _make
