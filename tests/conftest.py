"""Shared test setup: a scripted fake simulator so contract and unit tests never start F1TENTH.

Hypothesis runs derandomized (fixed examples, no example database) to keep tests deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import settings
from numpy.typing import NDArray

from lapzero.contract import RAW_BEAMS
from lapzero.io.telemetry import (
    DriverInfo,
    DriverType,
    InputDevice,
    TrackDirection,
    TrackInfo,
)
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.sim.env import EnvConfig, LapZeroEnv, RewardFn, zero_reward
from lapzero.sim.simulator import Pose, SimState

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


@pytest.fixture
def fake_sim() -> FakeSimulator:
    return FakeSimulator()


@pytest.fixture
def make_env(fake_sim: FakeSimulator) -> Callable[..., LapZeroEnv]:
    """Factory for a LapZeroEnv on `fake_sim`; keyword arguments go to EnvConfig."""

    def _make(reward_fn: RewardFn = zero_reward, **config: Any) -> LapZeroEnv:
        return LapZeroEnv(fake_sim, START_POSE, reward_fn=reward_fn, config=EnvConfig(**config))

    return _make
