"""Unit tests for lapzero/record/session.py: the real-time human driving loop.

LapZeroEnv runs on the scripted FakeSimulator, laps are written to a temporary directory and wall
time comes from a fake clock, so nothing here waits, opens a window or starts F1TENTH.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import pytest

from lapzero.contract import CTRL_DT, STEER_MAX
from lapzero.io.telemetry import (
    DriverType,
    InputDevice,
    LapStatus,
    TrackDirection,
    TrackInfo,
    read_lap_telemetry,
)
from lapzero.record.driver import InputFrame
from lapzero.record.pygame_input import PygameInput
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.record.session import drive_session
from lapzero.sim.env import EnvConfig, LapZeroEnv
from tests.conftest import HUMAN_DRIVER, START_POSE, FakeSimulator

TRACK = TrackInfo(track_id="track_02", direction=TrackDirection.CCW, length_m=82.4)
SEED = 11
GO = InputFrame(0.0, 1.0)
GO_LEFT = InputFrame(1.0, 1.0)
COAST = InputFrame(0.0, 0.0)
QUIT = InputFrame(quit=True)
RESTART = InputFrame(restart=True)


class ScriptedSource:
    """InputSource replaying frames, then quitting. `on_poll(n)` runs before poll n (from 1)."""

    def __init__(
        self, frames: Iterable[InputFrame], on_poll: Callable[[int], None] | None = None
    ) -> None:
        self.frames = iter(frames)
        self.on_poll = on_poll
        self.polls = 0

    def poll(self) -> InputFrame:
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        return next(self.frames, QUIT)


class FakeClock:
    """Monotonic clock that only advances when slept on (or advanced by a test)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Rig:
    """An env on the fake simulator, a recorder writing to tmp_path, and a fake clock."""

    def __init__(self, fake_sim: FakeSimulator, root: Path, **config: Any) -> None:
        env_config = EnvConfig(**config)
        self.sim = fake_sim
        self.env = LapZeroEnv(fake_sim, START_POSE, config=env_config)
        self.recorder = LapRecorder(
            TRACK, HUMAN_DRIVER, sim_version="test", env_config=env_config, root=root
        )
        self.clock = FakeClock()
        self.renders = 0

    def render(self) -> None:
        self.renders += 1

    def run(self, source: Any, **kwargs: Any) -> list[LapFiles]:
        return drive_session(
            self.env,
            self.recorder,
            source,
            seed=SEED,
            render=self.render,
            clock=self.clock,
            sleep=self.clock.sleep,
            **kwargs,
        )


@pytest.fixture
def rig(fake_sim: FakeSimulator, tmp_path: Path) -> Rig:
    return Rig(fake_sim, tmp_path)


def _finish_lap_at(sim: FakeSimulator, poll: int) -> Callable[[int], None]:
    """on_poll hook: the control step taken on `poll` crosses the start line."""

    def hook(n: int) -> None:
        if n == poll:
            sim.lap_count += 1

    return hook


# Live keyboard session end to end


class HeldKeys:
    """Held key set indexed like pygame.key.get_pressed()."""

    def __init__(self, held: set[int]) -> None:
        self.held = held

    def __getitem__(self, key: int) -> bool:
        return key in self.held


class HookedSource:
    """Wraps an InputSource to run `hook(n)` before poll n (from 1), like ScriptedSource."""

    def __init__(self, source: PygameInput, hook: Callable[[int], None]) -> None:
        self.source, self.hook, self.polls = source, hook, 0

    def poll(self) -> InputFrame:
        self.polls += 1
        self.hook(self.polls)
        return self.source.poll()


def test_live_keyboard_session_writes_a_valid_lap_with_ramped_actions(
    fake_sim: FakeSimulator, tmp_path: Path
) -> None:
    rig = Rig(fake_sim, tmp_path, laps_per_episode=1)
    held = [{pygame.K_UP, pygame.K_LEFT}] * 4 + [{pygame.K_UP}] * 3
    polls = iter(held)
    keyboard = PygameInput(get_pressed=lambda: HeldKeys(next(polls)), pump=lambda: None)

    (files,) = rig.run(HookedSource(keyboard, _finish_lap_at(fake_sim, len(held))), max_episodes=1)

    assert files.status is LapStatus.COMPLETED
    lap = read_lap_telemetry(files.json_path)
    assert lap.driver.type is DriverType.HUMAN
    assert lap.driver.human_input_device is InputDevice.KEYBOARD
    assert lap.sim.seed == SEED
    assert lap.result.sample_count == 7
    # Turn in at 0.06 per step while Left is held, return at 0.15 per step once released.
    assert lap.frames.steer == pytest.approx([0.06, 0.12, 0.18, 0.24, 0.09, 0.0, 0.0], abs=1e-3)
    assert lap.frames.throttle == [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # The ramp equals the env rate limit, so the applied angle is the recorded action's angle.
    applied = np.asarray(lap.frames.steer_rad)
    assert applied == pytest.approx(np.asarray(lap.frames.steer) * STEER_MAX, abs=1e-3)
    with np.load(files.npz_path) as arrays:
        assert arrays["actions"].shape == (7, 2)
        assert arrays["actions"].dtype == np.float32
        np.testing.assert_allclose(arrays["actions"][:, 0], lap.frames.steer, atol=5e-4)
        assert ((arrays["actions"] >= -1.0) & (arrays["actions"] <= 1.0)).all()
        assert np.isfinite(arrays["obs"]).all()
    raw = files.json_path.read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert {len(column) for column in json.loads(raw)["frames"].values()} == {7}


# Grid start, restart and quit


def test_car_waits_on_the_grid_until_throttle_is_pressed(rig: Rig) -> None:
    rig.run(ScriptedSource([COAST, COAST, InputFrame(1.0, 0.0), GO, GO]))
    assert len(rig.sim.commands) == 2 * 4  # only the two throttle frames stepped the env


def test_brake_does_not_start_the_car(rig: Rig) -> None:
    rig.run(ScriptedSource([InputFrame(0.0, -1.0)] * 3))
    assert rig.sim.commands == []


def test_quit_before_starting_writes_nothing(rig: Rig, tmp_path: Path) -> None:
    assert rig.run(ScriptedSource([COAST])) == []
    assert rig.sim.commands == []
    assert not any(tmp_path.iterdir())


def test_quit_mid_lap_discards_the_unfinished_lap(rig: Rig, tmp_path: Path) -> None:
    assert rig.run(ScriptedSource([GO] * 5 + [QUIT, GO])) == []
    assert len(rig.sim.commands) == 5 * 4
    assert not any(tmp_path.iterdir())


def test_restart_discards_the_lap_and_reseeds_the_next_episode_from_the_grid(
    fake_sim: FakeSimulator, tmp_path: Path
) -> None:
    rig = Rig(fake_sim, tmp_path, laps_per_episode=1)
    frames = [GO] * 3 + [RESTART, COAST] + [GO] * 2
    source = ScriptedSource(frames, on_poll=_finish_lap_at(fake_sim, len(frames)))
    (files,) = rig.run(source, max_episodes=2)
    lap = read_lap_telemetry(files.json_path)
    assert len(fake_sim.reset_poses) == 2
    assert lap.sim.seed == SEED + 1
    assert lap.lap_index == 1
    assert lap.result.sample_count == 2
    assert lap.frames.t[0] == 0.0
    assert lap.frames.throttle == [0.5, 1.0]  # the driver was reset with the episode


def test_crash_writes_a_crashed_lap_and_the_next_episode_waits_on_the_grid(rig: Rig) -> None:
    def crash_on_third(n: int) -> None:
        rig.sim.collision_at_step = rig.sim.steps + 1 if n == 3 else None

    files = rig.run(ScriptedSource([GO] * 3 + [COAST] * 4, on_poll=crash_on_third))
    assert [lap.status for lap in files] == [LapStatus.CRASHED]
    assert len(rig.sim.reset_poses) == 2
    assert len(rig.sim.commands) == 2 * 4 + 1  # crash stopped the third step after one tick


def test_max_episodes_ends_the_session_after_that_many_resets(rig: Rig) -> None:
    rig.run(ScriptedSource([RESTART] * 5), max_episodes=3)
    assert len(rig.sim.reset_poses) == 3


# Real-time pacing


def test_steps_are_paced_one_control_period_apart(rig: Rig) -> None:
    def busy_render() -> None:
        rig.clock.now += 0.01  # drawing takes 10 ms of the 40 ms period

    rig.render = busy_render  # type: ignore[method-assign]
    rig.run(ScriptedSource([GO] * 4))
    assert rig.clock.sleeps == pytest.approx([CTRL_DT - 0.01] * 4)


def test_a_late_step_is_not_followed_by_a_burst_of_catch_up_steps(rig: Rig) -> None:
    def slow_once(n: int) -> None:
        if n == 2:
            rig.clock.now += 0.25  # a 250 ms stall

    rig.run(ScriptedSource([GO] * 4, on_poll=slow_once))
    assert rig.clock.sleeps == pytest.approx([CTRL_DT, CTRL_DT, CTRL_DT])


def test_every_control_step_is_drawn_including_on_the_grid(rig: Rig) -> None:
    rig.run(ScriptedSource([COAST, COAST, GO, GO]))
    assert rig.renders == 5  # four frames, plus the poll that quit
