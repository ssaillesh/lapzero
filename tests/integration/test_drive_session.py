"""Integration tests: a recorded human driving session on the real F1TENTH Gym v1 simulator.

The F1TENTH render window opens on SDL's headless dummy video driver. Input is scripted keyboard
state, since no one can press keys in CI. The track is a synthetic 20 m x 20 m walled room centred
on the origin (inner wall faces 9.7 m from it), so no map download is needed.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pygame
import pytest

from lapzero.contract import OBS_HIGH, OBS_LOW, STEER_MAX
from lapzero.io.lap_arrays import read_lap_arrays
from lapzero.io.telemetry import (
    DriverType,
    InputDevice,
    LapStatus,
    TrackDirection,
    TrackInfo,
    read_lap_telemetry,
)
from lapzero.record.driver import InputFrame, KeyboardState, keyboard_target
from lapzero.record.pygame_input import open_input
from lapzero.record.recorder import LapRecorder
from lapzero.record.session import drive_session
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.f1tenth import RENDER_MIRRORED, F1TenthSimulator
from lapzero.sim.simulator import Pose
from tests.conftest import HUMAN_DRIVER

pytestmark = pytest.mark.integration

TRACK = TrackInfo(track_id="room", direction=TrackDirection.CCW, length_m=40.0)


@pytest.fixture(scope="module")
def room_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a ROS map (PGM + yaml) of an empty room with 0.3 m thick walls."""
    directory = tmp_path_factory.mktemp("room")
    size_px, wall_px = 400, 6  # 0.05 m per pixel
    pixels = np.full((size_px, size_px), 255, dtype=np.uint8)
    pixels[:wall_px, :] = pixels[-wall_px:, :] = 0
    pixels[:, :wall_px] = pixels[:, -wall_px:] = 0
    header = f"P5\n{size_px} {size_px}\n255\n".encode()
    (directory / "room.pgm").write_bytes(header + pixels.tobytes())
    (directory / "room.yaml").write_text(
        "image: room.pgm\nresolution: 0.05\norigin: [-10.0, -10.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    return directory / "room.yaml"


@pytest.fixture
def windowed_sim(room_map: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[F1TenthSimulator]:
    """F1TENTH with its pygame render window open, headless."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    simulator = F1TenthSimulator(room_map, seed=0, render=True)
    yield simulator
    simulator.close()


class KeyScript:
    """InputSource replaying held keyboard states, then quitting."""

    def __init__(self, held: list[KeyboardState]) -> None:
        self.held = iter(held)

    def poll(self) -> InputFrame:
        keys = next(self.held, None)
        return InputFrame(quit=True) if keys is None else InputFrame(*keyboard_target(keys))


def test_keyboard_session_into_a_wall_records_a_valid_crashed_lap(
    windowed_sim: F1TenthSimulator, tmp_path: Path
) -> None:
    config = EnvConfig(laps_per_episode=1)
    env = LapZeroEnv(windowed_sim, Pose(0.0, 0.0, 0.0), config=config)
    recorder = LapRecorder(
        TRACK, HUMAN_DRIVER, sim_version="test", env_config=config, root=tmp_path
    )
    turn_then_floor_it = [KeyboardState(left=True, throttle=True)] * 8
    turn_then_floor_it += [KeyboardState(throttle=True)] * 300

    files = drive_session(
        env,
        recorder,
        KeyScript(turn_then_floor_it),
        seed=3,
        render=windowed_sim.render,
        max_episodes=1,
        sleep=lambda seconds: None,  # run faster than real time
    )

    (lap_files,) = files
    assert lap_files.status is LapStatus.CRASHED
    lap = read_lap_telemetry(lap_files.json_path)
    assert lap.driver.type is DriverType.HUMAN
    assert lap.driver.human_input_device is InputDevice.KEYBOARD
    assert lap.result.collision is True
    assert lap.sim.seed == 3
    steer = np.asarray(lap.frames.steer)
    assert steer[:3] == pytest.approx([0.06, 0.12, 0.18], abs=1e-3)
    assert np.asarray(lap.frames.steer_rad) == pytest.approx(steer * STEER_MAX, abs=1e-3)
    assert max(lap.frames.v_target) <= config.v_max
    arrays = read_lap_arrays(lap_files.npz_path)
    assert arrays.sample_count == lap.result.sample_count
    assert ((arrays.obs >= OBS_LOW) & (arrays.obs <= OBS_HIGH)).all()
    assert ((arrays.actions >= -1.0) & (arrays.actions <= 1.0)).all()


def test_keyboard_input_reads_key_state_from_the_f1tenth_window(
    windowed_sim: F1TenthSimulator,
) -> None:
    LapZeroEnv(windowed_sim, Pose(0.0, 0.0, 0.0)).reset(seed=0)
    windowed_sim.render()  # the renderer drains pygame's event queue; key state must survive it
    assert open_input(InputDevice.KEYBOARD).poll() == InputFrame(0.0, 0.0)


def test_f1tenth_window_draws_the_world_mirrored_as_render_mirrored_says(
    windowed_sim: F1TenthSimulator,
) -> None:
    # Car at (0, -8) heading +x: the south wall is 1.7 m away on the car's right (world -y).
    # The window follows the car, so the car is at its centre; 30 px per metre at default zoom.
    LapZeroEnv(windowed_sim, Pose(0.0, -8.0, 0.0)).reset(seed=0)
    windowed_sim.render()
    frame = pygame.surfarray.array3d(pygame.display.get_surface())  # indexed [x, y, rgb]
    centre = frame.shape[0] // 2
    black_rows = np.flatnonzero(frame[centre].sum(axis=1) == 0)
    wall_rows = black_rows[np.abs(black_rows - centre) > 15]  # skip the car's own outline
    nearest_wall = wall_rows[np.argmin(np.abs(wall_rows - centre))]
    south_wall_drawn_above_car = bool(nearest_wall < centre)
    assert south_wall_drawn_above_car == RENDER_MIRRORED
