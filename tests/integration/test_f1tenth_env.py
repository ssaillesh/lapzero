"""Integration tests: LapZeroEnv rollouts on the real F1TENTH Gym v1 simulator.

Uses a synthetic 20 m x 20 m walled room centred on the origin, so no map download is needed.
Inner wall faces are 9.7 m from the origin.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from lapzero.io.telemetry import LapStatus, TrackDirection, TrackInfo
from lapzero.record.recorder import LapRecorder
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.f1tenth import F1TenthSimulator, make_env
from lapzero.sim.simulator import Pose
from tests.conftest import HUMAN_DRIVER

pytestmark = pytest.mark.integration

FULL_THROTTLE = np.array([0.0, 1.0], dtype=np.float32)


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
def centre_env(room_map: Path) -> Iterator[LapZeroEnv]:
    env = make_env(room_map, Pose(0.0, 0.0, 0.0), seed=0)
    yield env
    env.close()


def test_full_throttle_into_a_wall_ends_in_collision_with_contract_observations(
    centre_env: LapZeroEnv,
) -> None:
    obs, info = centre_env.reset(seed=0)
    assert centre_env.observation_space.contains(obs)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        obs, _, terminated, truncated, info = centre_env.step(FULL_THROTTLE)
        steps += 1
        assert obs.shape == (113,)
        assert obs.dtype == np.float32
        assert centre_env.observation_space.contains(obs)
        assert 0.0 <= info["v_target"] <= 8.0
    assert terminated
    assert info["collision"] is True
    assert 8.5 < info["x"] < 9.7
    assert info["v_long"] <= 8.0 + 1e-3


def test_beam_zero_side_is_the_cars_right_and_full_left_steer_turns_ccw(room_map: Path) -> None:
    env = make_env(room_map, Pose(0.0, -8.0, 0.0), seed=0)  # wall 1.7 m to the car's right
    try:
        obs, _ = env.reset(seed=0)
        right, left = obs[17], obs[90]  # beams pointing about -90 deg and +90 deg
        assert right < 0.1
        assert left > 0.5
        for _ in range(10):
            obs, *_ = env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert obs[110] > 0.0  # yaw rate CCW positive while steering full left
    finally:
        env.close()


def test_a_lap_after_a_reset_counts_from_zero_and_is_recorded(
    room_map: Path, tmp_path: Path
) -> None:
    # F1TENTH builds its reset observation before recounting laps, so without care the next
    # episode would start at the previous episode's lap count and its lap would go unrecorded.
    config = EnvConfig(laps_per_episode=1)
    env = make_env(room_map, Pose(0.0, 0.0, 0.0), seed=0, config=config)
    track = TrackInfo(track_id="room", direction=TrackDirection.CCW, length_m=4.0)
    recorder = LapRecorder(
        track, HUMAN_DRIVER, sim_version="test", env_config=config, root=tmp_path
    )
    statuses = []
    try:
        for episode in range(2):
            obs, info = env.reset(seed=episode)
            assert info["lap_count"] == 0
            recorder.start_episode(obs, info, seed=episode)
            done, steps = False, 0
            while not done and steps < 500:
                throttle = np.clip((1.5 - info["v_long"]) / 0.6, -1.0, 1.0)  # hold about 1.5 m/s
                action = np.array([1.0, throttle], dtype=np.float32)  # tight circle to the left
                obs, reward, terminated, truncated, info = env.step(action)
                laps = recorder.record_step(action, obs, reward, terminated, truncated, info)
                statuses += [lap.status for lap in laps]
                done, steps = terminated or truncated, steps + 1
    finally:
        env.close()
    assert statuses == [LapStatus.COMPLETED, LapStatus.COMPLETED]


def _heading_change_in_a_full_left_turn(
    room_map: Path, friction: float | None, config: EnvConfig | None = None
) -> float:
    """Accelerate straight for 35 control steps, steer full left for 12; return the turn, rad."""
    simulator = F1TenthSimulator(room_map, seed=0, friction=friction)
    env = LapZeroEnv(simulator, Pose(-8.0, -4.0, 0.0), config=config)
    try:
        env.reset(seed=0)
        for _ in range(35):
            *_, info = env.step(FULL_THROTTLE)
        heading = info["theta"]
        for _ in range(12):
            *_, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert not info["collision"]
        return float(info["theta"] - heading)
    finally:
        env.close()


def test_the_simulator_runs_with_the_requested_friction(room_map: Path) -> None:
    simulator = F1TenthSimulator(room_map, seed=0, friction=0.3)
    try:
        assert simulator.friction == 0.3
    finally:
        simulator.close()


def test_lower_friction_makes_the_car_understeer(room_map: Path) -> None:
    grippy = _heading_change_in_a_full_left_turn(room_map, friction=None)
    slippery = _heading_change_in_a_full_left_turn(room_map, friction=0.3)
    assert slippery < grippy - 0.1  # measured: about 1.29 rad against 1.53 rad


def test_set_friction_changes_the_friction_of_a_running_simulator(room_map: Path) -> None:
    simulator = F1TenthSimulator(room_map, seed=0)
    try:
        simulator.set_friction(0.6)
        assert simulator.friction == 0.6
        with pytest.raises(ValueError, match="friction"):
            simulator.set_friction(0.0)
        assert simulator.friction == 0.6
    finally:
        simulator.close()


def test_friction_drawn_on_reset_drives_like_a_simulator_built_with_it(room_map: Path) -> None:
    built_with = _heading_change_in_a_full_left_turn(room_map, friction=0.3)
    drawn = _heading_change_in_a_full_left_turn(
        room_map, friction=None, config=EnvConfig(friction_range=(0.3, 0.3))
    )
    assert drawn == built_with
