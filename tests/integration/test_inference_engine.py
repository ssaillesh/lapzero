"""Integration tests: InferenceEngine driving a policy checkpoint on the real F1TENTH Gym v1.

Uses a synthetic 20 m x 20 m walled room centred on the origin (inner wall faces 9.7 m from it),
so no map download is needed, with a 6 m square centreline starting at (-3, -3) facing +x.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from lapzero.inference.engine import InferenceEngine
from lapzero.io.telemetry import LapStatus
from lapzero.training.policy import BCPolicy, save_policy

pytestmark = pytest.mark.integration

SQUARE = [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]]
EAST_WALL_X = 9.7


@pytest.fixture(scope="module")
def room(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Write a ROS map (PGM + yaml) of an empty room with 0.3 m walls, and its centreline CSV."""
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
    np.savetxt(directory / "centreline.csv", SQUARE, delimiter=",", header="x_m, y_m")
    return directory / "room.yaml", directory / "centreline.csv"


@pytest.fixture
def engine(room: tuple[Path, Path], tmp_path: Path) -> Iterator[InferenceEngine]:
    """An engine at friction 0.8 whose policy drives straight at 0.9 throttle."""
    policy = BCPolicy(8)
    with torch.no_grad():
        policy.net[-2].weight.zero_()
        policy.net[-2].bias.copy_(torch.atanh(torch.tensor([0.0, 0.9])))
    save_policy(policy, tmp_path / "straight.pt")
    map_yaml, centreline = room
    with InferenceEngine(str(tmp_path / "straight.pt"), map_yaml, centreline, 0.8) as engine:
        yield engine


def test_driving_straight_into_the_wall_is_a_crashed_episode_with_valid_telemetry(
    engine: InferenceEngine,
) -> None:
    result = engine.run_episode()
    assert result.status is LapStatus.CRASHED
    assert result.frames[0].x == -3.0
    assert result.frames[-1].x > EAST_WALL_X - 1.0
    assert max(frame.speed for frame in result.frames) > 5.0
    assert {frame.friction_mu for frame in result.frames} == {0.8}
    assert all(frame.lap_count == 0 for frame in result.frames)


def test_step_by_step_driving_follows_the_given_actions(engine: InferenceEngine) -> None:
    start = engine.reset()
    for _ in range(25):
        frame, _, terminated, truncated, _ = engine.step(np.array([1.0, 0.5]))
        assert not (terminated or truncated)
    assert frame.steer == 1.0
    assert frame.theta > start.theta + 0.5  # full left steer turns counter-clockwise
    assert frame.v_lat != 0.0  # the dynamic model slips sideways in a turn
