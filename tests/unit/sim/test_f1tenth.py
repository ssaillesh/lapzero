"""Unit tests for lapzero/sim/f1tenth.py: adapter logic and the f1tenth_gym import boundary.

These do not start the simulator; F1TenthSimulator itself runs in tests/integration.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import lapzero
from lapzero.contract import RAW_BEAMS, ContractViolationError
from lapzero.sim.f1tenth import load_track, state_from_f1tenth_obs, verify_baseline
from lapzero.sim.simulator import Pose


def test_only_the_f1tenth_module_imports_f1tenth_gym() -> None:
    package_root = Path(lapzero.__file__).parent
    importers = set()
    for path in package_root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "f1tenth_gym" for name in modules):
                importers.add(path.relative_to(package_root).as_posix())
    assert importers == {"sim/f1tenth.py"}


def _f1tenth_obs(**overrides: Any) -> dict[str, dict[str, Any]]:
    ego: dict[str, Any] = {
        "scan": np.full(RAW_BEAMS, 7.0, dtype=np.float32),
        "pose_x": np.float32(1.5),
        "pose_y": np.float32(-2.0),
        "pose_theta": np.float32(0.25),
        "linear_vel_x": np.float32(3.0),
        "linear_vel_y": np.float32(-0.5),
        "ang_vel_z": np.float32(1.25),
        "collision": 1,
        "lap_count": np.float32(2.0),
    }
    return {"agent_0": ego | overrides}


def test_f1tenth_features_observation_converts_to_sim_state() -> None:
    state = state_from_f1tenth_obs(_f1tenth_obs())
    assert state.scan.shape == (RAW_BEAMS,)
    assert state.scan.dtype == np.float64
    assert state.pose == Pose(1.5, -2.0, 0.25)
    assert (state.v_long, state.v_lat, state.yaw_rate) == (3.0, -0.5, 1.25)
    assert state.collision is True
    assert state.lap_count == 2


def _fake_f110_env(**params: float) -> SimpleNamespace:
    scanner = SimpleNamespace(num_beams=1080, fov=4.7, max_range=30.0)
    defaults = {"s_min": -0.4189, "s_max": 0.4189, "v_min": -5.0, "v_max": 20.0}
    return SimpleNamespace(
        timestep=0.01,
        params=defaults | params,
        sim=SimpleNamespace(agents=[SimpleNamespace(scan_simulator=scanner)]),
    )


def test_verify_baseline_accepts_contract_values() -> None:
    verify_baseline(_fake_f110_env())


@pytest.mark.parametrize(
    ("params", "name"),
    [({"s_max": 0.5}, "s_max"), ({"s_min": -0.3}, "s_min"), ({"v_max": 25.0}, "SIM_V_MAX")],
)
def test_verify_baseline_names_the_value_that_differs(params: dict[str, float], name: str) -> None:
    with pytest.raises(ContractViolationError, match=name):
        verify_baseline(_fake_f110_env(**params))


def test_load_track_reads_a_local_ros_map(tmp_path: Path) -> None:
    pixels = np.full((40, 60), 255, dtype=np.uint8)
    pixels[0, :] = 0  # top image row is an occupied wall
    (tmp_path / "box.pgm").write_bytes(b"P5\n60 40\n255\n" + pixels.tobytes())
    (tmp_path / "box.yaml").write_text(
        "image: box.pgm\nresolution: 0.05\norigin: [-1.5, -1.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    track = load_track(tmp_path / "box.yaml")
    assert track.spec.resolution == 0.05
    assert track.occupancy_map.shape == (40, 60)
    assert (track.occupancy_map[-1] == 0.0).all()  # flipped: world-up rows, wall at max y
    assert (track.occupancy_map[:-1] == 255.0).all()


def test_load_track_raises_for_a_missing_map(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_track(tmp_path / "missing.yaml")
