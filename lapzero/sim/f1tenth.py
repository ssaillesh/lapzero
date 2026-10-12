"""F1TENTH Gym v1 (Gymnasium branch) backend: the only module that imports f1tenth_gym.

F1TenthSimulator implements the Simulator protocol and is driven only by LapZeroEnv, so raw
simulator commands always pass through the env's action mapping. f1tenth_gym ships no type
information, so its objects are typed as Any at this boundary.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lapzero.contract import (
    LIDAR_FOV,
    LIDAR_MAX,
    RAW_BEAMS,
    SIM_DT,
    SIM_V_MAX,
    SIM_V_MIN,
    STEER_MAX,
    ContractViolationError,
)
from lapzero.sim.env import EnvConfig, LapZeroEnv, RewardFn, zero_reward
from lapzero.sim.simulator import Pose, SimState

logger = logging.getLogger(__name__)

_EGO_ID = "agent_0"
_F1TENTH_FEATURES = [
    "scan",
    "pose_x",
    "pose_y",
    "pose_theta",
    "linear_vel_x",
    "linear_vel_y",
    "ang_vel_z",
    "collision",
    "lap_count",
]


def load_track(map_yaml: Path) -> Any:
    """Load a ROS map (yaml plus image) as an f1tenth_gym Track, bypassing its map downloader.

    F1TENTH Gym builds a reset sampler from the track's reference line even when explicit start
    poses are given, so a short placeholder line is attached; LapZero never uses it.

    Args:
        map_yaml: Map yaml with image, resolution, origin, negate and thresholds. The image path
            is relative to the yaml.

    Returns:
        An f1tenth_gym Track (untyped third-party object).

    Raises:
        FileNotFoundError: If the yaml or its image does not exist.
    """
    from f1tenth_gym.envs.track import Raceline, Track  # heavy (numba): import only when used

    spec = Track.load_spec(track=map_yaml.stem, filespec=str(map_yaml))
    image_path = map_yaml.parent / spec.image
    with Image.open(image_path) as image:
        # Same conversion as Track.from_track_name: flip to world-up rows, binarize at 128.
        pixels = np.array(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), dtype=np.float32)
    occupancy = np.where(pixels > 128, 255.0, 0.0).astype(np.float32)
    placeholder = Raceline(xs=np.array([0.0, 1.0, 2.0]), ys=np.zeros(3), velxs=np.ones(3))
    return Track(
        spec=spec,
        occupancy_map=occupancy,
        filepath=str(image_path.with_suffix("")),
        ext=image_path.suffix,
        centerline=placeholder,
        raceline=placeholder,
    )


def verify_baseline(sim_env: Any) -> None:
    """Check an F110Env's built-in values against the contract's simulator baseline table.

    Args:
        sim_env: An f1tenth_gym F110Env (untyped third-party object).

    Raises:
        ContractViolationError: Naming the first value that differs from the contract.
    """
    scanner = sim_env.sim.agents[0].scan_simulator
    pairs = {
        "SIM_DT": (sim_env.timestep, SIM_DT),
        "RAW_BEAMS": (scanner.num_beams, RAW_BEAMS),
        "LIDAR_FOV": (scanner.fov, LIDAR_FOV),
        "LIDAR_MAX": (scanner.max_range, LIDAR_MAX),
        "STEER_MAX (s_max)": (sim_env.params["s_max"], STEER_MAX),
        "STEER_MAX (s_min)": (-sim_env.params["s_min"], STEER_MAX),
        "SIM_V_MIN": (sim_env.params["v_min"], SIM_V_MIN),
        "SIM_V_MAX": (sim_env.params["v_max"], SIM_V_MAX),
    }
    for name, (actual, expected) in pairs.items():
        if not math.isclose(float(actual), expected, rel_tol=1e-9):
            raise ContractViolationError(f"simulator {name} is {actual}, contract says {expected}")


def state_from_f1tenth_obs(obs: Mapping[str, Any]) -> SimState:
    """Convert an F1TENTH Gym v1 "features" observation of the ego car into a SimState.

    The "features" type is used because the "original" type hard-codes linear_vels_y to 0.

    Args:
        obs: Observation dict keyed by agent id, with the _F1TENTH_FEATURES keys.

    Returns:
        The ego car's state.
    """
    ego = obs[_EGO_ID]
    return SimState(
        scan=np.asarray(ego["scan"], dtype=np.float64),
        pose=Pose(float(ego["pose_x"]), float(ego["pose_y"]), float(ego["pose_theta"])),
        v_long=float(ego["linear_vel_x"]),
        v_lat=float(ego["linear_vel_y"]),
        yaw_rate=float(ego["ang_vel_z"]),
        collision=bool(ego["collision"]),
        lap_count=int(ego["lap_count"]),
    )


class F1TenthSimulator:
    """F1TENTH Gym v1 running one car, behind the Simulator protocol.

    F1TENTH Gym shares a single LiDAR simulator, and so a single map, across every car in a
    process. Do not run envs on different maps in the same process.
    """

    def __init__(self, map_yaml: Path, *, seed: int, render: bool = False) -> None:
        """Load the map and start the simulator.

        Args:
            map_yaml: ROS map yaml; see load_track.
            seed: Seed for the simulator's LiDAR noise.
            render: Open F1TENTH's pygame window, drawn by render().

        Raises:
            FileNotFoundError: If the map files are missing.
            ContractViolationError: If the simulator baseline differs from the contract.
        """
        from f1tenth_gym.envs import F110Env  # heavy (numba, pygame): import only when used

        logger.info("starting F1TENTH Gym on %s with seed=%d", map_yaml, seed)
        self._env: Any = F110Env(
            config={
                "map": load_track(map_yaml),
                "num_agents": 1,
                "seed": seed,
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "features", "features": _F1TENTH_FEATURES},
            },
            render_mode="human" if render else None,
        )
        verify_baseline(self._env)

    def reset(self, pose: Pose) -> SimState:
        """Place the car at rest at `pose` and return the first reading."""
        obs, _ = self._env.reset(options={"poses": np.array([[pose.x, pose.y, pose.theta]])})
        return state_from_f1tenth_obs(obs)

    def step(self, steer_rad: float, v_target: float) -> SimState:
        """Advance one physics step; F1TENTH's own done flag is ignored (LapZeroEnv decides)."""
        obs, *_ = self._env.step(np.array([[steer_rad, v_target]]))
        return state_from_f1tenth_obs(obs)

    def render(self) -> None:
        """Draw the latest state in the F1TENTH window; does nothing unless opened with render."""
        self._env.render()

    def close(self) -> None:
        """Close the F1TENTH env."""
        self._env.close()


def make_env(
    map_yaml: Path,
    start_pose: Pose,
    *,
    seed: int,
    reward_fn: RewardFn = zero_reward,
    config: EnvConfig | None = None,
) -> LapZeroEnv:
    """Build a LapZeroEnv on F1TENTH Gym for one track.

    Args:
        map_yaml: ROS map yaml for the track.
        start_pose: Fixed start pose for the track.
        seed: Simulator seed (LiDAR noise). Seed the episode itself with env.reset(seed=...).
        reward_fn: Per-physics-step reward; defaults to the 0.0 placeholder.
        config: Episode and action-mapping settings; defaults to the contract values.

    Returns:
        The ready-to-reset env.
    """
    simulator = F1TenthSimulator(map_yaml, seed=seed)
    return LapZeroEnv(simulator, start_pose, reward_fn=reward_fn, config=config)
