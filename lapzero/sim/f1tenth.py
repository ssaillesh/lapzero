"""F1TENTH Gym v1 (Gymnasium branch) backend: the only module that imports f1tenth_gym.

F1TenthSimulator implements the Simulator protocol and is driven only by LapZeroEnv, so raw
simulator commands always pass through the env's action mapping. f1tenth_gym ships no type
information, so its objects are typed as Any at this boundary.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from lapzero.contract import (
    BEAM_BIN,
    LIDAR_FOV,
    LIDAR_MAX,
    N_BEAMS,
    RAW_BEAMS,
    SIM_DT,
    SIM_V_MAX,
    SIM_V_MIN,
    STEER_MAX,
    ContractViolationError,
)
from lapzero.sim.env import EnvConfig, LapZeroEnv, RewardFn, zero_reward
from lapzero.sim.simulator import Pose, SimState
from lapzero.sim.track import Centreline

logger = logging.getLogger(__name__)

RENDER_MIRRORED: Final = True
"""F1TENTH's window draws world +y downward, so it shows the world mirrored: a left (CCW) turn
looks like a right turn on screen. Human input steers by the screen to compensate."""

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


@dataclasses.dataclass(frozen=True, slots=True)
class BaselineValue:
    """One simulator value the contract fixes: what the simulator reports and what the contract
    says, in the contract's units (s, rad, m, m/s, or a beam count)."""

    name: str
    installed: float
    contract: float

    @property
    def matches(self) -> bool:
        """Whether the simulator value equals the contract value (relative tolerance 1e-9)."""
        return math.isclose(self.installed, self.contract, rel_tol=1e-9)


def baseline_values(sim_env: Any) -> list[BaselineValue]:
    """Read every row of the contract's simulator baseline table from an F110Env.

    Also reads the simulator's beams per downsampled beam, which must equal N_BEAMS for the
    LiDAR downsampling rule (contract Section 1.3) to apply.

    Args:
        sim_env: An f1tenth_gym F110Env (untyped third-party object).

    Returns:
        The simulator and contract value of each row, in baseline table order.
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
        "N_BEAMS (RAW_BEAMS / BEAM_BIN)": (scanner.num_beams / BEAM_BIN, N_BEAMS),
    }
    return [
        BaselineValue(name, float(installed), float(contract))
        for name, (installed, contract) in pairs.items()
    ]


def verify_baseline(sim_env: Any) -> None:
    """Check an F110Env's built-in values against the contract's simulator baseline table.

    Args:
        sim_env: An f1tenth_gym F110Env (untyped third-party object).

    Raises:
        ContractViolationError: Naming the first value that differs from the contract.
    """
    for value in baseline_values(sim_env):
        if not value.matches:
            raise ContractViolationError(
                f"simulator {value.name} is {value.installed}, contract says {value.contract}"
            )


def installed_baseline() -> list[BaselineValue]:
    """Start the installed F1TENTH Gym on a blank map and read its simulator baseline values.

    Tracks are not in git, so the simulator runs on an empty walled room written to a temporary
    directory. Nothing is checked here; compare with BaselineValue.matches.

    Returns:
        The simulator and contract value of each baseline row (see baseline_values).
    """
    from f1tenth_gym.envs import F110Env  # heavy (numba, pygame): import only when used

    with tempfile.TemporaryDirectory() as directory:
        config = f110_config(load_track(_write_empty_room(Path(directory))), seed=0, friction=None)
        sim_env = F110Env(config=config)
        try:
            return baseline_values(sim_env)
        finally:
            sim_env.close()


def _write_empty_room(directory: Path) -> Path:
    """Write a ROS map of an empty 5 m x 5 m room with walls in `directory`; return its yaml."""
    size_px, wall_px = 100, 4  # 0.05 m per pixel
    pixels = np.full((size_px, size_px), 255, dtype=np.uint8)
    pixels[:wall_px, :] = pixels[-wall_px:, :] = 0
    pixels[:, :wall_px] = pixels[:, -wall_px:] = 0
    (directory / "room.pgm").write_bytes(
        f"P5\n{size_px} {size_px}\n255\n".encode() + pixels.tobytes()
    )
    (directory / "room.yaml").write_text(
        "image: room.pgm\nresolution: 0.05\norigin: [-2.5, -2.5, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    return directory / "room.yaml"


def f110_config(track: Any, *, seed: int, friction: float | None) -> dict[str, Any]:
    """Build the F110Env config for one ego car on `track`.

    F110Env deep-merges "params" into its default vehicle, so only the friction is overridden.
    Friction is not a contract value; varying it is a robustness diagnostic.

    Args:
        track: f1tenth_gym Track from load_track (untyped third-party object).
        seed: Seed for the simulator's LiDAR noise.
        friction: Tyre-road friction coefficient mu (dimensionless), or None for F1TENTH's
            default vehicle.

    Returns:
        The config dict for F110Env.

    Raises:
        ValueError: If `friction` is not finite and > 0.
    """
    config: dict[str, Any] = {
        "map": track,
        "num_agents": 1,
        "seed": seed,
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "features", "features": _F1TENTH_FEATURES},
    }
    if friction is not None:
        _check_friction(friction)
        config["params"] = {"mu": friction}
    return config


def _check_friction(mu: float) -> None:
    """Raise ValueError unless the friction coefficient `mu` is finite and > 0."""
    if not (math.isfinite(mu) and mu > 0.0):
        raise ValueError(f"friction must be finite and > 0, got {mu}")


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

    def __init__(
        self, map_yaml: Path, *, seed: int, render: bool = False, friction: float | None = None
    ) -> None:
        """Load the map and start the simulator.

        Args:
            map_yaml: ROS map yaml; see load_track.
            seed: Seed for the simulator's LiDAR noise.
            render: Open F1TENTH's pygame window, drawn by render().
            friction: Tyre-road friction coefficient mu (dimensionless); None keeps F1TENTH's
                default. A robustness diagnostic, not a contract value.

        Raises:
            FileNotFoundError: If the map files are missing.
            ValueError: If `friction` is not finite and > 0.
            ContractViolationError: If the simulator baseline differs from the contract.
        """
        from f1tenth_gym.envs import F110Env  # heavy (numba, pygame): import only when used

        config = f110_config(load_track(map_yaml), seed=seed, friction=friction)
        self._env: Any = F110Env(config=config, render_mode="human" if render else None)
        verify_baseline(self._env)
        logger.info(
            "started F1TENTH Gym on %s with seed=%d, friction mu=%s", map_yaml, seed, self.friction
        )

    @property
    def friction(self) -> float:
        """Tyre-road friction coefficient mu the simulated car runs with (dimensionless)."""
        return float(self._env.params["mu"])

    def set_friction(self, mu: float) -> None:
        """Set the tyre-road friction coefficient mu for the following physics steps.

        F110Env.configure deep-merges the value into the vehicle params and hands them to the
        car, which reads them on every step, so the change needs no restart.

        Args:
            mu: Friction coefficient (dimensionless).

        Raises:
            ValueError: If `mu` is not finite and > 0.
        """
        _check_friction(mu)
        self._env.configure({"params": {"mu": mu}})

    def reset(self, pose: Pose) -> SimState:
        """Place the car at rest at `pose` and return the first reading.

        F1TENTH builds its reset observation before recounting laps, so that observation still
        carries the previous episode's lap count. A reset car has completed no laps, so the
        reading reports 0; F1TENTH's own count restarts from 0 on the next step.
        """
        obs, _ = self._env.reset(options={"poses": np.array([[pose.x, pose.y, pose.theta]])})
        return dataclasses.replace(state_from_f1tenth_obs(obs), lap_count=0)

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
    centreline: Centreline | None = None,
) -> LapZeroEnv:
    """Build a LapZeroEnv on F1TENTH Gym for one track.

    Args:
        map_yaml: ROS map yaml for the track.
        start_pose: Fixed start pose for the track.
        seed: Simulator seed (LiDAR noise). Seed the episode itself with env.reset(seed=...).
        reward_fn: Per-physics-step reward; defaults to the 0.0 placeholder.
        config: Episode and action-mapping settings; defaults to the contract values.
        centreline: Track centreline in driving order, for info["progress"]; optional.

    Returns:
        The ready-to-reset env.
    """
    simulator = F1TenthSimulator(map_yaml, seed=seed)
    return LapZeroEnv(
        simulator, start_pose, reward_fn=reward_fn, config=config, centreline=centreline
    )
