"""PPO training envs on several tracks at once: one F1TENTH track per SubprocVecEnv worker.

F1TENTH Gym keeps its LiDAR simulator, and with it the map, in class-level state
(RaceCar.scan_simulator), so a second track built in the same process silently replaces the
first one's map in every scan. Each track therefore runs in its own worker process.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lapzero.sim.env import EnvConfig
from lapzero.sim.f1tenth import make_env
from lapzero.sim.reward import RacingReward, RewardConfig
from lapzero.sim.simulator import Pose
from lapzero.sim.track import load_centreline, start_pose_from_centreline

logger = logging.getLogger(__name__)

CENTRELINE_FILE = "centreline.csv"


def track_files(tracks_dir: Path, track_id: str) -> tuple[Path, Path]:
    """Return the map yaml and centreline CSV of a track stored as <tracks_dir>/<track_id>/.

    The layout is <track_id>/<track_id>.yaml (with the image it names) and
    <track_id>/centreline.csv, as in data/tracks/spielberg/.

    Args:
        tracks_dir: Directory holding one subdirectory per track, e.g. data/tracks.
        track_id: Name of the track's subdirectory, e.g. "spielberg".

    Returns:
        (map_yaml, centreline_csv).

    Raises:
        FileNotFoundError: Naming the first of the two files that does not exist.
    """
    directory = tracks_dir / track_id
    map_yaml, centreline_csv = directory / f"{track_id}.yaml", directory / CENTRELINE_FILE
    for path in (map_yaml, centreline_csv):
        if not path.is_file():
            raise FileNotFoundError(f"track {track_id!r}: {path} does not exist")
    return map_yaml, centreline_csv


@dataclass(frozen=True, slots=True)
class TrackEnvFactory:
    """Builds the monitored PPO training env for one track; pickled into a SubprocVecEnv worker.

    The env is Monitor(RacingReward(LapZeroEnv)): SB3 adds the Monitor itself only around a
    single env, and RaceMetricsCallback and rollout/ep_rew_mean read the episode stats it adds.

    Attributes:
        track_id: Names the track in logs and per-track metrics.
        map_yaml: ROS map yaml of the track.
        centreline_csv: Centreline CSV in driving order (f1tenth_racetracks format).
        seed: Simulator seed (LiDAR noise).
        env_config: Episode, action-mapping and friction randomization settings.
        reward_config: Racing reward weights.
        start_pose: Fixed start pose, or None for the centreline's first point facing the second.
    """

    track_id: str
    map_yaml: Path
    centreline_csv: Path
    seed: int
    env_config: EnvConfig
    reward_config: RewardConfig
    start_pose: Pose | None = None

    def __call__(self) -> Monitor[NDArray[np.float32], NDArray[np.float32]]:
        """Load the centreline, start the simulator and wrap the env for training.

        Returns:
            The ready-to-reset training env.
        """
        centreline = load_centreline(self.centreline_csv)
        start = self.start_pose or start_pose_from_centreline(centreline)
        logger.info(
            "building training env for %s from %s, start %s", self.track_id, self.map_yaml, start
        )
        sim_env = make_env(
            self.map_yaml, start, seed=self.seed, config=self.env_config, centreline=centreline
        )
        return Monitor(RacingReward(sim_env, self.reward_config))


def make_track_vec_env(factories: Sequence[TrackEnvFactory]) -> SubprocVecEnv:
    """Start one worker process per factory; env i of the result drives factories[i]'s track.

    Args:
        factories: One per env, normally one per track.

    Returns:
        The VecEnv. Close it to stop the workers.

    Raises:
        ValueError: If `factories` is empty.
    """
    if not factories:
        raise ValueError("make_track_vec_env needs at least one track")
    logger.info(
        "starting %d env workers: %s", len(factories), ", ".join(f.track_id for f in factories)
    )
    return SubprocVecEnv(list(factories))
