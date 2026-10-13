"""Inference engine: drive a trained policy on one track, one control step at a time.

The engine loads a policy (local checkpoint, registry alias or MLflow run), starts one F1TENTH
simulator for its lifetime and reports every control step as StepTelemetry, either step by step
(reset, step), as a generator (stream_episode) or for a whole episode (run_episode).

Every episode runs on a LapZeroEnv with one lap per episode and the friction fixed at the
engine's mu (friction_range (mu, mu), as in the friction sweep). Lap counts are the env's
(contract Section 3); rewards are the training reward (lapzero.sim.reward.RacingReward).

This module imports lapzero.sim and lapzero.training, so the backend must not import it: the API
reaches it through a separate worker process and only shares lapzero.inference.schemas.
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

from lapzero.contract import PREV_STEER, PREV_THROTTLE
from lapzero.eval.gate import NOMINAL_MU
from lapzero.inference.schemas import EpisodeResult, StepTelemetry
from lapzero.io.telemetry import LapStatus
from lapzero.registry import get_tracking_uri
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.f1tenth import F1TenthSimulator
from lapzero.sim.reward import RacingReward
from lapzero.sim.simulator import Pose
from lapzero.sim.track import load_centreline, start_pose_from_centreline
from lapzero.training.policy import BCPolicy, load_policy
from lapzero.training.tracking import (
    POLICY_FLAVOR,
    REGISTERED_MODEL,
    ModelAlias,
    load_registered_policy,
    mlflow_client,
    read_policy_flavor,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS: Final = 5000
"""Control steps run_episode drives at most (200 s at CTRL_HZ, beyond the env's time limit)."""
_TIME_TOLERANCE_S: Final = 1e-9
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_RUNS_URI = re.compile(r"runs:/[0-9a-f]{32}/.+")
_MODELS_URI = re.compile(r"models:/(?P<name>[^/@]+)@(?P<alias>[^/@]+)")


def load_policy_source(
    source: str, *, tracking_uri: str | None = None, model_name: str = REGISTERED_MODEL
) -> BCPolicy:
    """Load a policy from a local checkpoint, the model registry or an MLflow run.

    Args:
        source: One of
            - "runs:/<run_id>/<path>": a lapzero_policy MLflow Model logged by a run;
            - "models:/<name>@<alias>": the version an alias of a registered model points at;
            - a path ending in .pt: a save_policy checkpoint on disk;
            - "champion" or "challenger": that alias of `model_name`;
            - a 32-character MLflow run ID: the version of `model_name` that run registered.
        tracking_uri: MLflow tracking URI; defaults to the repo-root store (get_tracking_uri).
        model_name: Registered model for bare aliases and run IDs.

    Returns:
        The policy, in eval mode.

    Raises:
        ValueError: If `source` matches none of the forms, the run registered no version of
            `model_name`, or the artifact is not a lapzero_policy model.
        FileNotFoundError: If a .pt checkpoint does not exist.
        ContractViolationError: If the policy was saved under another contract major version.
        mlflow.exceptions.MlflowException: If the model, alias or run does not exist.
    """
    uri = tracking_uri or get_tracking_uri()
    if _RUNS_URI.fullmatch(source):
        return _load_logged_policy(source, uri)
    if match := _MODELS_URI.fullmatch(source):
        return load_registered_policy(
            tracking_uri=uri, model_name=match["name"], alias=match["alias"]
        )
    if Path(source).suffix == ".pt":
        if not Path(source).is_file():
            raise FileNotFoundError(f"policy checkpoint {source} does not exist")
        return load_policy(Path(source))
    if source in tuple(ModelAlias):
        return load_registered_policy(tracking_uri=uri, model_name=model_name, alias=source)
    if _RUN_ID.fullmatch(source):
        return _load_logged_policy(_registered_source(source, uri, model_name), uri)
    raise ValueError(
        f"model source {source!r} is not a .pt path, champion/challenger, a run ID, "
        "runs:/<run_id>/<path> or models:/<name>@<alias>"
    )


def _registered_source(run_id: str, tracking_uri: str, model_name: str) -> str:
    """Return the artifact URI of the latest version of `model_name` registered by `run_id`."""
    client = mlflow_client(tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}' AND run_id='{run_id}'")
    if not versions:
        raise ValueError(f"run {run_id} registered no version of {model_name}")
    return str(max(versions, key=lambda version: int(version.version)).source)


def _load_logged_policy(artifact_uri: str, tracking_uri: str) -> BCPolicy:
    """Download a lapzero_policy MLflow Model and load its checkpoint."""
    import mlflow.artifacts  # heavy import, only when MLflow is used

    mlflow_client(tracking_uri)  # allows a local file: store, as for every other MLflow call
    with tempfile.TemporaryDirectory() as download:
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=download, tracking_uri=tracking_uri
            )
        )
        flavor = read_policy_flavor(local)
        if flavor is None:
            raise ValueError(f"{artifact_uri} is not a {POLICY_FLAVOR} model")
        return load_policy(local / flavor["checkpoint"])


class InferenceEngine:
    """A policy driving one track on F1TENTH Gym, reporting each control step as telemetry.

    Episode i is reset with seed + i unless reset is given a seed. The policy always drives its
    deterministic action. Close the engine (or use it as a context manager) to stop the
    simulator.
    """

    def __init__(
        self,
        model_source: str,
        track_yaml: str | Path,
        centreline_csv: str | Path,
        friction: float = NOMINAL_MU,
        *,
        seed: int = 0,
        start_pose: Pose | None = None,
        tracking_uri: str | None = None,
        model_name: str = REGISTERED_MODEL,
    ) -> None:
        """Load the policy and the track, and start the simulator.

        Args:
            model_source: Policy to drive; see load_policy_source for the accepted forms.
            track_yaml: ROS map yaml of the track.
            centreline_csv: Track centreline CSV in driving order, for progress and laps.
            friction: Tyre-road friction coefficient mu (dimensionless) of every episode.
            seed: Simulator seed (LiDAR noise) and seed of the first episode; logged.
            start_pose: Fixed start pose; defaults to the first centreline point facing the
                second (the f1tenth_racetracks start line).
            tracking_uri: MLflow tracking URI for registry and run sources.
            model_name: Registered model for bare aliases and run IDs.

        Raises:
            ValueError: If `friction` is not finite and > 0, or `model_source` is not valid.
            FileNotFoundError: If the checkpoint, map or centreline file is missing.
        """
        self._policy: nn.Module = load_policy_source(
            model_source, tracking_uri=tracking_uri, model_name=model_name
        )
        self._centreline = load_centreline(Path(centreline_csv))
        self._start_pose = start_pose or start_pose_from_centreline(self._centreline)
        self._friction = friction
        config = self._config(randomize_start=False)  # rejects a bad friction before the sim
        self._simulator = F1TenthSimulator(Path(track_yaml), seed=seed)
        self._env, self._rewarded = self._wrap(config)
        self._seed = seed
        self._episodes = 0
        self._obs: NDArray[np.float32] | None = None
        self._frame: StepTelemetry | None = None
        self._done = True
        logger.info(
            "inference engine: policy %s on %s, friction mu=%.4f, seed=%d",
            model_source,
            track_yaml,
            friction,
            seed,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the simulator."""
        self._env.close()

    def predict(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the policy's deterministic action for an observation.

        Args:
            obs: Contract observation, shape (OBS_DIM,).

        Returns:
            Normalized [steer, throttle], shape (ACT_DIM,), float32.
        """
        with torch.no_grad():
            action = self._policy(torch.from_numpy(obs))
        return np.asarray(action.numpy(), dtype=np.float32)

    def reset(self, randomize_start: bool = False, *, seed: int | None = None) -> StepTelemetry:
        """Start a new episode and return its first frame (step 0).

        Args:
            randomize_start: Jitter the start pose (a robustness diagnostic; the contract
                reserves it for training).
            seed: Episode seed; defaults to the engine seed plus the number of earlier episodes.

        Returns:
            Telemetry of the car at rest on the start pose.
        """
        if randomize_start != self._env.config.randomize_start:
            self._env, self._rewarded = self._wrap(self._config(randomize_start))
        episode_seed = self._seed + self._episodes if seed is None else seed
        self._episodes += 1
        obs, info = self._rewarded.reset(seed=episode_seed)
        self._obs, self._done = obs, False
        self._frame = self._telemetry(0, obs, info, reward=0.0, lap_fraction=0.0)
        return self._frame

    def step(
        self, action: ArrayLike | None = None
    ) -> tuple[StepTelemetry, float, bool, bool, dict[str, Any]]:
        """Drive one control step with `action`, or the policy's action if it is None.

        Args:
            action: Normalized [steer, throttle]; out-of-range values are clipped by the env.

        Returns:
            The step's telemetry, reward, terminated, truncated and the env's info dict
            (contract Section 4).

        Raises:
            RuntimeError: If no episode is running (before reset() or after it ended).
            ValueError: If `action` does not have shape (ACT_DIM,).
        """
        if self._obs is None or self._frame is None or self._done:
            raise RuntimeError("call reset() before step() and after an episode ends")
        act = self.predict(self._obs) if action is None else np.asarray(action, np.float32)
        obs, reward, terminated, truncated, info = self._rewarded.step(act)
        prev = self._frame
        step_fraction = (float(info["progress"]) - prev.progress + 0.5) % 1.0 - 0.5  # short way
        self._obs, self._done = obs, terminated or truncated
        self._frame = self._telemetry(
            prev.step + 1, obs, info, reward=reward, lap_fraction=prev.lap_fraction + step_fraction
        )
        return self._frame, reward, terminated, truncated, info

    def stream_episode(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        randomize_start: bool = False,
        *,
        seed: int | None = None,
    ) -> Generator[StepTelemetry, None, EpisodeResult]:
        """Run one episode with the policy, yielding each frame as it is driven.

        Args:
            max_steps: Control steps after which the episode is cut short as a timeout.
            randomize_start: Jitter the start pose; see reset.
            seed: Episode seed; see reset.

        Returns:
            A generator of the frames from step 0, whose return value is the EpisodeResult.

        Raises:
            ValueError: If max_steps < 1 (on the call, before the episode starts).
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        return self._episode(max_steps, randomize_start, seed)

    def run_episode(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        randomize_start: bool = False,
        *,
        seed: int | None = None,
    ) -> EpisodeResult:
        """Run one episode with the policy and return its result; arguments as stream_episode."""
        episode = self.stream_episode(max_steps, randomize_start, seed=seed)
        while True:
            try:
                next(episode)
            except StopIteration as end:
                result: EpisodeResult = end.value
                return result

    def _episode(
        self, max_steps: int, randomize_start: bool, seed: int | None
    ) -> Generator[StepTelemetry, None, EpisodeResult]:
        frames = [self.reset(randomize_start, seed=seed)]
        yield frames[0]
        terminated = truncated = False
        while not (terminated or truncated) and len(frames) <= max_steps:
            frame, _, terminated, truncated, _ = self.step()
            frames.append(frame)
            yield frame
        self._done = True
        last = frames[-1]
        if last.collision:
            status = LapStatus.CRASHED
        elif terminated:
            status = LapStatus.COMPLETED
        elif truncated and last.t < self._env.config.max_episode_s - _TIME_TOLERANCE_S:
            status = LapStatus.STUCK
        else:  # the env's time limit or max_steps
            status = LapStatus.TIMEOUT
        logger.info("episode ended %s after %d steps, %.2f s", status, last.step, last.t)
        return EpisodeResult(
            status=status,
            collision=last.collision,
            steps=last.step,
            lap_count=last.lap_count,
            episode_time_s=last.t,
            lap_time_s=last.t if status is LapStatus.COMPLETED else None,
            frames=tuple(frames),
        )

    def _config(self, randomize_start: bool) -> EnvConfig:
        return EnvConfig(
            randomize_start=randomize_start, friction_range=(self._friction, self._friction)
        )

    def _wrap(self, config: EnvConfig) -> tuple[LapZeroEnv, RacingReward]:
        """Build an env with `config` around the engine's simulator, scored by RacingReward."""
        env = LapZeroEnv(
            self._simulator, self._start_pose, config=config, centreline=self._centreline
        )
        return env, RacingReward(env)

    def _telemetry(
        self,
        step: int,
        obs: NDArray[np.float32],
        info: dict[str, Any],
        *,
        reward: float,
        lap_fraction: float,
    ) -> StepTelemetry:
        """Build a frame from the env's info dict, its observation and its latest reading."""
        v_long, v_lat = float(info["v_long"]), self._env.v_lat
        return StepTelemetry(
            step=step,
            t=float(info["t"]),
            x=float(info["x"]),
            y=float(info["y"]),
            theta=float(info["theta"]),
            v_long=v_long,
            v_lat=v_lat,
            speed=math.hypot(v_long, v_lat),
            # The env's clipped action, echoed in the observation (contract Section 1.4).
            steer=float(obs[PREV_STEER]),
            throttle=float(obs[PREV_THROTTLE]),
            steer_rad=float(info["steer_rad"]),
            v_target=float(info["v_target"]),
            progress=float(info["progress"]),
            lap_fraction=lap_fraction,
            lap_count=int(info["lap_count"]),
            lap_time=float(info["lap_time"]),
            collision=bool(info["collision"]),
            reward=reward,
            friction_mu=self._env.friction,
        )
