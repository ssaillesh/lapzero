"""Records a driving session lap by lap into contract telemetry files (contract Section 5).

The recorder only observes: a driver (human keyboard/gamepad or model) picks actions, the caller
steps the env and hands every reset and step result to LapRecorder. Each completed or failed lap
becomes <root>/<track_id>/<lap_id>.json (validated LapTelemetry) plus <lap_id>.npz holding the
observations, actions, rewards and dones for behaviour cloning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lapzero.contract import (
    ACT_DIM,
    ACT_HIGH,
    ACT_LOW,
    ACTION_REPEAT,
    CONTRACT_VERSION,
    CTRL_HZ,
    DECIMALS_ACTION,
    DECIMALS_ANGLE,
    DECIMALS_POSITION,
    DECIMALS_PROGRESS,
    DECIMALS_SPEED,
    DECIMALS_TIME,
    LAPS_DIR,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    SIM_DT,
    SIMULATOR_NAME,
    STEER,
    THROTTLE,
    ContractViolationError,
)
from lapzero.io.telemetry import (
    DriverInfo,
    LapFrames,
    LapResult,
    LapStatus,
    LapTelemetry,
    SimInfo,
    TrackInfo,
)
from lapzero.record.ulid import new_ulid
from lapzero.sim.env import EnvConfig

logger = logging.getLogger(__name__)

_FRAME_DECIMALS = {
    "t": DECIMALS_TIME,
    "x": DECIMALS_POSITION,
    "y": DECIMALS_POSITION,
    "theta": DECIMALS_ANGLE,
    "v_long": DECIMALS_SPEED,
    "steer": DECIMALS_ACTION,
    "throttle": DECIMALS_ACTION,
    "steer_rad": DECIMALS_ANGLE,
    "v_target": DECIMALS_SPEED,
    "progress": DECIMALS_PROGRESS,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _checked_action(action: ArrayLike) -> NDArray[np.float32]:
    """Return a copy of a driver action, raising if it is outside the action space (no clipping)."""
    array = np.array(action, dtype=np.float32)
    if array.shape != (ACT_DIM,):
        raise ContractViolationError(f"action must have shape ({ACT_DIM},), got {array.shape}")
    if not (np.isfinite(array).all() and (array >= ACT_LOW).all() and (array <= ACT_HIGH).all()):
        raise ContractViolationError(f"action must be finite and in [-1, 1], got {array.tolist()}")
    return array


def _checked_observation(obs: ArrayLike) -> NDArray[np.float32]:
    """Return a copy of an observation, raising if it is outside the observation space."""
    array = np.array(obs, dtype=np.float32)
    if array.shape != (OBS_DIM,):
        raise ContractViolationError(f"observation must have shape ({OBS_DIM},), got {array.shape}")
    if not (np.isfinite(array).all() and (array >= OBS_LOW).all() and (array <= OBS_HIGH).all()):
        raise ContractViolationError("observation must be finite and inside OBS_LOW..OBS_HIGH")
    return array


@dataclass(frozen=True, slots=True)
class LapFiles:
    """The pair of files written for one lap."""

    lap_id: str
    status: LapStatus
    json_path: Path
    npz_path: Path


@dataclass(slots=True)
class _LapBuffer:
    """Samples of the lap in progress, one entry per control step."""

    index: int
    start_t: float
    frames: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in _FRAME_DECIMALS}
    )
    obs: list[NDArray[np.float32]] = field(default_factory=list)
    actions: list[NDArray[np.float32]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


class LapRecorder:
    """Turns env reset/step results into one telemetry JSON + .npz pair per lap."""

    def __init__(
        self,
        track: TrackInfo,
        driver: DriverInfo,
        *,
        sim_version: str,
        env_config: EnvConfig | None = None,
        root: Path = Path(LAPS_DIR),
        clock: Callable[[], datetime] = _utc_now,
        new_id: Callable[[], str] = new_ulid,
    ) -> None:
        """Set up a recorder for one track and driver.

        Args:
            track: Track metadata written to every lap.
            driver: Driver metadata; human drivers name their keyboard or gamepad.
            sim_version: Installed simulator version (importlib.metadata.version("f1tenth-gym")).
            env_config: Config of the env being recorded (speed cap, time limit); contract defaults.
            root: Directory receiving <track_id>/<lap_id>.json and .npz files.
            clock: Returns the current UTC time; injectable for tests.
            new_id: Returns a new ULID; injectable for tests.
        """
        self._track = track
        self._driver = driver
        self._sim_version = sim_version
        self._config = env_config or EnvConfig()
        self._root = root
        self._clock = clock
        self._new_id = new_id
        self._episode_id = ""
        self._seed = 0
        self._lap_count = 0
        self._lap: _LapBuffer | None = None
        self._last_obs: NDArray[np.float32] | None = None
        self._last_info: Mapping[str, Any] | None = None

    def start_episode(self, obs: ArrayLike, info: Mapping[str, Any], *, seed: int) -> None:
        """Begin recording from an env.reset() result.

        An unfinished lap from an earlier episode is discarded: it has no contract status.

        Args:
            obs: Observation returned by env.reset().
            info: Info dict returned by env.reset().
            seed: Seed passed to env.reset(); written to every lap of the episode.

        Raises:
            ContractViolationError: If `obs` is outside the observation space.
        """
        if self._lap is not None and self._lap.actions:
            logger.warning(
                "discarding unfinished lap %d of episode %s", self._lap.index, self._episode_id
            )
        self._last_obs = _checked_observation(obs)
        self._last_info = info
        self._episode_id = self._new_id()
        self._seed = seed
        self._lap_count = int(info["lap_count"])
        self._lap = _LapBuffer(index=1, start_t=float(info["t"]))
        logger.info("recording episode %s with seed=%d", self._episode_id, seed)

    def record_step(
        self,
        action: ArrayLike,
        obs: ArrayLike,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> list[LapFiles]:
        """Record one control step and write every lap it completes or ends.

        Each frame pairs the state before the step with the action taken in it and the simulator
        command that action produced, i.e. the (observation, action) pairs used for cloning.

        Args:
            action: Normalized [steer, throttle] the driver passed to env.step().
            obs: Observation returned by env.step().
            reward: Reward returned by env.step().
            terminated: Terminated flag returned by env.step().
            truncated: Truncated flag returned by env.step().
            info: Info dict returned by env.step().

        Returns:
            Files written by this step: none, or one entry per finished lap.

        Raises:
            RuntimeError: If no episode is being recorded.
            ContractViolationError: If the action, observation or reward breaks the contract.
        """
        lap, prev_obs, prev = self._lap, self._last_obs, self._last_info
        if lap is None or prev_obs is None or prev is None:
            raise RuntimeError(
                "call start_episode() before record_step() and after an episode ends"
            )
        act = _checked_action(action)
        next_obs = _checked_observation(obs)
        if not math.isfinite(reward):
            raise ContractViolationError(f"reward must be finite, got {reward}")
        frame = {
            "t": prev["t"],
            "x": prev["x"],
            "y": prev["y"],
            "theta": prev["theta"],
            "v_long": prev["v_long"],
            "steer": act[STEER],
            "throttle": act[THROTTLE],
            "steer_rad": info["steer_rad"],
            "v_target": info["v_target"],
            "progress": prev["progress"],
        }
        for name, value in frame.items():
            lap.frames[name].append(float(value))
        lap.obs.append(prev_obs)
        lap.actions.append(act)
        lap.rewards.append(float(reward))
        lap.dones.append(terminated or truncated)
        self._last_obs, self._last_info = next_obs, info

        written = []
        if int(info["lap_count"]) > self._lap_count:
            self._lap_count = int(info["lap_count"])
            written.append(self._write_lap(lap, LapStatus.COMPLETED, info))
            lap = self._lap = _LapBuffer(index=lap.index + 1, start_t=float(info["t"]))
        if terminated or truncated:
            if lap.actions:
                written.append(self._write_lap(lap, self._failure_status(info, truncated), info))
            self._lap = self._last_obs = self._last_info = None
        return written

    def _failure_status(self, info: Mapping[str, Any], truncated: bool) -> LapStatus:
        """Classify an episode that ended mid-lap (contract Section 3 end conditions)."""
        if info["collision"]:
            return LapStatus.CRASHED
        if not truncated:
            raise ContractViolationError("episode terminated without a collision or a finished lap")
        timed_out = round(float(info["t"]) / SIM_DT) >= round(self._config.max_episode_s / SIM_DT)
        return LapStatus.TIMEOUT if timed_out else LapStatus.STUCK

    def _write_lap(self, lap: _LapBuffer, status: LapStatus, info: Mapping[str, Any]) -> LapFiles:
        """Validate the lap as LapTelemetry and write its .npz, then its .json."""
        speeds = lap.frames["v_long"]
        lap_time = float(info["t"]) - lap.start_t
        telemetry = LapTelemetry(
            contract_version=CONTRACT_VERSION,
            lap_id=self._new_id(),
            episode_id=self._episode_id,
            lap_index=lap.index,
            created_at=self._clock().replace(microsecond=0),
            track=self._track,
            driver=self._driver,
            sim=SimInfo(
                simulator=SIMULATOR_NAME,
                sim_version=self._sim_version,
                sim_dt=SIM_DT,
                action_repeat=ACTION_REPEAT,
                ctrl_hz=CTRL_HZ,
                v_max_train=self._config.v_max,
                seed=self._seed,
            ),
            result=LapResult(
                status=status,
                lap_time_s=round(lap_time, DECIMALS_TIME)
                if status is LapStatus.COMPLETED
                else None,
                collision=bool(info["collision"]),
                off_track_events=0,  # F1TENTH tracks are walled: leaving the track is a collision
                max_speed_mps=round(max(speeds), DECIMALS_SPEED),
                mean_speed_mps=round(sum(speeds) / len(speeds), DECIMALS_SPEED),
                sample_count=len(speeds),
            ),
            frames=LapFrames.model_validate(
                {
                    name: [round(value, _FRAME_DECIMALS[name]) for value in values]
                    for name, values in lap.frames.items()
                }
            ),
        )
        directory = self._root / self._track.track_id
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{telemetry.lap_id}.json"
        npz_path = directory / f"{telemetry.lap_id}.npz"
        np.savez(  # .npz first, so an existing .json always has its arrays next to it
            npz_path,
            obs=np.stack(lap.obs),
            actions=np.stack(lap.actions),
            rewards=np.asarray(lap.rewards, dtype=np.float32),
            dones=np.asarray(lap.dones, dtype=np.bool_),
            contract_version=np.asarray(CONTRACT_VERSION),
        )
        json_path.write_text(telemetry.model_dump_json(by_alias=True))
        logger.info(
            "wrote %s lap %s (%d samples) to %s", status, telemetry.lap_id, len(speeds), json_path
        )
        return LapFiles(telemetry.lap_id, status, json_path, npz_path)
