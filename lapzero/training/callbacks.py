"""Stable-Baselines3 hooks that send reinforcement learning progress to a RunTracker.

TrackerOutputFormat forwards SB3's own training metrics; RaceMetricsCallback adds episode and
lap metrics from the contract Section 4 info dict. Both use the SB3 timestep (control steps
across all envs) as the metric step. PolicyCheckpointCallback saves the actor periodically.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from lapzero.training.policy import BCPolicy, save_policy
from lapzero.training.tracking import RunTracker

logger = logging.getLogger(__name__)


class TrackerOutputFormat(KVWriter):
    """SB3 logger output that records every finite numeric value as a tracker metric.

    SB3 dumps its logger once per update with keys such as train/approx_kl, train/value_loss,
    train/clip_fraction and rollout/ep_rew_mean. Non-numeric and non-finite values (SB3 reports
    NaN explained variance for constant returns) are skipped, so no NaN reaches the tracker.
    """

    def __init__(self, tracker: RunTracker) -> None:
        """Send metrics to `tracker`."""
        self._tracker = tracker

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0
    ) -> None:
        """Log each finite number in `key_values` at `step` and summarize them at INFO level.

        Args:
            key_values: Metric name to value, as recorded on the SB3 logger.
            key_excluded: Ignored; exclusions name SB3's built-in formats (stdout, tensorboard).
            step: SB3 timestep of the dump.
        """
        logged: dict[str, float] = {}
        for key, value in sorted(key_values.items()):
            number = float(value) if isinstance(value, numbers.Real) else math.nan
            if not math.isfinite(number):
                logger.debug("metric %s=%r skipped: not a finite number", key, value)
                continue
            self._tracker.log_metric(key, number, step=step)
            logged[key] = number
        logger.info("step %d: %s", step, " ".join(f"{k}={v:.4g}" for k, v in logged.items()))

    def close(self) -> None:
        """Release nothing: the tracker's run is closed by whoever opened it."""


class RaceMetricsCallback(BaseCallback):
    """Logs every finished episode's reward, length, collision and friction, and every lap time.

    Metrics: race/episode_reward and race/episode_length (from info["episode"], which SB3's
    Monitor wrapper adds; PPO wraps envs in a Monitor by default), race/episode_collision (1.0 or
    0.0), and race/lap_time in seconds. A lap ends on the control step whose info["lap_count"]
    rises; its time is info["t"] minus the time the previous lap ended (0 at reset).

    When the envs expose LapZeroEnv.friction, race/episode_friction_mu is the friction mu the
    episode ran with. It is read when the episode starts: the VecEnv resets a finished env before
    the callback sees it, so reading at the end would give the next episode's value.

    With track_ids, every metric is also logged per track as race/<track_id>/<name> (e.g.
    race/spielberg/lap_time), so a multi-track run shows each track's progress separately.
    """

    def __init__(self, tracker: RunTracker, track_ids: Sequence[str] | None = None) -> None:
        """Send metrics to `tracker`.

        Args:
            tracker: Receives the metrics.
            track_ids: Track driven by each env of the training VecEnv, in env order, or None
                to log only the metrics pooled over all envs.
        """
        super().__init__()
        self._tracker = tracker
        self._track_ids = None if track_ids is None else list(track_ids)
        self._lap_counts: dict[int, int] = {}  # per env index, laps since its last reset
        self._lap_starts: dict[int, float] = {}  # per env index, s since reset the lap began
        self._frictions: dict[int, float] = {}  # per env index, mu of its current episode

    def _on_training_start(self) -> None:
        """Check there is a track id per env, and read each env's friction if they expose one.

        Raises:
            ValueError: If track_ids was given and its length differs from the number of envs.
        """
        if self._track_ids is not None and len(self._track_ids) != self.training_env.num_envs:
            raise ValueError(
                f"RaceMetricsCallback got {len(self._track_ids)} track_ids for "
                f"{self.training_env.num_envs} envs"
            )
        if self.training_env.has_attr("friction"):
            frictions = self.training_env.get_attr("friction")
            self._frictions = {index: float(mu) for index, mu in enumerate(frictions)}

    def _on_step(self) -> bool:
        """Log the laps and episodes that ended on this step, for every env; never stops training.

        Returns:
            True, so training continues.
        """
        infos: list[dict[str, Any]] = self.locals["infos"]
        dones = self.locals["dones"]
        for env_index, (info, done) in enumerate(zip(infos, dones, strict=True)):
            self._log_lap(env_index, info)
            if done:
                self._log_episode(env_index, info)
        return True

    def _log_lap(self, env_index: int, info: dict[str, Any]) -> None:
        """Log a lap time if env `env_index` just completed a lap."""
        lap_count = int(info["lap_count"])
        if lap_count <= self._lap_counts.get(env_index, 0):
            return
        t = float(info["t"])
        lap_time = t - self._lap_starts.get(env_index, 0.0)
        self._log_metric(env_index, "lap_time", lap_time)
        logger.info(
            "lap %d finished in %.2f s at step %d%s",
            lap_count,
            lap_time,
            self.num_timesteps,
            "" if self._track_ids is None else f" on {self._track_ids[env_index]}",
        )
        self._lap_counts[env_index], self._lap_starts[env_index] = lap_count, t

    def _log_episode(self, env_index: int, info: dict[str, Any]) -> None:
        """Log a finished episode and restart lap timing for env `env_index`."""
        self._lap_counts.pop(env_index, None)
        self._lap_starts.pop(env_index, None)
        episode = info.get("episode")
        if episode is not None:
            self._log_metric(env_index, "episode_reward", float(episode["r"]))
            self._log_metric(env_index, "episode_length", float(episode["l"]))
        self._log_metric(env_index, "episode_collision", float(info["collision"]))
        if env_index in self._frictions:
            self._log_metric(env_index, "episode_friction_mu", self._frictions[env_index])
            (next_friction,) = self.training_env.get_attr("friction", [env_index])
            self._frictions[env_index] = float(next_friction)

    def _log_metric(self, env_index: int, name: str, value: float) -> None:
        """Log race/<name> and, with track ids, race/<track_id>/<name> at the current step."""
        step = self.num_timesteps
        self._tracker.log_metric(f"race/{name}", value, step=step)
        if self._track_ids is not None:
            self._tracker.log_metric(f"race/{self._track_ids[env_index]}/{name}", value, step=step)


class PolicyCheckpointCallback(BaseCallback):
    """Saves the deterministic actor every `interval` timesteps, to evaluate training mid-way.

    Files are <directory>/<prefix>_step_<N>.pt, written by save_policy, with N the multiple of
    `interval` just reached, so scripts/eval.py --checkpoint and scripts/eval_sweep.py load them.
    A checkpoint holds the actor as of the last PPO update; the rollout in progress has not
    changed it yet.
    """

    def __init__(
        self, interval: int, directory: Path, prefix: str, actor: Callable[[], BCPolicy]
    ) -> None:
        """Save `actor()` to `directory` every `interval` timesteps.

        Args:
            interval: Timesteps (control steps across all envs) between checkpoints, >= 1.
            directory: Where to write the checkpoints; created if missing.
            prefix: File name before "_step_<N>.pt", e.g. the output model's stem.
            actor: Returns the current deterministic actor, e.g. export_actor(model.policy).

        Raises:
            ValueError: If `interval` < 1.
        """
        super().__init__()
        if interval < 1:
            raise ValueError(f"checkpoint interval must be >= 1 timestep, got {interval}")
        self._interval = interval
        self._directory = directory
        self._prefix = prefix
        self._actor = actor
        self._next = interval
        self.saved: list[Path] = []

    def _on_step(self) -> bool:
        """Save a checkpoint once the timestep count reaches the next multiple of `interval`.

        Returns:
            True, so training continues.
        """
        if self.num_timesteps < self._next:
            return True
        step = self.num_timesteps // self._interval * self._interval
        path = self._directory / f"{self._prefix}_step_{step}.pt"
        save_policy(self._actor(), path)
        logger.info("saved checkpoint %s at step %d", path, self.num_timesteps)
        self.saved.append(path)
        self._next = step + self._interval
        return True
