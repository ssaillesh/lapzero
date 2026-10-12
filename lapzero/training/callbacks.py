"""Stable-Baselines3 hooks that send reinforcement learning progress to a RunTracker.

TrackerOutputFormat forwards SB3's own training metrics; RaceMetricsCallback adds episode and
lap metrics from the contract Section 4 info dict. Both use the SB3 timestep (control steps
across all envs) as the metric step.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

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
    """Logs every finished episode's reward, length and collision, and every lap time.

    Metrics: race/episode_reward and race/episode_length (from info["episode"], which SB3's
    Monitor wrapper adds; PPO wraps envs in a Monitor by default), race/episode_collision (1.0 or
    0.0), and race/lap_time in seconds. A lap ends on the control step whose info["lap_count"]
    rises; its time is info["t"] minus the time the previous lap ended (0 at reset).
    """

    def __init__(self, tracker: RunTracker) -> None:
        """Send metrics to `tracker`."""
        super().__init__()
        self._tracker = tracker
        self._lap_counts: dict[int, int] = {}  # per env index, laps since its last reset
        self._lap_starts: dict[int, float] = {}  # per env index, s since reset the lap began

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
        self._tracker.log_metric("race/lap_time", lap_time, step=self.num_timesteps)
        logger.info("lap %d finished in %.2f s at step %d", lap_count, lap_time, self.num_timesteps)
        self._lap_counts[env_index], self._lap_starts[env_index] = lap_count, t

    def _log_episode(self, env_index: int, info: dict[str, Any]) -> None:
        """Log a finished episode and restart lap timing for env `env_index`."""
        self._lap_counts.pop(env_index, None)
        self._lap_starts.pop(env_index, None)
        step = self.num_timesteps
        episode = info.get("episode")
        if episode is not None:
            self._tracker.log_metric("race/episode_reward", float(episode["r"]), step=step)
            self._tracker.log_metric("race/episode_length", float(episode["l"]), step=step)
        self._tracker.log_metric("race/episode_collision", float(info["collision"]), step=step)
