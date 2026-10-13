"""Evaluation runner: deterministic one-lap rollouts of a policy on one track.

Each episode resets the env with seed + i and drives the policy's deterministic action (no
exploration noise) until the lap is completed, the car crashes, or the episode is truncated. The
per-episode results are what the release gate compares (CLAUDE.md section 9). For robustness
diagnostics, Gaussian sensor noise can be added to what the policy observes (lapzero.eval.perturb).
Every episode keeps a per-step trace for trajectory analysis (lapzero.eval.export).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from lapzero.contract import PREV_STEER, PREV_THROTTLE
from lapzero.eval.perturb import ObservationNoise, jitter_observation
from lapzero.io.telemetry import LapStatus
from lapzero.sim.env import LapZeroEnv

logger = logging.getLogger(__name__)

_OBS_NOISE_STREAM: Final = 1
"""Seeds observation noise with (episode seed, this), a stream independent of env.reset(seed)."""


@dataclass(frozen=True, slots=True, eq=False)  # eq on array fields would raise; compare by identity
class EpisodeTrace:
    """Per-control-step record of one episode, one row per env.step(), at CTRL_HZ.

    Row i is the state after control step i, with the action taken in step i and the simulator
    command it produced. Lap telemetry frames (contract Section 5) instead pair the state before
    each step with its action; here the last row is where the episode ended, crash included.

    Attributes:
        t: Episode time at the end of the step, s.
        x: World-frame x, metres.
        y: World-frame y, metres.
        theta: Heading, rad, in (-pi, pi].
        v_long: Longitudinal speed, m/s.
        steer: Normalized steering action as the env applied it (clipped), in [-1, 1].
        throttle: Normalized throttle action as the env applied it (clipped), in [-1, 1].
        steer_rad: Steering angle commanded during the step, rad.
        v_target: Target velocity commanded during the step, m/s.
        progress: Fraction of the lap from the start line along the centreline, in [0, 1).
        cte_m: Cross-track error: signed distance from the centreline, metres, positive to the
            left of the driving direction; 0.0 if the env has no centreline.
        collision: True on the step the car hit something.
    """

    t: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    theta: NDArray[np.float64]
    v_long: NDArray[np.float64]
    steer: NDArray[np.float64]
    throttle: NDArray[np.float64]
    steer_rad: NDArray[np.float64]
    v_target: NDArray[np.float64]
    progress: NDArray[np.float64]
    cte_m: NDArray[np.float64]
    collision: NDArray[np.bool_]

    def __post_init__(self) -> None:
        rows = len(self.t)
        if rows < 1:
            raise ValueError("EpisodeTrace needs at least one row")
        for column in fields(self):
            length = len(getattr(self, column.name))
            if length != rows:
                raise ValueError(f"EpisodeTrace.{column.name} has {length} rows, t has {rows}")

    @property
    def max_speed_mps(self) -> float:
        """Highest longitudinal speed, m/s."""
        return float(np.max(self.v_long))

    @property
    def mean_abs_cte_m(self) -> float:
        """Mean distance from the centreline, metres."""
        return float(np.mean(np.abs(self.cte_m)))

    @property
    def max_abs_cte_m(self) -> float:
        """Largest distance from the centreline, metres."""
        return float(np.max(np.abs(self.cte_m)))

    @property
    def rms_cte_m(self) -> float:
        """Root mean square cross-track error, metres."""
        return float(np.sqrt(np.mean(self.cte_m**2)))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome of one evaluation episode.

    Attributes:
        seed: Seed the episode was reset with.
        status: completed, crashed, timeout or stuck (contract Section 3 episode rules).
        lap_time_s: Lap time in seconds; None unless status is completed.
        lap_fraction: Fraction of the lap driven along the centreline (1.0 for a full lap,
            negative if the car went backwards); 0.0 if the env has no centreline.
        mean_speed_mps: Mean longitudinal speed over the episode, m/s.
        steps: Control steps driven.
        trace: Per-step record; always set by run_episode. Kept out of repr, so log lines stay
            short, and out of comparisons.
    """

    seed: int
    status: LapStatus
    lap_time_s: float | None
    lap_fraction: float
    mean_speed_mps: float
    steps: int
    trace: EpisodeTrace | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """All episodes of one evaluation and their aggregate statistics."""

    episodes: tuple[EpisodeResult, ...]

    def _rate(self, status: LapStatus) -> float:
        return sum(e.status is status for e in self.episodes) / len(self.episodes)

    @property
    def completion_rate(self) -> float:
        """Fraction of episodes that completed the lap."""
        return self._rate(LapStatus.COMPLETED)

    @property
    def crash_rate(self) -> float:
        """Fraction of episodes that ended in a collision."""
        return self._rate(LapStatus.CRASHED)

    @property
    def lap_times_s(self) -> tuple[float, ...]:
        """Lap times of the completed episodes, seconds."""
        return tuple(e.lap_time_s for e in self.episodes if e.lap_time_s is not None)

    @property
    def mean_lap_time_s(self) -> float | None:
        """Mean lap time over completed episodes, seconds; None if none completed."""
        times = self.lap_times_s
        return float(np.mean(times)) if times else None


def run_episode(
    env: LapZeroEnv,
    policy: nn.Module,
    seed: int,
    *,
    obs_noise: ObservationNoise | None = None,
) -> EpisodeResult:
    """Drive one lap with the policy's deterministic action.

    Args:
        env: Env set up for one lap per episode.
        policy: Maps a (OBS_DIM,) float32 observation tensor to a (ACT_DIM,) action tensor,
            e.g. a BCPolicy (whose output is its deterministic action).
        seed: Seed for env.reset, and with _OBS_NOISE_STREAM for the observation noise.
        obs_noise: Sensor noise added to what the policy sees (never to the env); None for
            clean observations.

    Returns:
        The episode's outcome, with its trace.
    """
    policy.eval()
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng((seed, _OBS_NOISE_STREAM))
    centreline = env.centreline
    progress, lap_fraction = float(info["progress"]), 0.0
    columns: dict[str, list[float | bool]] = {column.name: [] for column in fields(EpisodeTrace)}
    terminated = truncated = False
    while not (terminated or truncated):
        seen = obs if obs_noise is None else jitter_observation(obs, obs_noise, rng)
        with torch.no_grad():
            action = policy(torch.from_numpy(seen)).numpy()
        obs, _, terminated, truncated, info = env.step(action)
        step_fraction = (float(info["progress"]) - progress + 0.5) % 1.0 - 0.5  # short way
        lap_fraction, progress = lap_fraction + step_fraction, float(info["progress"])
        x, y = float(info["x"]), float(info["y"])
        row: dict[str, float | bool] = {
            "t": float(info["t"]),
            "x": x,
            "y": y,
            "theta": float(info["theta"]),
            "v_long": float(info["v_long"]),
            # The env's clipped action, echoed in the clean observation (contract Section 1.4).
            "steer": float(obs[PREV_STEER]),
            "throttle": float(obs[PREV_THROTTLE]),
            "steer_rad": float(info["steer_rad"]),
            "v_target": float(info["v_target"]),
            "progress": progress,
            "cte_m": centreline.locate(x, y)[1] if centreline is not None else 0.0,
            "collision": bool(info["collision"]),
        }
        for name, value in row.items():
            columns[name].append(value)
    trace = EpisodeTrace(**{name: np.asarray(values) for name, values in columns.items()})
    t = float(info["t"])
    if info["collision"]:
        status = LapStatus.CRASHED
    elif terminated:
        status = LapStatus.COMPLETED
    elif t >= env.config.max_episode_s - 1e-9:
        status = LapStatus.TIMEOUT
    else:
        status = LapStatus.STUCK
    return EpisodeResult(
        seed=seed,
        status=status,
        lap_time_s=t if status is LapStatus.COMPLETED else None,
        lap_fraction=lap_fraction,
        mean_speed_mps=float(np.mean(trace.v_long)),
        steps=len(trace.t),
        trace=trace,
    )


def evaluate_policy(
    env: LapZeroEnv,
    policy: nn.Module,
    *,
    episodes: int,
    seed: int,
    obs_noise: ObservationNoise | None = None,
) -> EvalSummary:
    """Run `episodes` deterministic one-lap episodes, episode i reset with seed + i.

    Args:
        env: Env with laps_per_episode == 1.
        policy: Deterministic policy, see run_episode.
        episodes: Number of episodes, at least 1.
        seed: Seed of the first episode; logged.
        obs_noise: Sensor noise added to what the policy sees; None for clean observations.

    Returns:
        Every episode's result.

    Raises:
        ValueError: If episodes < 1 or the env runs more than one lap per episode.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if env.config.laps_per_episode != 1:
        raise ValueError(
            f"evaluation needs laps_per_episode == 1, got {env.config.laps_per_episode}"
        )
    logger.info(
        "evaluating %d episodes from seed=%d, observation noise %s", episodes, seed, obs_noise
    )
    results = []
    for i in range(episodes):
        result = run_episode(env, policy, seed + i, obs_noise=obs_noise)
        logger.info("episode %d: %s", i, result)
        results.append(result)
    return EvalSummary(tuple(results))
