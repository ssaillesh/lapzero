"""Racing reward for reinforcement learning: centreline progress, speed, collisions, smoothness.

The reward is not part of the state space contract (contract Section 3). RacingReward reads only
LapZeroEnv's contract Section 4 info dict (including the env's own info["progress"]) and the
observation, so it never changes what the policy sees. Bump REWARD_VERSION whenever the reward
changes; every training run logs it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from lapzero.contract import CTRL_DT, PREV_STEER, SIM_V_MAX
from lapzero.sim.env import LapZeroEnv

REWARD_VERSION: Final = "1.0.0"

MAX_PROGRESS_STEP: Final = SIM_V_MAX * CTRL_DT
"""Furthest the car can travel in one control step, metres. Progress per step is clipped to it,
so a projection that jumps to a nearby part of the track (hairpins, close straights) cannot be
farmed for reward."""

Obs = NDArray[np.float32]
Action = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights of the racing reward terms; logged with REWARD_VERSION on every training run.

    Attributes:
        progress_weight: Reward per metre of progress along the centreline.
        speed_weight: Reward per m/s of forward speed, each control step.
        collision_penalty: Subtracted once, on the control step that collides.
        steer_jerk_weight: Penalty per unit of squared change in normalized steering between
            consecutive control steps (the largest change, -1 to +1, costs 4x this).
    """

    progress_weight: float = 1.0
    speed_weight: float = 0.01
    collision_penalty: float = 50.0
    steer_jerk_weight: float = 0.1

    def __post_init__(self) -> None:
        for name in (f.name for f in dataclasses.fields(self)):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"RewardConfig.{name} must be finite and >= 0, got {value}")


class RacingReward(gym.Wrapper[Obs, Action, Obs, Action]):
    """Replaces a LapZeroEnv's reward with the racing reward, once per control step.

    reward = progress_weight * progress            (m along the centreline, +/-MAX_PROGRESS_STEP)
           + speed_weight * max(v_long, 0)         (m/s)
           - steer_jerk_weight * (steer - prev_steer)**2   (normalized policy steering)
           - collision_penalty                     (only on the step that collides)

    Progress is the change in the env's info["progress"] lap fraction, taken the short way round
    the loop (so crossing the start line counts forward) and scaled by the centreline length.
    The steering terms are the observation's PREV_STEER entries, i.e. the sanitized action the
    env applied. A collision ends the episode: LapZeroEnv terminates on it (contract Section 3).
    The env's own reward is discarded, so build it with the default zero reward.
    """

    def __init__(self, env: LapZeroEnv, config: RewardConfig | None = None) -> None:
        """Wrap a LapZeroEnv that measures progress along a centreline.

        Args:
            env: Env to wrap; must be built with a centreline.
            config: Reward weights; defaults to RewardConfig().

        Raises:
            ValueError: If the env has no centreline, so info["progress"] would always be 0.0.
        """
        if env.centreline is None:
            raise ValueError(
                "RacingReward needs a LapZeroEnv built with a centreline; without one "
                'info["progress"] is always 0.0 and progress would never be rewarded'
            )
        super().__init__(env)
        self.config = config or RewardConfig()
        self._track_length = env.centreline.length
        self._progress = 0.0
        self._steer = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        """Reset the wrapped env and take its first reading as the progress reference."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._progress = float(info["progress"])
        self._steer = float(obs[PREV_STEER])
        return obs, info

    def step(self, action: Action) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        """Step the wrapped env and return its transition with the racing reward."""
        obs, _, terminated, truncated, info = self.env.step(action)
        cfg = self.config
        progress = float(info["progress"])
        lap_fraction = (progress - self._progress + 0.5) % 1.0 - 0.5  # short way round
        metres = min(max(lap_fraction * self._track_length, -MAX_PROGRESS_STEP), MAX_PROGRESS_STEP)
        steer = float(obs[PREV_STEER])
        reward = (
            cfg.progress_weight * metres
            + cfg.speed_weight * max(float(info["v_long"]), 0.0)
            - cfg.steer_jerk_weight * (steer - self._steer) ** 2
        )
        if info["collision"]:
            reward -= cfg.collision_penalty
        self._progress, self._steer = progress, steer
        return obs, reward, terminated, truncated, info
