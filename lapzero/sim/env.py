"""Gymnasium env that puts a physics backend behind the LapZero state space contract.

This module is the only place that converts policy actions in [-1, 1] into simulator commands
(steering angle in rad, target velocity in m/s), and LapZeroEnv is the only caller of
Simulator.step. The F1TENTH Gym backend and its make_env factory live in lapzero/sim/f1tenth.py.
Contract: docs/contracts/state_space.md, Sections 1 to 4.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from lapzero.contract import (
    ACCEL_STEP,
    ACT_DIM,
    ACT_HIGH,
    ACT_LOW,
    ACTION_REPEAT,
    BRAKE_STEP,
    LAP_MIN_PROGRESS,
    LAPS_PER_EPISODE_EVAL,
    MAX_EPISODE_S,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    REVERSE_ENABLED,
    SIM_DT,
    SIM_V_MAX,
    SIM_V_MIN,
    START_OFFSET_THETA,
    START_OFFSET_XY,
    STEER,
    STEER_MAX,
    STEER_RATE_MAX,
    STUCK_S,
    STUCK_SPEED,
    THROTTLE,
    V_MAX_TRAIN,
)
from lapzero.sim.observation import build_observation
from lapzero.sim.simulator import Pose, SimState, Simulator
from lapzero.sim.track import Centreline

logger = logging.getLogger(__name__)

RewardFn = Callable[[SimState, SimState], float]
"""Reward for one physics step from the states before and after it; summed per control step."""


def zero_reward(prev: SimState, curr: SimState) -> float:
    """Default per-physics-step reward; training adds its own with lapzero.sim.reward.RacingReward.

    Args:
        prev: State before the physics step.
        curr: State after the physics step.

    Returns:
        Always 0.0.
    """
    return 0.0


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Episode and action-mapping settings. Defaults are the contract values.

    Attributes:
        laps_per_episode: Laps that end the episode (contract: 1 for eval, 3 for training).
        max_episode_s: Episode time limit, seconds.
        v_max: Target velocity cap, m/s, at most SIM_V_MAX.
        accel_step: Target velocity gain per control step at full throttle, m/s.
        brake_step: Target velocity loss per control step at full brake, m/s.
        steer_rate_limit: Limit steering change to STEER_RATE_MAX rad per control step.
        randomize_start: Jitter the start pose by START_OFFSET_XY m and START_OFFSET_THETA rad
            (training only).
        friction_range: (low, high) tyre-road friction coefficient mu (dimensionless) to draw
            uniformly on every reset (domain randomization, training only), or None to leave
            the simulator's friction alone. Not a contract value.
    """

    laps_per_episode: int = LAPS_PER_EPISODE_EVAL
    max_episode_s: float = MAX_EPISODE_S
    v_max: float = V_MAX_TRAIN
    accel_step: float = ACCEL_STEP
    brake_step: float = BRAKE_STEP
    steer_rate_limit: bool = True
    randomize_start: bool = False
    friction_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.laps_per_episode < 1:
            raise ValueError(
                f"EnvConfig.laps_per_episode must be >= 1, got {self.laps_per_episode}"
            )
        for name in ("max_episode_s", "accel_step", "brake_step"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"EnvConfig.{name} must be finite and > 0, got {value}")
        if not 0.0 < self.v_max <= SIM_V_MAX:
            raise ValueError(f"EnvConfig.v_max must be in (0, {SIM_V_MAX}] m/s, got {self.v_max}")
        if self.friction_range is not None:
            low, high = self.friction_range
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
                raise ValueError(
                    "EnvConfig.friction_range must be finite with 0 < low <= high, "
                    f"got {self.friction_range}"
                )


def _clip(value: float, low: float, high: float, nan_to: float) -> float:
    """Clip `value` to [low, high]; NaN becomes `nan_to` and +/-inf the matching bound."""
    if math.isnan(value):
        return nan_to
    return min(max(value, low), high)


def wrap_angle(theta: float) -> float:
    """Wrap an angle in rad to (-pi, pi]."""
    wrapped = math.pi - (math.pi - theta) % (2.0 * math.pi)
    return math.pi if wrapped <= -math.pi else wrapped


def sanitize_action(action: ArrayLike) -> NDArray[np.float32]:
    """Return a policy action as finite float32 values clipped to [-1, 1].

    Out-of-range values are clipped, never rejected. NaN becomes 0.0 (straight, hold speed) and
    is logged because it means the policy is broken.

    Args:
        action: Normalized policy action [steer, throttle].

    Returns:
        Array of shape (ACT_DIM,), dtype float32, every element in [-1, 1].

    Raises:
        ValueError: If `action` does not have shape (ACT_DIM,).
    """
    raw = np.asarray(action, dtype=np.float64)
    if raw.shape != (ACT_DIM,):
        raise ValueError(f"action must have shape ({ACT_DIM},), got {raw.shape}")
    if np.isnan(raw).any():
        logger.warning("NaN in policy action %s replaced with 0.0", raw)
    finite = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(finite, -1.0, 1.0).astype(np.float32)


def steer_to_rad(steer: float, prev_steer_rad: float, rate_max: float | None) -> float:
    """Map normalized steering to a steering angle (contract Section 2.2).

    Args:
        steer: -1 full right to +1 full left. Clipped to [-1, 1]; NaN means 0.
        prev_steer_rad: Steering angle applied on the previous control step, rad.
        rate_max: Largest change per control step in rad, or None for no rate limit.

    Returns:
        Steering angle in rad, within +/-STEER_MAX.
    """
    steer_rad = _clip(steer, -1.0, 1.0, nan_to=0.0) * STEER_MAX
    if rate_max is not None:
        low, high = prev_steer_rad - rate_max, prev_steer_rad + rate_max
        steer_rad = _clip(steer_rad, low, high, nan_to=prev_steer_rad)
    return steer_rad


def throttle_to_v_target(
    throttle: float,
    v_long: float,
    v_max: float = V_MAX_TRAIN,
    accel_step: float = ACCEL_STEP,
    brake_step: float = BRAKE_STEP,
) -> float:
    """Map normalized throttle/brake to a target velocity (contract Section 2.3).

    Args:
        throttle: +1 full throttle, 0 hold speed, -1 full brake. Clipped; NaN means 0.
        v_long: Current longitudinal speed, m/s.
        v_max: Target velocity cap, m/s.
        accel_step: Target velocity gain per control step at full throttle, m/s.
        brake_step: Target velocity loss per control step at full brake, m/s.

    Returns:
        Target velocity in m/s, within [0, v_max] (or [SIM_V_MIN, v_max] with reverse enabled).
    """
    a = _clip(throttle, -1.0, 1.0, nan_to=0.0)
    v_floor = SIM_V_MIN if REVERSE_ENABLED else 0.0
    v_target = v_long + a * (accel_step if a >= 0.0 else brake_step)
    return _clip(v_target, v_floor, v_max, nan_to=v_floor)


class LapZeroEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Single-car racing env with the contract observation (113,) and action (2,) spaces.

    Each step maps the action once, then runs ACTION_REPEAT physics steps (stopping early on a
    collision) and sums the per-physics-step rewards.

    The env, not the simulator, decides when a lap is complete (contract Section 3): the
    simulator's lap counter only marks a pass through the start line, and the pass counts as a
    lap if the car has driven at least LAP_MIN_PROGRESS of the centreline since reset or the
    previous lap, net of any distance driven backwards.
    """

    def __init__(
        self,
        simulator: Simulator,
        start_pose: Pose,
        reward_fn: RewardFn = zero_reward,
        config: EnvConfig | None = None,
        centreline: Centreline | None = None,
    ) -> None:
        """Create the env around an already started simulator.

        Args:
            simulator: Physics backend, e.g. lapzero.sim.f1tenth.F1TenthSimulator.
            start_pose: Fixed start pose for this track (from tracks/<track_id>/start.yaml).
            reward_fn: Per-physics-step reward; defaults to the 0.0 placeholder.
            config: Episode and action-mapping settings; defaults to the contract values.
            centreline: Track centreline in driving order, for info["progress"] and lap
                completion; without one progress is always 0.0 and every lap the simulator
                counts is a lap.
        """
        super().__init__()
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(ACT_LOW, ACT_HIGH, shape=(ACT_DIM,), dtype=np.float32)
        self._sim = simulator
        self._start_pose = start_pose
        self._reward_fn = reward_fn
        self._config = config or EnvConfig()
        self._centreline = centreline
        # Progress is measured from the fixed start pose (the start line), not a jittered one.
        self._start_arc = centreline.project(start_pose.x, start_pose.y) if centreline else 0.0
        self._max_steps = round(self._config.max_episode_s / SIM_DT)
        self._stuck_limit = round(STUCK_S / SIM_DT)
        self._state: SimState | None = None
        self._done = False
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._steer_rad = 0.0
        self._v_target = 0.0
        self._sim_steps = 0
        self._stuck_steps = 0
        self._lap_count = 0
        self._lap_start_step = 0
        self._sim_lap_count = 0
        self._progress = 0.0
        self._lap_progress = 0.0  # net lap fraction driven since reset or the last lap

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a new episode at the start pose (jittered if randomize_start is set).

        With friction_range set, the episode's friction mu is drawn uniformly from it.

        Args:
            seed: Seeds the start-pose jitter and friction draw; logged when given.
            options: Unused, accepted for the Gymnasium API.

        Returns:
            The first observation and info dict.
        """
        super().reset(seed=seed)
        if seed is not None:
            logger.info("LapZeroEnv reset with seed=%d", seed)
        pose = self._start_pose
        if self._config.randomize_start:
            dx, dy = self.np_random.uniform(-START_OFFSET_XY, START_OFFSET_XY, size=2)
            dtheta = self.np_random.uniform(-START_OFFSET_THETA, START_OFFSET_THETA)
            pose = Pose(pose.x + float(dx), pose.y + float(dy), pose.theta + float(dtheta))
        # Drawn after the jitter so a seed gives the same start pose with or without it.
        if self._config.friction_range is not None:
            mu = float(self.np_random.uniform(*self._config.friction_range))
            self._sim.set_friction(mu)
            logger.debug("LapZeroEnv episode friction mu=%.4f", mu)
        self._state = self._sim.reset(pose)
        self._done = False
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._steer_rad = 0.0
        self._v_target = 0.0
        self._sim_steps = 0
        self._stuck_steps = 0
        self._lap_count = 0
        self._lap_start_step = 0
        self._sim_lap_count = self._state.lap_count
        self._progress = self._progress_at(self._state.pose)
        self._lap_progress = 0.0
        return build_observation(self._state, self._prev_action), self._info(self._state)

    def step(
        self, action: ArrayLike
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one policy action for one control step (ACTION_REPEAT physics steps).

        Args:
            action: Normalized [steer, throttle]; out-of-range values are clipped.

        Returns:
            observation, summed reward, terminated, truncated, info (contract Section 4).

        Raises:
            RuntimeError: If called before reset() or after the episode ended.
            ValueError: If `action` has the wrong shape or reward_fn returns a non-finite value.
        """
        if self._state is None or self._done:
            raise RuntimeError("call reset() before step() and after an episode ends")
        cfg = self._config
        act = sanitize_action(action)
        rate_max = STEER_RATE_MAX if cfg.steer_rate_limit else None
        self._steer_rad = steer_to_rad(float(act[STEER]), self._steer_rad, rate_max)
        self._v_target = throttle_to_v_target(
            float(act[THROTTLE]), self._state.v_long, cfg.v_max, cfg.accel_step, cfg.brake_step
        )

        state, reward, steps = self._state, 0.0, 0
        while steps < ACTION_REPEAT and not state.collision:
            prev, state = state, self._sim.step(self._steer_rad, self._v_target)
            step_reward = self._reward_fn(prev, state)
            if not math.isfinite(step_reward):
                raise ValueError(f"reward_fn returned a non-finite reward: {step_reward}")
            reward += step_reward
            steps += 1
        self._state = state
        self._prev_action = act
        self._sim_steps += steps
        self._stuck_steps = self._stuck_steps + steps if state.v_long < STUCK_SPEED else 0
        self._update_laps(state)

        terminated = state.collision or self._lap_count >= cfg.laps_per_episode
        truncated = self._sim_steps >= self._max_steps or self._stuck_steps >= self._stuck_limit
        self._done = terminated or truncated
        obs = build_observation(state, self._prev_action)
        return obs, reward, terminated, truncated, self._info(state)

    def close(self) -> None:
        """Close the underlying simulator."""
        self._sim.close()

    @property
    def config(self) -> EnvConfig:
        """Episode and action-mapping settings the env runs with."""
        return self._config

    @property
    def friction(self) -> float:
        """Tyre-road friction coefficient mu of the current episode (dimensionless).

        Kept out of the info dict, whose keys are fixed by contract Section 4.
        """
        return self._sim.friction

    @property
    def v_lat(self) -> float:
        """Car-frame lateral speed (y left) of the latest reading, m/s, raw and unclipped.

        Kept out of the info dict, whose keys are fixed by contract Section 4; the observation
        holds it normalized and clipped to +/-VY_NORM.

        Raises:
            RuntimeError: If called before reset().
        """
        if self._state is None:
            raise RuntimeError("call reset() before reading v_lat")
        return self._state.v_lat

    @property
    def centreline(self) -> Centreline | None:
        """Centreline that info["progress"] is measured along, or None (progress stays 0.0)."""
        return self._centreline

    def _progress_at(self, pose: Pose) -> float:
        """Fraction of a lap driven from the start line to `pose` along the centreline, in [0, 1].

        The car's position is projected onto the centreline; 0.0 is the start pose's
        projection and the value wraps back to 0.0 each lap. Always 0.0 without a centreline.
        """
        if self._centreline is None:
            return 0.0
        arc = self._centreline.project(pose.x, pose.y)
        return (arc - self._start_arc) / self._centreline.length % 1.0

    def _update_laps(self, state: SimState) -> None:
        """Track net progress and count a lap if the simulator reports a start-line pass.

        F1TENTH counts any pass through a gate at the reset pose, in either direction, so a
        U-turn or a spin back through the start would be a lap. A pass counts only once the car
        has driven LAP_MIN_PROGRESS of the lap since reset or the previous lap, net of distance
        driven backwards (contract Section 3). Without a centreline every pass counts.

        Args:
            state: Reading after the control step's last physics step.
        """
        progress = self._progress_at(state.pose)
        self._lap_progress += (progress - self._progress + 0.5) % 1.0 - 0.5  # short way round
        self._progress = progress
        crossed, self._sim_lap_count = state.lap_count > self._sim_lap_count, state.lap_count
        if not crossed:
            return
        if self._centreline is not None and self._lap_progress < LAP_MIN_PROGRESS:
            logger.debug(
                "start-line pass at t=%.2f s is not a lap: only %.1f%% of the lap driven",
                self._sim_steps * SIM_DT,
                100.0 * self._lap_progress,
            )
            return
        self._lap_count += 1
        self._lap_start_step = self._sim_steps
        self._lap_progress = 0.0

    def _info(self, state: SimState) -> dict[str, Any]:
        """Build the contract Section 4 info dict (raw SI values, not seen by the policy).

        `state` must be the latest reading, the one info["progress"] was last measured at.
        """
        return {
            "t": self._sim_steps * SIM_DT,
            "x": state.pose.x,
            "y": state.pose.y,
            "theta": wrap_angle(state.pose.theta),
            "v_long": state.v_long,
            "steer_rad": self._steer_rad,
            "v_target": self._v_target,
            "lap_count": self._lap_count,
            "lap_time": (self._sim_steps - self._lap_start_step) * SIM_DT,
            "collision": state.collision,
            "progress": self._progress,
        }
