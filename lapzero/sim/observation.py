"""Builds the contract observation vector from a raw simulator state (Section 1.2)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from lapzero.contract import (
    LIDAR_SLICE,
    OBS_DIM,
    PREV_STEER,
    PREV_THROTTLE,
    STEER,
    THROTTLE,
    V_LAT,
    V_LONG,
    V_NORM,
    VY_NORM,
    YAW_NORM,
    YAW_RATE,
)
from lapzero.sim.lidar import downsample_scan
from lapzero.sim.simulator import SimState


def normalize(value: float, scale: float) -> float:
    """Divide by `scale` and clip to [-1, 1]; NaN becomes 0.0."""
    ratio = value / scale
    if math.isnan(ratio):
        return 0.0
    return min(max(ratio, -1.0), 1.0)


def build_observation(state: SimState, prev_action: NDArray[np.float32]) -> NDArray[np.float32]:
    """Build the (OBS_DIM,) float32 observation from a sim state and the previous policy action.

    World pose is deliberately not used: it must never enter the observation.

    Args:
        state: Raw state after the last physics step.
        prev_action: Sanitized policy action from the previous control step, normalized.

    Returns:
        Observation laid out as in contract Section 1.2, inside OBS_LOW and OBS_HIGH.
    """
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[LIDAR_SLICE] = downsample_scan(state.scan)
    obs[V_LONG] = normalize(state.v_long, V_NORM)
    obs[V_LAT] = normalize(state.v_lat, VY_NORM)
    obs[YAW_RATE] = normalize(state.yaw_rate, YAW_NORM)
    obs[PREV_STEER] = prev_action[STEER]
    obs[PREV_THROTTLE] = prev_action[THROTTLE]
    return obs
