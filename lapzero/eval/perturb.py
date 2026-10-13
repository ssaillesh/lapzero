"""Evaluation-only sensor noise: Gaussian jitter on the observation the policy sees.

The env always emits the clean contract observation. Noise is added between the env and the
policy, so it never reaches recorded laps or training data. Physics perturbations such as surface
friction belong to the simulator backend (lapzero.sim.f1tenth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lapzero.contract import (
    LIDAR_MAX,
    LIDAR_SLICE,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    V_LAT,
    V_LONG,
    V_NORM,
    VY_NORM,
    YAW_NORM,
    YAW_RATE,
    ContractViolationError,
)


@dataclass(frozen=True, slots=True)
class ObservationNoise:
    """Standard deviations of zero-mean Gaussian noise on the observation, in SI units.

    Attributes:
        lidar_std_m: Added to every downsampled LiDAR beam, metres.
        speed_std_mps: Added to v_long and to v_lat, m/s.
        yaw_rate_std_radps: Added to the yaw rate, rad/s.
    """

    lidar_std_m: float = 0.0
    speed_std_mps: float = 0.0
    yaw_rate_std_radps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lidar_std_m", "speed_std_mps", "yaw_rate_std_radps"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"ObservationNoise.{name} must be finite and >= 0, got {value}")

    @property
    def is_zero(self) -> bool:
        """True when no noise is added at all."""
        return self.lidar_std_m == self.speed_std_mps == self.yaw_rate_std_radps == 0.0


def jitter_observation(
    obs: ArrayLike, noise: ObservationNoise, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Add Gaussian noise to an observation's sensor entries and clip back into the contract bounds.

    Noise goes on the downsampled beams, not the raw scan, so it stays zero-mean before clipping;
    near a bound the clip makes it one-sided. The previous-action entries are the policy's own
    outputs, not sensor readings, and are never perturbed. One normal is drawn per entry whatever
    the stds, so a seeded `rng` gives the same stream for every noise level. NaN is read as in
    build_observation: no LiDAR return (1.0), or 0.0 for the other entries.

    Args:
        obs: Observation of shape (OBS_DIM,), normalized as in contract Section 1.2.
        noise: Standard deviations in SI units.
        rng: Seeded generator the noise is drawn from.

    Returns:
        Array of shape (OBS_DIM,), dtype float32, inside OBS_LOW and OBS_HIGH.

    Raises:
        ContractViolationError: If `obs` does not have shape (OBS_DIM,).
    """
    clean = np.asarray(obs, dtype=np.float64)
    if clean.shape != (OBS_DIM,):
        raise ContractViolationError(f"obs must have shape ({OBS_DIM},), got {clean.shape}")
    nan_value = np.zeros(OBS_DIM)
    nan_value[LIDAR_SLICE] = 1.0
    clean = np.where(np.isnan(clean), nan_value, clean)
    std = np.zeros(OBS_DIM)
    std[LIDAR_SLICE] = noise.lidar_std_m / LIDAR_MAX
    std[V_LONG] = noise.speed_std_mps / V_NORM
    std[V_LAT] = noise.speed_std_mps / VY_NORM
    std[YAW_RATE] = noise.yaw_rate_std_radps / YAW_NORM
    noisy = clean + rng.standard_normal(OBS_DIM) * std
    return np.clip(noisy, OBS_LOW, OBS_HIGH).astype(np.float32)
