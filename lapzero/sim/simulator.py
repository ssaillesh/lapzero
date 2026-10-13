"""Interface between LapZeroEnv and a physics backend: the Simulator protocol and its types.

Backends (F1TENTH Gym in lapzero/sim/f1tenth.py, fakes in tests) implement Simulator and speak
raw SI units. Only LapZeroEnv drives a Simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from lapzero.contract import RAW_BEAMS, ContractViolationError


def _require_finite(owner: str, **values: float) -> None:
    """Raise ValueError naming the first non-finite value."""
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{owner}.{name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class Pose:
    """Car pose in the world (map) frame: x, y in metres, theta in rad (CCW positive)."""

    x: float
    y: float
    theta: float

    def __post_init__(self) -> None:
        _require_finite("Pose", x=self.x, y=self.y, theta=self.theta)


@dataclass(frozen=True, slots=True)
class SimState:
    """Raw ego-car reading after one physics step, in SI units and never normalized.

    Attributes:
        scan: LiDAR ranges in metres, shape (RAW_BEAMS,), right-most ray first. May hold NaN or
            inf; they are sanitized when the observation is built.
        pose: World-frame pose. Goes to info and reward only, never into the observation.
        v_long: Car-frame longitudinal speed, m/s.
        v_lat: Car-frame lateral speed, m/s.
        yaw_rate: Yaw rate, rad/s, CCW positive.
        collision: True if the car hit something on this step.
        lap_count: Start-line passes the simulator has counted since reset. LapZeroEnv decides
            which of them are laps (contract Section 3).
    """

    scan: NDArray[np.float64]
    pose: Pose
    v_long: float
    v_lat: float
    yaw_rate: float
    collision: bool
    lap_count: int

    def __post_init__(self) -> None:
        if self.scan.shape != (RAW_BEAMS,):
            raise ContractViolationError(
                f"SimState.scan must have shape ({RAW_BEAMS},), got {self.scan.shape}"
            )
        _require_finite("SimState", v_long=self.v_long, v_lat=self.v_lat, yaw_rate=self.yaw_rate)


class Simulator(Protocol):
    """Single-car physics backend driven by LapZeroEnv; each step advances SIM_DT seconds."""

    def reset(self, pose: Pose) -> SimState:
        """Place the car at rest at `pose` and return the first reading."""
        ...

    def step(self, steer_rad: float, v_target: float) -> SimState:
        """Advance one physics step with a steering angle (rad) and target velocity (m/s)."""
        ...

    @property
    def friction(self) -> float:
        """Tyre-road friction coefficient mu the car runs with (dimensionless)."""
        ...

    def set_friction(self, mu: float) -> None:
        """Set the tyre-road friction coefficient mu (dimensionless) for the following steps.

        Raises:
            ValueError: If `mu` is not finite and > 0.
        """
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...
