"""Human driver: turns keyboard or gamepad state into the contract action [steer, throttle].

Pure maths, no device access (lapzero/record/pygame_input.py reads the devices). Humans emit
actions in the same [-1, 1] space as models (contract Section 2.1), so recorded laps are
behaviour cloning data. Requests from a device are clipped to [-1, 1], then slew-rate limited per
control step, so a key press ramps the action instead of jumping straight to full lock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from lapzero.contract import ACT_DIM, STEER, STEER_MAX, STEER_RATE_MAX, THROTTLE

DEFAULT_DEADZONE = 0.05  # fraction of stick travel around centre that reads as straight


def _clip_unit(value: float) -> float:
    """Clip `value` to [-1, 1]; NaN becomes 0.0 and +/-inf the matching bound."""
    if math.isnan(value):
        return 0.0
    return min(max(value, -1.0), 1.0)


@dataclass(frozen=True, slots=True)
class KeyboardState:
    """Which driving keys are held."""

    left: bool = False
    right: bool = False
    throttle: bool = False
    brake: bool = False


@dataclass(frozen=True, slots=True)
class GamepadState:
    """Gamepad readings.

    Attributes:
        stick_x: Steering stick, -1 full left to +1 full right (SDL convention).
        throttle: Throttle trigger travel, 0 released to 1 fully pressed.
        brake: Brake trigger travel, 0 released to 1 fully pressed.
    """

    stick_x: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass(frozen=True, slots=True)
class InputFrame:
    """One poll of a human input device.

    Attributes:
        steer: Requested steering, +1 full left to -1 full right; the driver clips it.
        throttle: Requested pedal, +1 full throttle, 0 coast, -1 full brake; the driver clips it.
        quit: The human asked to end the session.
        restart: The human asked to restart the episode (set only on the poll the request began).
    """

    steer: float = 0.0
    throttle: float = 0.0
    quit: bool = False
    restart: bool = False


class InputSource(Protocol):
    """A human input device, polled once per control step."""

    def poll(self) -> InputFrame:
        """Read the device now."""
        ...


def keyboard_target(keys: KeyboardState) -> tuple[float, float]:
    """Map held keys to a requested action.

    Args:
        keys: Driving keys held.

    Returns:
        (steer, throttle). Steer is +1 with only left held, -1 with only right, else 0. Throttle
        is -1 while brake is held (brake wins over throttle), +1 with only throttle, else 0.
    """
    steer = float(keys.left) - float(keys.right)
    throttle = -1.0 if keys.brake else float(keys.throttle)
    return steer, throttle


def gamepad_target(pad: GamepadState, deadzone: float = DEFAULT_DEADZONE) -> tuple[float, float]:
    """Map gamepad readings to a requested action.

    Args:
        pad: Stick and trigger readings. NaN reads as centred or released; out-of-range values
            are clipped to their travel.
        deadzone: Stick travel around centre that reads as straight, in [0, 1).

    Returns:
        (steer, throttle) in [-1, 1]. Steering rises linearly from 0 at the deadzone edge to full
        lock at full travel, stick left giving +1 (left). Throttle is throttle minus brake travel.

    Raises:
        ValueError: If `deadzone` is not in [0, 1).
    """
    if not 0.0 <= deadzone < 1.0:
        raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")
    x = _clip_unit(pad.stick_x)
    magnitude = max(abs(x) - deadzone, 0.0) / (1.0 - deadzone)
    steer = math.copysign(magnitude, -x) if magnitude > 0.0 else 0.0
    travel = max(_clip_unit(pad.throttle), 0.0) - max(_clip_unit(pad.brake), 0.0)
    return steer, travel


def slew(current: float, target: float, rate: float) -> float:
    """Move `current` toward `target` by at most `rate`.

    Args:
        current: Value on the previous control step, in [-1, 1].
        target: Requested value. Clipped to [-1, 1]; NaN means 0.
        rate: Largest change allowed, > 0.

    Returns:
        `target` if it is within `rate` of `current`, else `current` moved `rate` toward it.
    """
    goal = _clip_unit(target)
    delta = goal - current
    if abs(delta) <= rate:
        return goal
    return current + math.copysign(rate, delta)


@dataclass(frozen=True, slots=True)
class DriverConfig:
    """Human driver settings, in normalized action units per control step (CTRL_DT).

    Attributes:
        steer_rate: Largest steer change per control step. The default, STEER_RATE_MAX /
            STEER_MAX, equals the env's steering rate limit (contract Section 2.2), so the
            recorded steer action, the prev_steer observation and the applied angle agree.
        throttle_rate: Largest throttle change per control step. The default takes the pedal
            from released to floored in two control steps (0.08 s).
    """

    steer_rate: float = STEER_RATE_MAX / STEER_MAX
    throttle_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in ("steer_rate", "throttle_rate"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 < value <= 2.0):
                raise ValueError(f"DriverConfig.{name} must be in (0, 2], got {value}")


class HumanDriver:
    """Turns successive input requests into slew-limited contract actions, one per control step."""

    def __init__(self, config: DriverConfig | None = None) -> None:
        """Start centred with the pedals released.

        Args:
            config: Rate limits; defaults to DriverConfig().
        """
        self._config = config or DriverConfig()
        self._action = np.zeros(ACT_DIM, dtype=np.float32)

    def reset(self) -> None:
        """Centre the steering and release the pedals, matching env.reset() (contract 1.4)."""
        self._action = np.zeros(ACT_DIM, dtype=np.float32)

    def act(self, steer: float, throttle: float) -> NDArray[np.float32]:
        """Return this control step's action, moved toward the request.

        Args:
            steer: Requested steering, +1 full left to -1 full right. Clipped; NaN means 0.
            throttle: Requested pedal, +1 full throttle to -1 full brake. Clipped; NaN means 0.

        Returns:
            New float32 array of shape (ACT_DIM,) in [-1, 1]: each element moved from the
            previous action toward its request by at most its rate.
        """
        action = np.zeros(ACT_DIM, dtype=np.float32)
        action[STEER] = slew(float(self._action[STEER]), steer, self._config.steer_rate)
        action[THROTTLE] = slew(float(self._action[THROTTLE]), throttle, self._config.throttle_rate)
        self._action = action
        return action.copy()
