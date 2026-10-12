"""Keyboard and gamepad input through pygame, read once per control step.

F1TENTH Gym draws in a pygame window, which has keyboard focus while driving. Its renderer drains
pygame's event queue on every frame, so this module reads device state (pygame.key.get_pressed,
joystick axes) instead of events, and detects the restart key press by comparing polls.

Keys: arrows steer, Up is throttle and Down is brake, R restarts the episode, Esc quits. WASD is
not used because F1TENTH's renderer toggles drawing on S.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pygame

from lapzero.io.telemetry import InputDevice
from lapzero.record.driver import (
    DEFAULT_DEADZONE,
    GamepadState,
    InputFrame,
    KeyboardState,
    gamepad_target,
    keyboard_target,
)

logger = logging.getLogger(__name__)


class Joystick(Protocol):
    """The part of pygame.joystick.JoystickType this module uses."""

    def get_name(self) -> str:
        """Device name."""
        ...

    def get_numaxes(self) -> int:
        """Number of axes."""
        ...

    def get_axis(self, axis_number: int) -> float:
        """Axis position, -1 to 1."""
        ...


@dataclass(frozen=True, slots=True)
class GamepadLayout:
    """Joystick axis numbers. The defaults fit Xbox and PlayStation pads under SDL2.

    Attributes:
        steer_axis: Left stick horizontal, -1 left to +1 right.
        throttle_axis: Right trigger, -1 released to +1 pressed.
        brake_axis: Left trigger, -1 released to +1 pressed.
    """

    steer_axis: int = 0
    throttle_axis: int = 5
    brake_axis: int = 4


def trigger_travel(raw: float) -> float:
    """Convert an SDL trigger axis (-1 released, +1 pressed) to travel (0 released, 1 pressed)."""
    return (raw + 1.0) / 2.0


class PygameInput:
    """InputSource reading the arrow keys, or a gamepad's stick and triggers when one is given.

    Esc (quit) and R (restart) are always read from the keyboard.
    """

    def __init__(
        self,
        gamepad: Joystick | None = None,
        layout: GamepadLayout | None = None,
        deadzone: float = DEFAULT_DEADZONE,
        *,
        get_pressed: Callable[[], Sequence[bool]] = pygame.key.get_pressed,
        pump: Callable[[], None] = pygame.event.pump,
    ) -> None:
        """Set up polling. pygame's display must be open (the F1TENTH render window).

        Args:
            gamepad: Opened joystick to drive with; None drives with the arrow keys.
            layout: Gamepad axis numbers; defaults to GamepadLayout().
            deadzone: Gamepad stick deadzone, in [0, 1).
            get_pressed: Reads key state; injectable for tests.
            pump: Updates pygame's device state; injectable for tests.

        Raises:
            ValueError: If a layout axis does not exist on the gamepad or `deadzone` is invalid.
        """
        self._layout = layout or GamepadLayout()
        if gamepad is not None:
            axes = gamepad.get_numaxes()
            for name in ("steer_axis", "throttle_axis", "brake_axis"):
                axis = getattr(self._layout, name)
                if not 0 <= axis < axes:
                    raise ValueError(f"GamepadLayout.{name}={axis}, the gamepad has {axes} axes")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")
        self._gamepad = gamepad
        self._deadzone = deadzone
        self._get_pressed = get_pressed
        self._pump = pump
        self._restart_held = False

    def poll(self) -> InputFrame:
        """Read the devices now.

        Returns:
            The requested action and the quit and restart requests.
        """
        self._pump()
        keys = self._get_pressed()
        restart_held = bool(keys[pygame.K_r])
        restart = restart_held and not self._restart_held
        self._restart_held = restart_held
        if self._gamepad is None:
            held = KeyboardState(
                left=bool(keys[pygame.K_LEFT]),
                right=bool(keys[pygame.K_RIGHT]),
                throttle=bool(keys[pygame.K_UP]),
                brake=bool(keys[pygame.K_DOWN]),
            )
            steer, throttle = keyboard_target(held)
        else:
            pad = self._gamepad
            reading = GamepadState(
                stick_x=pad.get_axis(self._layout.steer_axis),
                throttle=trigger_travel(pad.get_axis(self._layout.throttle_axis)),
                brake=trigger_travel(pad.get_axis(self._layout.brake_axis)),
            )
            steer, throttle = gamepad_target(reading, self._deadzone)
        return InputFrame(steer, throttle, quit=bool(keys[pygame.K_ESCAPE]), restart=restart)


def open_input(
    device: InputDevice, *, gamepad_index: int = 0, layout: GamepadLayout | None = None
) -> PygameInput:
    """Open keyboard or gamepad input. Call once the F1TENTH render window is open.

    Args:
        device: Input device the human drives with.
        gamepad_index: Joystick number to use when `device` is a gamepad.
        layout: Gamepad axis numbers; defaults to GamepadLayout().

    Returns:
        The input source.

    Raises:
        RuntimeError: If `device` is a gamepad and none is connected at `gamepad_index`.
        ValueError: If `layout` names an axis the gamepad does not have.
    """
    if device is InputDevice.KEYBOARD:
        return PygameInput()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if not 0 <= gamepad_index < count:
        raise RuntimeError(f"no gamepad at gamepad_index={gamepad_index}; {count} connected")
    gamepad = pygame.joystick.Joystick(gamepad_index)
    logger.info("driving with gamepad %d: %s", gamepad_index, gamepad.get_name())
    return PygameInput(gamepad, layout)
