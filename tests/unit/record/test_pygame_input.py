"""Unit tests for lapzero/record/pygame_input.py: key and gamepad polling into input frames.

Key state and joysticks are faked; the open_input tests use SDL's headless dummy video driver.
"""

from collections.abc import Iterator
from typing import Any

import pygame
import pytest

from lapzero.io.telemetry import InputDevice
from lapzero.record.driver import InputFrame
from lapzero.record.pygame_input import GamepadLayout, PygameInput, open_input, trigger_travel


class FakeKeys:
    """Key state as returned by pygame.key.get_pressed, indexed by pygame key constants."""

    def __init__(self, *held: int) -> None:
        self.held = set(held)

    def __getitem__(self, key: int) -> bool:
        return key in self.held


class FakeJoystick:
    """A joystick whose axis positions tests set directly (SDL rest positions by default)."""

    def __init__(self, axes: int = 6) -> None:
        self.axes = [0.0] * axes
        self.axes[4] = self.axes[5] = -1.0  # triggers rest at -1

    def get_name(self) -> str:
        return "fake pad"

    def get_numaxes(self) -> int:
        return len(self.axes)

    def get_axis(self, axis_number: int) -> float:
        return self.axes[axis_number]


class KeyScript:
    """get_pressed stand-in returning the key states one poll at a time; counts pumps."""

    def __init__(self, *polls: FakeKeys) -> None:
        self.polls = iter(polls)
        self.pumps = 0

    def get_pressed(self) -> FakeKeys:
        return next(self.polls)

    def pump(self) -> None:
        self.pumps += 1


def _keyboard(*polls: FakeKeys) -> tuple[PygameInput, KeyScript]:
    script = KeyScript(*polls)
    return PygameInput(get_pressed=script.get_pressed, pump=script.pump), script


def _gamepad(pad: FakeJoystick, *polls: FakeKeys, **kwargs: Any) -> PygameInput:
    script = KeyScript(*(polls or (FakeKeys(),)))
    return PygameInput(pad, get_pressed=script.get_pressed, pump=script.pump, **kwargs)


# Keyboard


@pytest.mark.parametrize(
    ("held", "expected"),
    [
        ((), InputFrame(0.0, 0.0)),
        ((pygame.K_LEFT,), InputFrame(1.0, 0.0)),
        ((pygame.K_RIGHT,), InputFrame(-1.0, 0.0)),
        ((pygame.K_UP,), InputFrame(0.0, 1.0)),
        ((pygame.K_DOWN, pygame.K_UP), InputFrame(0.0, -1.0)),
        ((pygame.K_LEFT, pygame.K_UP), InputFrame(1.0, 1.0)),
        ((pygame.K_ESCAPE,), InputFrame(0.0, 0.0, quit=True)),
    ],
)
def test_arrow_keys_and_escape_map_to_the_input_frame(
    held: tuple[int, ...], expected: InputFrame
) -> None:
    source, _ = _keyboard(FakeKeys(*held))
    assert source.poll() == expected


def test_wasd_does_not_drive_because_f1tenth_uses_s_to_toggle_drawing() -> None:
    source, _ = _keyboard(FakeKeys(pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
    assert source.poll() == InputFrame(0.0, 0.0)


def test_restart_fires_once_per_press_of_r() -> None:
    r = FakeKeys(pygame.K_r)
    source, _ = _keyboard(r, r, FakeKeys(), r)
    assert [source.poll().restart for _ in range(4)] == [True, False, False, True]


def test_every_poll_pumps_pygame_before_reading_keys() -> None:
    source, script = _keyboard(FakeKeys(), FakeKeys())
    source.poll()
    source.poll()
    assert script.pumps == 2


# Gamepad


def test_trigger_travel_maps_sdl_rest_to_zero_and_full_press_to_one() -> None:
    assert [trigger_travel(raw) for raw in (-1.0, 0.0, 1.0)] == [0.0, 0.5, 1.0]


def test_gamepad_at_rest_requests_straight_and_coast() -> None:
    assert _gamepad(FakeJoystick()).poll() == InputFrame(0.0, 0.0)


def test_gamepad_stick_left_and_right_trigger_steer_left_on_full_throttle() -> None:
    pad = FakeJoystick()
    pad.axes[0], pad.axes[5] = -1.0, 1.0
    assert _gamepad(pad).poll() == InputFrame(1.0, 1.0)


def test_gamepad_left_trigger_brakes() -> None:
    pad = FakeJoystick()
    pad.axes[4] = 1.0
    assert _gamepad(pad).poll().throttle == -1.0


def test_gamepad_ignores_arrow_keys_but_escape_and_r_still_work() -> None:
    keys = FakeKeys(pygame.K_LEFT, pygame.K_UP, pygame.K_ESCAPE, pygame.K_r)
    assert _gamepad(FakeJoystick(), keys).poll() == InputFrame(0.0, 0.0, quit=True, restart=True)


def test_gamepad_layout_picks_the_axes_to_read() -> None:
    pad = FakeJoystick(axes=8)
    pad.axes[2], pad.axes[6], pad.axes[7] = 1.0, 1.0, -1.0
    layout = GamepadLayout(steer_axis=2, throttle_axis=6, brake_axis=7)
    assert _gamepad(pad, layout=layout).poll() == InputFrame(-1.0, 1.0)


def test_gamepad_deadzone_is_applied_to_the_stick() -> None:
    pad = FakeJoystick()
    pad.axes[0] = 0.15
    assert _gamepad(pad, deadzone=0.2).poll().steer == 0.0


@pytest.mark.parametrize("field", ["steer_axis", "throttle_axis", "brake_axis"])
def test_layout_axis_missing_on_the_gamepad_is_rejected(field: str) -> None:
    layout = GamepadLayout(**{field: 6})
    with pytest.raises(ValueError, match=field):
        PygameInput(FakeJoystick(axes=6), layout)


@pytest.mark.parametrize("deadzone", [-0.1, 1.0])
def test_invalid_deadzone_is_rejected_when_the_input_is_built(deadzone: float) -> None:
    with pytest.raises(ValueError, match="deadzone"):
        PygameInput(deadzone=deadzone)


# open_input


@pytest.fixture
def headless_display(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A 1x1 pygame window on SDL's dummy video driver, standing in for the F1TENTH window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.display.quit()


@pytest.mark.usefixtures("headless_display")
def test_keyboard_input_polls_the_real_pygame_key_state() -> None:
    source = open_input(InputDevice.KEYBOARD)
    assert source.poll() == InputFrame(0.0, 0.0)


@pytest.mark.usefixtures("headless_display")
def test_gamepad_input_opens_the_joystick_at_the_given_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pad = FakeJoystick()
    pad.axes[5] = 1.0
    opened: list[int] = []
    monkeypatch.setattr(pygame.joystick, "init", lambda: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 2)
    monkeypatch.setattr(pygame.joystick, "Joystick", lambda index: opened.append(index) or pad)
    source = open_input(InputDevice.GAMEPAD, gamepad_index=1)
    assert opened == [1]
    assert source.poll() == InputFrame(0.0, 1.0)


@pytest.mark.parametrize("index", [0, -1])
def test_gamepad_input_without_a_connected_gamepad_raises(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    monkeypatch.setattr(pygame.joystick, "init", lambda: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no gamepad"):
        open_input(InputDevice.GAMEPAD, gamepad_index=index)
