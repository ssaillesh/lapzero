"""Unit tests for lapzero/record/driver.py: input mapping, clipping and per-step rate limiting."""

import math

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from lapzero.contract import STEER_MAX, STEER_RATE_MAX
from lapzero.record.driver import (
    DriverConfig,
    GamepadState,
    HumanDriver,
    KeyboardState,
    gamepad_target,
    keyboard_target,
    slew,
)

ANY_FLOAT = st.floats(allow_nan=True, allow_infinity=True)
UNIT = st.floats(-1.0, 1.0)
STEER_STEP = STEER_RATE_MAX / STEER_MAX  # default steer rate, about 0.358 per control step


# Keyboard mapping


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (KeyboardState(), (0.0, 0.0)),
        (KeyboardState(left=True), (1.0, 0.0)),
        (KeyboardState(right=True), (-1.0, 0.0)),
        (KeyboardState(left=True, right=True), (0.0, 0.0)),
        (KeyboardState(throttle=True), (0.0, 1.0)),
        (KeyboardState(brake=True), (0.0, -1.0)),
        (KeyboardState(left=True, throttle=True), (1.0, 1.0)),
    ],
)
def test_keyboard_maps_left_to_plus_one_and_brake_to_minus_one(
    keys: KeyboardState, expected: tuple[float, float]
) -> None:
    assert keyboard_target(keys) == expected


def test_keyboard_brake_wins_when_throttle_and_brake_are_both_held() -> None:
    assert keyboard_target(KeyboardState(throttle=True, brake=True)) == (0.0, -1.0)


# Gamepad mapping


@pytest.mark.parametrize(("stick_x", "steer"), [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)])
def test_gamepad_stick_left_steers_left(stick_x: float, steer: float) -> None:
    assert gamepad_target(GamepadState(stick_x=stick_x))[0] == steer


@pytest.mark.parametrize("stick_x", [0.04, -0.04, 0.05])
def test_gamepad_stick_inside_the_deadzone_reads_as_positive_zero(stick_x: float) -> None:
    steer, _ = gamepad_target(GamepadState(stick_x=stick_x), deadzone=0.05)
    assert steer == 0.0
    assert math.copysign(1.0, steer) == 1.0  # never -0.0 in the telemetry


def test_gamepad_steering_rises_linearly_from_the_deadzone_edge() -> None:
    steer, _ = gamepad_target(GamepadState(stick_x=-0.6), deadzone=0.2)
    assert steer == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("throttle", "brake", "expected"),
    [(1.0, 0.0, 1.0), (0.0, 1.0, -1.0), (1.0, 1.0, 0.0), (0.75, 0.25, 0.5), (0.0, 0.0, 0.0)],
)
def test_gamepad_throttle_is_throttle_travel_minus_brake_travel(
    throttle: float, brake: float, expected: float
) -> None:
    _, travel = gamepad_target(GamepadState(throttle=throttle, brake=brake))
    assert travel == pytest.approx(expected)


def test_gamepad_nan_readings_mean_centred_and_released() -> None:
    assert gamepad_target(GamepadState(np.nan, np.nan, np.nan)) == (0.0, 0.0)


@pytest.mark.parametrize("deadzone", [-0.1, 1.0, np.nan])
def test_gamepad_rejects_a_deadzone_outside_zero_to_one(deadzone: float) -> None:
    with pytest.raises(ValueError, match="deadzone"):
        gamepad_target(GamepadState(), deadzone=deadzone)


@given(ANY_FLOAT, ANY_FLOAT, ANY_FLOAT, st.floats(0.0, 0.99))
@example(np.inf, -np.inf, np.nan, 0.0)
@example(-1e300, 1e300, -1e300, 0.5)
def test_gamepad_target_is_finite_and_in_unit_range_for_any_reading(
    stick_x: float, throttle: float, brake: float, deadzone: float
) -> None:
    steer, travel = gamepad_target(GamepadState(stick_x, throttle, brake), deadzone)
    assert math.isfinite(steer)
    assert math.isfinite(travel)
    assert -1.0 <= steer <= 1.0
    assert -1.0 <= travel <= 1.0


# Slew limiting


def test_slew_moves_at_most_rate_toward_the_target() -> None:
    assert slew(0.0, 1.0, 0.25) == 0.25
    assert slew(0.5, -1.0, 0.25) == 0.25


def test_slew_lands_exactly_on_a_target_within_reach() -> None:
    assert slew(0.9, 1.0, 0.25) == 1.0
    assert slew(-0.1, 0.0, 0.25) == 0.0


def test_slew_clips_the_target_and_treats_nan_as_zero() -> None:
    assert slew(0.9, 5.0, 0.5) == 1.0
    assert slew(0.2, np.nan, 0.5) == 0.0


@given(UNIT, ANY_FLOAT, st.floats(1e-6, 2.0))
@example(1.0, -np.inf, 2.0)
@example(-1.0, np.nan, 1e-6)
def test_slew_output_is_finite_in_range_and_within_rate_of_the_last_value(
    current: float, target: float, rate: float
) -> None:
    value = slew(current, target, rate)
    assert math.isfinite(value)
    assert -1.0 <= value <= 1.0
    assert abs(value - current) <= rate + 1e-12


# HumanDriver


def test_holding_left_ramps_to_full_lock_in_three_control_steps() -> None:
    driver = HumanDriver()
    steers = [float(driver.act(1.0, 0.0)[0]) for _ in range(4)]
    assert steers == pytest.approx([STEER_STEP, 2 * STEER_STEP, 1.0, 1.0], abs=1e-6)


def test_releasing_the_steering_returns_to_centre_at_the_same_rate() -> None:
    driver = HumanDriver()
    for _ in range(3):
        driver.act(-1.0, 0.0)
    steers = [float(driver.act(0.0, 0.0)[0]) for _ in range(3)]
    assert steers == pytest.approx([-1.0 + STEER_STEP, -1.0 + 2 * STEER_STEP, 0.0], abs=1e-6)


def test_default_steer_rate_equals_the_env_steering_rate_limit() -> None:
    assert DriverConfig().steer_rate * STEER_MAX == pytest.approx(STEER_RATE_MAX)


def test_throttle_floors_in_two_steps_and_full_brake_takes_four_from_full_throttle() -> None:
    driver = HumanDriver()
    throttles = [float(driver.act(0.0, 1.0)[1]) for _ in range(2)]
    throttles += [float(driver.act(0.0, -1.0)[1]) for _ in range(4)]
    assert throttles == [0.5, 1.0, 0.5, 0.0, -0.5, -1.0]


def test_act_returns_a_float32_copy_of_shape_two() -> None:
    driver = HumanDriver()
    action = driver.act(1.0, 1.0)
    assert action.dtype == np.float32
    assert action.shape == (2,)
    action[:] = 99.0
    assert driver.act(1.0, 1.0)[1] == 1.0  # the driver's state was not modified


def test_reset_centres_the_steering_and_releases_the_pedals() -> None:
    driver = HumanDriver(DriverConfig(steer_rate=2.0, throttle_rate=2.0))
    driver.act(1.0, 1.0)
    driver.reset()
    np.testing.assert_array_equal(driver.act(0.0, 0.0), [0.0, 0.0])


def test_custom_rates_set_the_ramp() -> None:
    driver = HumanDriver(DriverConfig(steer_rate=0.1, throttle_rate=1.0))
    np.testing.assert_allclose(driver.act(1.0, 1.0), [0.1, 1.0])


@pytest.mark.parametrize("name", ["steer_rate", "throttle_rate"])
@pytest.mark.parametrize("value", [0.0, -0.5, 2.5, np.nan, np.inf])
def test_config_rejects_rates_outside_zero_to_two(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        DriverConfig(**{name: value})


@given(st.lists(st.tuples(ANY_FLOAT, ANY_FLOAT), min_size=1, max_size=30))
def test_any_request_sequence_gives_bounded_rate_limited_actions(
    requests: list[tuple[float, float]],
) -> None:
    driver, config = HumanDriver(), DriverConfig()
    previous = np.zeros(2, dtype=np.float32)
    for steer, throttle in requests:
        action = driver.act(steer, throttle)
        assert np.isfinite(action).all()
        assert ((action >= -1.0) & (action <= 1.0)).all()
        assert abs(action[0] - previous[0]) <= config.steer_rate + 1e-6
        assert abs(action[1] - previous[1]) <= config.throttle_rate + 1e-6
        previous = action
