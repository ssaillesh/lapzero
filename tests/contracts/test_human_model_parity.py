"""Contract Section 8 / test_human_model_parity: human driver actions pass the model action check.

Section 2.1: the keyboard driver and every model emit actions in the same [-1, 1] space, so
recorded human laps and model actions are interchangeable for behaviour cloning.
"""

import itertools
from pathlib import Path

import numpy as np
from gymnasium import spaces
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from lapzero.io.lap_arrays import read_lap_arrays
from lapzero.record.driver import (
    GamepadState,
    HumanDriver,
    KeyboardState,
    gamepad_target,
    keyboard_target,
)
from lapzero.sim.env import LapZeroEnv
from tests.conftest import START_POSE, FakeSimulator, record_lap

ACTION_SPACE = LapZeroEnv(FakeSimulator(), START_POSE).action_space  # the space models act in
ALL_KEYS = [KeyboardState(*held) for held in itertools.product([False, True], repeat=4)]
ANY_FLOAT = st.floats(allow_nan=True, allow_infinity=True)


def _passes_model_action_check(action: NDArray[np.float32], space: spaces.Box) -> bool:
    return action.dtype == np.float32 and action.shape == (2,) and space.contains(action)


def test_every_keyboard_combination_held_and_switched_stays_in_the_action_space() -> None:
    for first, second in itertools.product(ALL_KEYS, repeat=2):
        driver = HumanDriver()
        for keys in [first] * 4 + [second] * 4:
            action = driver.act(*keyboard_target(keys))
            assert _passes_model_action_check(action, ACTION_SPACE), (first, second, action)


@given(st.lists(st.sampled_from(ALL_KEYS), min_size=1, max_size=50))
def test_any_keyboard_sequence_stays_in_the_action_space(sequence: list[KeyboardState]) -> None:
    driver = HumanDriver()
    for keys in sequence:
        assert _passes_model_action_check(driver.act(*keyboard_target(keys)), ACTION_SPACE)


@given(st.lists(st.tuples(ANY_FLOAT, ANY_FLOAT, ANY_FLOAT), min_size=1, max_size=20))
def test_any_gamepad_readings_stay_in_the_action_space(
    readings: list[tuple[float, float, float]],
) -> None:
    driver = HumanDriver()
    for stick_x, throttle, brake in readings:
        action = driver.act(*gamepad_target(GamepadState(stick_x, throttle, brake)))
        assert _passes_model_action_check(action, ACTION_SPACE)


def test_a_recorded_keyboard_lap_loads_as_behaviour_cloning_actions_in_the_space(
    tmp_path: Path,
) -> None:
    driver = HumanDriver()
    accelerate_left = [KeyboardState(left=True, throttle=True)] * 3
    keys = itertools.cycle([*accelerate_left, KeyboardState(right=True, brake=True)])

    def keyboard_policy(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return driver.act(*keyboard_target(next(keys)))

    files = record_lap(tmp_path, steps=12, policy=keyboard_policy)
    arrays = read_lap_arrays(files.npz_path)
    assert arrays.actions.dtype == np.float32
    assert all(ACTION_SPACE.contains(action) for action in arrays.actions)
