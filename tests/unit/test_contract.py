"""Unit tests for lapzero/contract.py: the constants agree with each other."""

import math

import numpy as np
import pytest

from lapzero import contract


def test_downsampled_beams_cover_every_raw_beam_exactly() -> None:
    assert contract.N_BEAMS * contract.BEAM_BIN == contract.RAW_BEAMS


def test_state_indices_follow_lidar_contiguously_and_fill_the_observation() -> None:
    state_indices = [
        contract.V_LONG,
        contract.V_LAT,
        contract.YAW_RATE,
        contract.PREV_STEER,
        contract.PREV_THROTTLE,
    ]
    assert contract.LIDAR_SLICE.stop == contract.N_BEAMS
    assert state_indices == list(range(contract.N_BEAMS, contract.OBS_DIM))


@pytest.mark.parametrize(
    "bound", [contract.OBS_LOW, contract.OBS_HIGH, contract.ACT_LOW, contract.ACT_HIGH]
)
def test_bound_arrays_are_float32_and_read_only(bound: np.ndarray) -> None:
    assert bound.dtype == np.float32
    with pytest.raises(ValueError, match="read-only"):
        bound[0] = 0.5


def test_bound_arrays_match_their_space_sizes() -> None:
    assert contract.OBS_LOW.shape == contract.OBS_HIGH.shape == (contract.OBS_DIM,)
    assert contract.ACT_LOW.shape == contract.ACT_HIGH.shape == (contract.ACT_DIM,)


def test_control_period_is_action_repeat_physics_steps() -> None:
    assert math.isclose(contract.CTRL_DT, 0.04)


def test_control_rate_is_the_inverse_of_the_control_period() -> None:
    assert math.isclose(contract.CTRL_HZ * contract.CTRL_DT, 1.0)


def test_training_speed_cap_never_exceeds_simulator_limit() -> None:
    assert 0.0 < contract.V_MAX_TRAIN <= contract.SIM_V_MAX


def test_replay_rate_is_a_downsampling_of_the_control_rate() -> None:
    assert 0 < contract.REPLAY_HZ <= contract.CTRL_HZ
