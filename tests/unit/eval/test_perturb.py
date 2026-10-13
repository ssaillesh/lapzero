"""Unit tests for lapzero/eval/perturb.py: observation noise in SI units, inside contract bounds."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from lapzero.contract import (
    LIDAR_MAX,
    LIDAR_SLICE,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    PREV_STEER,
    PREV_THROTTLE,
    V_LAT,
    V_LONG,
    V_NORM,
    VY_NORM,
    YAW_NORM,
    YAW_RATE,
    ContractViolationError,
)
from lapzero.eval.perturb import ObservationNoise, jitter_observation


def _mid_observation() -> np.ndarray:
    """Walls 15 m away, 5 m/s, no slip or yaw, previous action [0.3, -0.2]: far from bounds."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[LIDAR_SLICE] = 0.5
    obs[V_LONG] = 0.25
    obs[PREV_STEER], obs[PREV_THROTTLE] = 0.3, -0.2
    return obs


def test_zero_noise_returns_the_observation_unchanged() -> None:
    obs = _mid_observation()
    noisy = jitter_observation(obs, ObservationNoise(), np.random.default_rng(0))
    assert noisy.dtype == np.float32
    np.testing.assert_array_equal(noisy, obs)


def test_noise_has_the_requested_spread_in_si_units() -> None:
    noise = ObservationNoise(lidar_std_m=0.3, speed_std_mps=0.5, yaw_rate_std_radps=0.2)
    rng = np.random.default_rng(0)
    obs = _mid_observation()
    diffs = np.array([jitter_observation(obs, noise, rng) - obs for _ in range(500)])
    assert diffs[:, LIDAR_SLICE].std() * LIDAR_MAX == pytest.approx(0.3, rel=0.05)
    assert diffs[:, V_LONG].std() * V_NORM == pytest.approx(0.5, rel=0.1)
    assert diffs[:, V_LAT].std() * VY_NORM == pytest.approx(0.5, rel=0.1)
    assert diffs[:, YAW_RATE].std() * YAW_NORM == pytest.approx(0.2, rel=0.1)
    assert abs(diffs[:, LIDAR_SLICE].mean()) * LIDAR_MAX < 0.01  # zero-mean


def test_the_previous_action_entries_are_never_perturbed() -> None:
    noise = ObservationNoise(lidar_std_m=1.0, speed_std_mps=1.0, yaw_rate_std_radps=1.0)
    obs = _mid_observation()
    noisy = jitter_observation(obs, noise, np.random.default_rng(3))
    assert (noisy[PREV_STEER], noisy[PREV_THROTTLE]) == (obs[PREV_STEER], obs[PREV_THROTTLE])


def test_the_same_generator_seed_gives_the_same_noise() -> None:
    noise = ObservationNoise(lidar_std_m=0.1)
    first = jitter_observation(_mid_observation(), noise, np.random.default_rng(42))
    second = jitter_observation(_mid_observation(), noise, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


@given(
    arrays(np.float64, OBS_DIM, elements=st.floats(allow_nan=True, allow_infinity=True)),
    st.floats(0.0, 1e3),
    st.floats(0.0, 1e3),
    st.floats(0.0, 1e3),
    st.integers(0, 2**32),
)
def test_noisy_observations_are_always_finite_and_inside_the_contract_bounds(
    obs: np.ndarray, lidar: float, speed: float, yaw: float, seed: int
) -> None:
    noise = ObservationNoise(lidar_std_m=lidar, speed_std_mps=speed, yaw_rate_std_radps=yaw)
    noisy = jitter_observation(obs, noise, np.random.default_rng(seed))
    assert noisy.shape == (OBS_DIM,)
    assert noisy.dtype == np.float32
    assert np.isfinite(noisy).all()
    assert (noisy >= OBS_LOW).all()
    assert (noisy <= OBS_HIGH).all()


def test_nan_reads_as_no_lidar_return_or_zero_like_the_observation_builder() -> None:
    obs = np.full(OBS_DIM, np.nan)
    noisy = jitter_observation(obs, ObservationNoise(), np.random.default_rng(0))
    assert (noisy[LIDAR_SLICE] == 1.0).all()
    assert (noisy[V_LONG:] == 0.0).all()


def test_an_observation_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="shape"):
        jitter_observation(np.zeros(OBS_DIM - 1), ObservationNoise(), np.random.default_rng(0))


@pytest.mark.parametrize("name", ["lidar_std_m", "speed_std_mps", "yaw_rate_std_radps"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_noise_must_be_finite_and_not_negative(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        ObservationNoise(**{name: value})


def test_noise_is_zero_only_when_every_std_is_zero() -> None:
    assert ObservationNoise().is_zero
    assert not ObservationNoise(yaw_rate_std_radps=0.01).is_zero
