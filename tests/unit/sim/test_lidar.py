"""Unit tests for lapzero/sim/lidar.py: downsampling properties over any raw scan."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from lapzero.contract import BEAM_BIN, LIDAR_MAX, N_BEAMS, RAW_BEAMS, ContractViolationError
from lapzero.sim.lidar import downsample_scan

ANY_SCAN = arrays(np.float64, RAW_BEAMS, elements=st.floats(allow_nan=True, allow_infinity=True))
VALID_SCAN = arrays(np.float64, RAW_BEAMS, elements=st.floats(0.0, LIDAR_MAX))


@given(ANY_SCAN)
def test_downsampled_scan_is_finite_float32_in_unit_range_for_any_input(scan: np.ndarray) -> None:
    result = downsample_scan(scan)
    assert result.shape == (N_BEAMS,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert ((result >= 0.0) & (result <= 1.0)).all()


@given(VALID_SCAN)
def test_each_beam_is_the_closest_range_in_its_bin_over_lidar_max(scan: np.ndarray) -> None:
    expected = [min(scan[i * BEAM_BIN : (i + 1) * BEAM_BIN]) / LIDAR_MAX for i in range(N_BEAMS)]
    np.testing.assert_allclose(downsample_scan(scan), expected, rtol=1e-6, atol=1e-7)


def test_negative_infinity_counts_as_no_return() -> None:
    scan = np.full(RAW_BEAMS, 15.0)
    scan[0:BEAM_BIN] = -math.inf
    assert downsample_scan(scan)[0] == 1.0


def test_downsample_scan_rejects_the_wrong_number_of_beams() -> None:
    with pytest.raises(ContractViolationError, match="1080"):
        downsample_scan(np.zeros(540))
