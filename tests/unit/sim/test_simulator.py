"""Unit tests for lapzero/sim/simulator.py: validation of the values a backend hands the env."""

import math

import numpy as np
import pytest

from lapzero.contract import RAW_BEAMS, ContractViolationError
from lapzero.sim.simulator import Pose
from tests.conftest import make_state


def test_sim_state_rejects_a_scan_of_the_wrong_length() -> None:
    with pytest.raises(ContractViolationError, match="scan"):
        make_state(scan=np.zeros(RAW_BEAMS - 1))


@pytest.mark.parametrize("field", ["v_long", "v_lat", "yaw_rate"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_sim_state_rejects_non_finite_speeds(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=rf"SimState\.{field}"):
        make_state(**{field: bad})


def test_sim_state_accepts_non_finite_scan_ranges() -> None:
    scan = np.full(RAW_BEAMS, np.inf)
    scan[0] = np.nan
    assert make_state(scan=scan).scan.shape == (RAW_BEAMS,)


@pytest.mark.parametrize(
    ("x", "y", "theta", "field"), [(math.nan, 0, 0, "x"), (0, math.inf, 0, "y")]
)
def test_pose_rejects_non_finite_coordinates(x: float, y: float, theta: float, field: str) -> None:
    with pytest.raises(ValueError, match=rf"Pose\.{field}"):
        Pose(x, y, theta)
