"""Unit tests for lapzero/sim/observation.py: the observation always satisfies the contract."""

import math

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from lapzero.contract import OBS_DIM, OBS_HIGH, OBS_LOW, RAW_BEAMS
from lapzero.sim.observation import build_observation, normalize
from lapzero.sim.simulator import Pose
from tests.conftest import make_state

FINITE = st.floats(allow_nan=False, allow_infinity=False)
ANY_SCAN = arrays(np.float64, RAW_BEAMS, elements=st.floats(allow_nan=True, allow_infinity=True))
PREV_ACTION = arrays(np.float32, 2, elements=st.floats(-1.0, 1.0, width=32))


@given(st.floats(allow_nan=True, allow_infinity=True), st.floats(0.1, 100.0))
@example(math.nan, 5.0)
@example(math.inf, 5.0)
def test_normalize_is_finite_and_within_unit_range(value: float, scale: float) -> None:
    result = normalize(value, scale)
    assert math.isfinite(result)
    assert -1.0 <= result <= 1.0


@given(scan=ANY_SCAN, v_long=FINITE, v_lat=FINITE, yaw_rate=FINITE, prev_action=PREV_ACTION)
def test_observation_is_finite_and_inside_contract_bounds_for_any_state(
    scan: np.ndarray, v_long: float, v_lat: float, yaw_rate: float, prev_action: np.ndarray
) -> None:
    state = make_state(scan=scan, v_long=v_long, v_lat=v_lat, yaw_rate=yaw_rate)
    obs = build_observation(state, prev_action)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert ((obs >= OBS_LOW) & (obs <= OBS_HIGH)).all()


@given(x=FINITE, y=FINITE, theta=FINITE)
def test_world_pose_never_changes_the_observation(x: float, y: float, theta: float) -> None:
    prev_action = np.array([0.3, -0.6], dtype=np.float32)
    at_origin = build_observation(make_state(), prev_action)
    elsewhere = build_observation(make_state(pose=Pose(x, y, theta)), prev_action)
    np.testing.assert_array_equal(at_origin, elsewhere)


def test_huge_speeds_clip_and_previous_action_is_copied() -> None:
    state = make_state(v_long=100.0, v_lat=-100.0, yaw_rate=5.0)
    obs = build_observation(state, np.array([0.3, -0.6], dtype=np.float32))
    np.testing.assert_allclose(obs[108:113], [1.0, -1.0, 0.5, 0.3, -0.6], rtol=1e-6)
