"""Contract Section 8 / test_lidar_downsample: LiDAR downsampling rule (Section 1.3)."""

from collections.abc import Callable

import numpy as np
import pytest

from lapzero.sim.env import LapZeroEnv
from lapzero.sim.lidar import downsample_scan
from tests.conftest import FakeSimulator


def test_close_point_in_bin_37_sets_obs_37_to_distance_over_30(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator
) -> None:
    fake_sim.scan = np.full(1080, 30.0)
    fake_sim.scan[37 * 10 + 4] = 2.4
    obs, _ = make_env().reset(seed=0)
    assert obs[37] == pytest.approx(2.4 / 30.0)
    np.testing.assert_array_equal(np.delete(obs[0:108], 37), np.ones(107, dtype=np.float32))


def test_nan_and_inf_ranges_map_to_one(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator
) -> None:
    fake_sim.scan = np.full(1080, 15.0)
    fake_sim.scan[0:10] = np.nan
    fake_sim.scan[10:20] = np.inf
    obs, _ = make_env().reset(seed=0)
    assert obs[0] == 1.0
    assert obs[1] == 1.0
    assert obs[2] == pytest.approx(0.5)


def test_bin_keeps_its_minimum_not_its_mean() -> None:
    scan = np.full(1080, 30.0)
    scan[0:10] = [30, 30, 30, 30, 3, 30, 30, 30, 30, 30]
    assert downsample_scan(scan)[0] == pytest.approx(0.1)


def test_ranges_beyond_lidar_max_map_to_one_and_negatives_to_zero() -> None:
    scan = np.full(1080, 15.0)
    scan[0:10] = 1e6
    scan[10:20] = -2.0
    result = downsample_scan(scan)
    assert result[0] == 1.0
    assert result[1] == 0.0


def test_beam_zero_is_the_right_most_ray_and_order_is_not_reversed() -> None:
    scan = np.full(1080, 30.0)
    scan[0] = 3.0  # first raw ray: right-most, -FOV/2
    scan[1079] = 6.0  # last raw ray: left-most, +FOV/2
    result = downsample_scan(scan)
    assert result[0] == pytest.approx(0.1)
    assert result[107] == pytest.approx(0.2)
