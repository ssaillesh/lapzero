"""LiDAR downsampling from raw simulator scans to the contract's normalized beams (Section 1.3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lapzero.contract import BEAM_BIN, LIDAR_MAX, N_BEAMS, RAW_BEAMS, ContractViolationError


def downsample_scan(scan: ArrayLike) -> NDArray[np.float32]:
    """Downsample a raw LiDAR scan to N_BEAMS normalized ranges (contract Section 1.3).

    NaN and +/-inf count as "no return" (LIDAR_MAX); negatives become 0. Each bin keeps its
    minimum so thin obstacles stay visible. Beam order is unchanged: index 0 is the right-most ray.

    Args:
        scan: Raw ranges in metres, shape (RAW_BEAMS,).

    Returns:
        Array of shape (N_BEAMS,), dtype float32, values in [0, 1].

    Raises:
        ContractViolationError: If `scan` does not have shape (RAW_BEAMS,).
    """
    ranges = np.asarray(scan, dtype=np.float64)
    if ranges.shape != (RAW_BEAMS,):
        raise ContractViolationError(f"scan must have shape ({RAW_BEAMS},), got {ranges.shape}")
    ranges = np.clip(np.where(np.isfinite(ranges), ranges, LIDAR_MAX), 0.0, LIDAR_MAX)
    binned: NDArray[np.float64] = ranges.reshape(N_BEAMS, BEAM_BIN).min(axis=1) / LIDAR_MAX
    return binned.astype(np.float32)
