"""Reader for the <lap_id>.npz arrays that accompany each lap telemetry file (contract 5.3).

Recorded laps are training data, so they must satisfy the same observation and action spaces as
live env data. Must not import lapzero.sim or lapzero.training (the backend imports lapzero.io).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lapzero.contract import (
    ACT_DIM,
    ACT_HIGH,
    ACT_LOW,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    ContractViolationError,
)

_KEYS = ("obs", "actions", "rewards", "dones", "contract_version")


@dataclass(frozen=True, slots=True)
class LapArrays:
    """Training arrays of one lap; row i is the observation the driver saw and the action taken.

    Attributes:
        obs: (N, OBS_DIM) float32 observations.
        actions: (N, ACT_DIM) float32 normalized actions.
        rewards: (N,) float32 rewards.
        dones: (N,) bool episode-end flags.
        contract_version: Contract version the lap was recorded under.
    """

    obs: NDArray[np.float32]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    contract_version: str

    @property
    def sample_count(self) -> int:
        """Number of recorded control steps (N)."""
        return int(self.obs.shape[0])


def read_lap_arrays(path: Path) -> LapArrays:
    """Read a lap's .npz file and check it against contract Section 5.3 and the spaces.

    Args:
        path: Path to a <lap_id>.npz file.

    Returns:
        The validated arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ContractViolationError: If an array is missing or has the wrong shape, dtype or values.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(_KEYS) - set(data.files))
        if missing:
            raise ContractViolationError(f"{path.name}: missing arrays {missing}")
        obs, actions, rewards, dones, version = (data[key] for key in _KEYS)
    rows = obs.shape[0] if obs.ndim == 2 else -1
    expected = {
        "obs": (obs, (rows, OBS_DIM), np.float32),
        "actions": (actions, (rows, ACT_DIM), np.float32),
        "rewards": (rewards, (rows,), np.float32),
        "dones": (dones, (rows,), np.bool_),
    }
    for name, (array, shape, dtype) in expected.items():
        if rows < 1 or array.shape != shape or array.dtype != dtype:
            raise ContractViolationError(
                f"{path.name}: {name} must be {np.dtype(dtype)} with shape (N >= 1, ...) matching "
                f"obs, got {array.dtype} {array.shape}"
            )
    if not (np.isfinite(obs).all() and (obs >= OBS_LOW).all() and (obs <= OBS_HIGH).all()):
        raise ContractViolationError(
            f"{path.name}: obs must be finite and inside OBS_LOW..OBS_HIGH"
        )
    if not (
        np.isfinite(actions).all() and (actions >= ACT_LOW).all() and (actions <= ACT_HIGH).all()
    ):
        raise ContractViolationError(f"{path.name}: actions must be finite and inside [-1, 1]")
    if not np.isfinite(rewards).all():
        raise ContractViolationError(f"{path.name}: rewards must be finite")
    if version.shape != () or version.dtype.kind != "U":
        raise ContractViolationError(f"{path.name}: contract_version must be a string scalar")
    return LapArrays(obs, actions, rewards, dones, str(version))
