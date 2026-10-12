"""Unit tests for lapzero/io/lap_arrays.py: reading and validating a lap's .npz arrays."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lapzero.contract import ContractViolationError
from lapzero.io.lap_arrays import read_lap_arrays
from tests.conftest import record_lap


def _arrays(rows: int = 4, **overrides: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    arrays = {
        "obs": rng.uniform(0.0, 1.0, (rows, 113)).astype(np.float32),
        "actions": rng.uniform(-1.0, 1.0, (rows, 2)).astype(np.float32),
        "rewards": np.zeros(rows, dtype=np.float32),
        "dones": np.zeros(rows, dtype=np.bool_),
        "contract_version": np.asarray("1.0.0"),
    }
    return arrays | overrides


def _write(tmp_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "lap.npz"
    np.savez(path, **arrays)
    return path


def test_valid_arrays_are_read_back_unchanged(tmp_path: Path) -> None:
    arrays = _arrays()
    loaded = read_lap_arrays(_write(tmp_path, arrays))
    np.testing.assert_array_equal(loaded.obs, arrays["obs"])
    np.testing.assert_array_equal(loaded.actions, arrays["actions"])
    assert loaded.sample_count == 4
    assert loaded.contract_version == "1.0.0"


def test_arrays_written_by_the_recorder_are_valid(tmp_path: Path) -> None:
    files = record_lap(tmp_path, steps=3)
    assert read_lap_arrays(files.npz_path).sample_count == 3


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"obs": np.zeros((4, 112), dtype=np.float32)}, "obs"),
        ({"obs": np.zeros((4, 113), dtype=np.float64)}, "obs"),
        ({"actions": np.zeros((4, 3), dtype=np.float32)}, "actions"),
        ({"rewards": np.zeros(3, dtype=np.float32)}, "rewards"),
        ({"dones": np.zeros(4, dtype=np.int8)}, "dones"),
        ({"obs": np.full((4, 113), np.nan, dtype=np.float32)}, "obs"),
        ({"obs": np.full((4, 113), 2.0, dtype=np.float32)}, "obs"),
        ({"actions": np.full((4, 2), 1.5, dtype=np.float32)}, "actions"),
        ({"rewards": np.full(4, np.inf, dtype=np.float32)}, "rewards"),
        ({"contract_version": np.asarray(1)}, "contract_version"),
        ({"contract_version": np.asarray(["1.0.0"])}, "contract_version"),
    ],
)
def test_arrays_breaking_the_contract_are_rejected(
    tmp_path: Path, overrides: dict[str, np.ndarray], field: str
) -> None:
    with pytest.raises(ContractViolationError, match=field):
        read_lap_arrays(_write(tmp_path, _arrays(**overrides)))


def test_a_lap_without_samples_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractViolationError, match="N >= 1"):
        read_lap_arrays(_write(tmp_path, _arrays(rows=0)))


def test_a_missing_array_is_rejected(tmp_path: Path) -> None:
    arrays = _arrays()
    del arrays["dones"]
    with pytest.raises(ContractViolationError, match="missing arrays"):
        read_lap_arrays(_write(tmp_path, arrays))


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_lap_arrays(tmp_path / "missing.npz")
