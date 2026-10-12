"""Unit tests for lapzero/training/bc.py: dataset loading, lap split, training and run logging.

Laps are recorded with LapRecorder on the fake simulator; training runs on CPU with tiny networks.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from lapzero.contract import V_LONG, ContractViolationError
from lapzero.io.lap_arrays import read_lap_arrays
from lapzero.io.telemetry import DriverInfo, DriverType, LapStatus, read_lap_telemetry
from lapzero.training.bc import (
    BCConfig,
    load_bc_dataset,
    mse,
    run_bc_training,
    split_by_lap,
    train_bc,
    train_step,
)
from lapzero.training.policy import BCPolicy, load_policy
from tests.conftest import record_lap

MODEL_DRIVER = DriverInfo(
    type=DriverType.MODEL, name="lapzero-ppo", model_version="1", mlflow_run_id="run1"
)
SMALL = {"batch_size": 4, "hidden_size": 16}


class FakeTracker:
    """Records RunTracker calls; like MLflow, a param key may only be logged once."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[str, float, int]] = []
        self.artifacts: list[Path] = []

    def log_params(self, params: Mapping[str, Any]) -> None:
        assert not set(params) & set(self.params), "param logged twice"
        self.params.update(params)

    def log_metric(self, key: str, value: float, step: int) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, path: Path) -> None:
        self.artifacts.append(path)

    def steps(self, key: str) -> list[int]:
        return [step for name, _, step in self.metrics if name == key]


def expert(obs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Deterministic driver: steer toward the side with more room, ease off as speed rises."""
    steer = np.tanh(4.0 * (obs[54:108].mean() - obs[0:54].mean()))
    throttle = np.tanh(2.0 * (0.25 - obs[V_LONG]))
    return np.array([steer, throttle], dtype=np.float32)


@pytest.fixture
def laps_dir(tmp_path: Path) -> Path:
    """Three completed human laps (16 samples on two tracks) plus two laps BC must skip.

    The human laps follow `expert`, so their actions are a learnable function of the observation.
    """
    root = tmp_path / "laps"
    record_lap(root, track_id="track_a", steps=5, seed=1, policy=expert)
    record_lap(root, track_id="track_a", steps=7, seed=2, policy=expert)
    record_lap(root, track_id="track_b", steps=4, seed=3, policy=expert)
    record_lap(root, track_id="track_a", steps=3, crash=True, seed=4, policy=expert)
    record_lap(root, track_id="track_a", steps=6, driver=MODEL_DRIVER, seed=5)
    return root


def _human_completed(laps_dir: Path) -> list[Path]:
    paths = sorted(laps_dir.glob("*/*.json"))
    return [
        path
        for path in paths
        if (telemetry := read_lap_telemetry(path)).driver.type is DriverType.HUMAN
        and telemetry.result.status is LapStatus.COMPLETED
    ]


# Dataset loading


def test_only_completed_human_laps_are_loaded(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    assert len(dataset.lap_ids) == 3
    assert dataset.obs.shape == (16, 113)
    assert dataset.actions.shape == (16, 2)
    assert dataset.obs.dtype == dataset.actions.dtype == np.float32
    assert dataset.track_ids == ("track_a", "track_b")


def test_rows_are_the_recorded_observations_and_actions_of_each_lap(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    for index, lap_id in enumerate(dataset.lap_ids):
        (npz_path,) = laps_dir.glob(f"*/{lap_id}.npz")
        arrays = read_lap_arrays(npz_path)
        rows = dataset.lap_of_row == index
        np.testing.assert_array_equal(dataset.obs[rows], arrays.obs)
        np.testing.assert_array_equal(dataset.actions[rows], arrays.actions)


def test_track_filter_loads_only_the_named_tracks(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir, track_ids={"track_b"})
    assert dataset.track_ids == ("track_b",)
    assert dataset.obs.shape == (4, 113)


def test_laps_from_another_contract_major_version_are_skipped(laps_dir: Path) -> None:
    path = _human_completed(laps_dir)[0]
    document = json.loads(path.read_text())
    document["contract_version"] = "2.0.0"
    path.write_text(json.dumps(document))
    assert len(load_bc_dataset(laps_dir).lap_ids) == 2


def test_npz_that_disagrees_with_its_json_is_rejected(laps_dir: Path) -> None:
    npz_path = _human_completed(laps_dir)[0].with_suffix(".npz")
    with np.load(npz_path) as data:
        shortened = {
            key: data[key] if key == "contract_version" else data[key][:-1] for key in data
        }
    np.savez(npz_path, **shortened)
    with pytest.raises(ContractViolationError, match="samples"):
        load_bc_dataset(laps_dir)


def test_missing_npz_raises_file_not_found(laps_dir: Path) -> None:
    _human_completed(laps_dir)[0].with_suffix(".npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_bc_dataset(laps_dir)


@pytest.mark.parametrize("track_ids", [{"no_such_track"}, set()])
def test_no_qualifying_laps_raises_value_error(laps_dir: Path, track_ids: set[str]) -> None:
    with pytest.raises(ValueError, match="no completed human laps"):
        load_bc_dataset(laps_dir, track_ids=track_ids)


# Lap split


def test_split_keeps_every_lap_whole_on_one_side(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    train, val = split_by_lap(dataset, 0.2, np.random.default_rng(0))
    assert val is not None
    assert len(val.lap_ids) == 1
    assert sorted(train.lap_ids + val.lap_ids) == sorted(dataset.lap_ids)
    assert len(train.obs) + len(val.obs) == len(dataset.obs)
    for part in (train, val):
        for index, lap_id in enumerate(part.lap_ids):
            original = dataset.lap_ids.index(lap_id)
            np.testing.assert_array_equal(
                part.obs[part.lap_of_row == index], dataset.obs[dataset.lap_of_row == original]
            )


def test_split_is_reproducible_for_a_seed(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    splits = [split_by_lap(dataset, 0.4, np.random.default_rng(3)) for _ in range(2)]
    assert splits[0][0].lap_ids == splits[1][0].lap_ids


def test_zero_val_fraction_trains_on_every_lap(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    train, val = split_by_lap(dataset, 0.0, np.random.default_rng(0))
    assert val is None
    assert train.lap_ids == dataset.lap_ids


def test_split_must_leave_a_training_lap(laps_dir: Path) -> None:
    with pytest.raises(ValueError, match="no training laps"):
        split_by_lap(load_bc_dataset(laps_dir), 0.9, np.random.default_rng(0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"hidden_size": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"val_fraction": 1.0},
        {"val_fraction": -0.1},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=rf"BCConfig\.{next(iter(overrides))}"):
        BCConfig(**overrides)


# Training


def _learnable_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    obs = rng.uniform(0.0, 1.0, (256, 113)).astype(np.float32)
    logits = obs @ rng.normal(0.0, 0.3, (113, 2))
    return obs, np.tanh(logits - logits.mean(axis=0)).astype(np.float32)


def test_train_step_returns_the_batch_mse_before_updating() -> None:
    obs, actions = _learnable_data()
    torch.manual_seed(0)
    policy = BCPolicy(hidden_size=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    before = mse(policy, obs, actions)
    loss = train_step(policy, optimizer, torch.from_numpy(obs), torch.from_numpy(actions))
    assert loss == pytest.approx(before, rel=1e-5)
    assert mse(policy, obs, actions) < before


def test_repeated_train_steps_converge_on_a_learnable_mapping() -> None:
    obs, actions = _learnable_data()
    torch.manual_seed(0)
    policy = BCPolicy(hidden_size=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    before = mse(policy, obs, actions)
    for _ in range(200):
        train_step(policy, optimizer, torch.from_numpy(obs), torch.from_numpy(actions))
    assert mse(policy, obs, actions) < 0.05 * before


def test_training_on_recorded_laps_lowers_the_training_mse(laps_dir: Path) -> None:
    result = train_bc(load_bc_dataset(laps_dir), BCConfig(epochs=30, val_fraction=0.0, **SMALL))
    assert result.train_mse[-1] < 0.5 * result.train_mse[0]
    assert not result.policy.training


def test_train_and_val_mse_are_logged_once_per_epoch(laps_dir: Path) -> None:
    tracker = FakeTracker()
    dataset = load_bc_dataset(laps_dir)
    result = train_bc(dataset, BCConfig(epochs=3, val_fraction=0.2, **SMALL), tracker)
    assert tracker.steps("train_mse") == tracker.steps("val_mse") == [0, 1, 2]
    assert len(result.train_mse) == len(result.val_mse) == 3
    assert not set(result.train_laps) & set(result.val_laps)
    assert sorted(result.train_laps + result.val_laps) == sorted(dataset.lap_ids)


def test_without_validation_laps_only_train_mse_is_logged(laps_dir: Path) -> None:
    tracker = FakeTracker()
    result = train_bc(
        load_bc_dataset(laps_dir), BCConfig(epochs=2, val_fraction=0.0, **SMALL), tracker
    )
    assert tracker.steps("val_mse") == []
    assert result.val_mse == ()
    assert result.val_laps == ()


def test_training_is_reproducible_for_a_seed(laps_dir: Path) -> None:
    dataset = load_bc_dataset(laps_dir)
    first, second = (train_bc(dataset, BCConfig(epochs=3, seed=11, **SMALL)) for _ in range(2))
    assert first.train_mse == second.train_mse
    for name, weights in first.policy.state_dict().items():
        assert torch.equal(weights, second.policy.state_dict()[name]), name


# Full run


def test_run_saves_loadable_weights_and_logs_the_required_params(
    laps_dir: Path, tmp_path: Path
) -> None:
    tracker = FakeTracker()
    output = tmp_path / "models" / "bc_v1.pt"
    result = run_bc_training(laps_dir, output, BCConfig(epochs=2, **SMALL), tracker)
    assert tracker.artifacts == [output]
    obs = torch.rand(3, 113)
    with torch.no_grad():
        torch.testing.assert_close(load_policy(output)(obs), result.policy(obs))
    expected = {
        "contract_version": "1.0.0",
        "contract.OBS_DIM": 113,
        "bc.epochs": 2,
        "seed": 0,
        "track_ids": "track_a,track_b",
        "laps": 3,
        "samples": 16,
        "train_laps": 2,
        "val_laps": 1,
    }
    assert {key: tracker.params[key] for key in expected} == expected
    assert tracker.params["git_commit"]
    assert "reward_version" in tracker.params
    assert tracker.steps("train_mse") == [0, 1]
