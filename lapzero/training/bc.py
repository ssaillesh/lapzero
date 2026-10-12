"""Behaviour cloning: fit an MLP policy to recorded human laps by minimising action MSE.

Data: completed human laps in <laps_dir>/<track_id>/<lap_id>.json + .npz (contract Section 5).
Model: lapzero.training.policy.BCPolicy, a three-layer MLP with actions in [-1, 1].
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from lapzero.contract import CONTRACT_VERSION, ContractViolationError
from lapzero.io.lap_arrays import read_lap_arrays
from lapzero.io.telemetry import DriverType, LapStatus, read_lap_telemetry
from lapzero.training.policy import BCPolicy, major_version, save_policy
from lapzero.training.tracking import RunTracker, contract_params, git_commit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BCConfig:
    """Behaviour cloning hyperparameters.

    Attributes:
        epochs: Passes over the training laps.
        batch_size: Samples per gradient step.
        learning_rate: Adam learning rate.
        hidden_size: Units in each of the two hidden layers.
        val_fraction: Fraction of laps (not samples) held out for validation; 0 disables it.
        seed: Seeds the lap split, batch order and weight initialisation.
    """

    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_size: int = 256
    val_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "hidden_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"BCConfig.{name} must be >= 1, got {getattr(self, name)}")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError(f"BCConfig.learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"BCConfig.val_fraction must be in [0, 1), got {self.val_fraction}")


@dataclass(frozen=True, slots=True)
class BCDataset:
    """(observation, action) rows from whole laps.

    Attributes:
        obs: (N, OBS_DIM) float32 observations.
        actions: (N, ACT_DIM) float32 recorded actions.
        lap_of_row: (N,) index into lap_ids of the lap each row came from.
        lap_ids: IDs of the laps, in load order.
        lap_tracks: Track ID of each lap, aligned with lap_ids.
    """

    obs: NDArray[np.float32]
    actions: NDArray[np.float32]
    lap_of_row: NDArray[np.int64]
    lap_ids: tuple[str, ...]
    lap_tracks: tuple[str, ...]

    @property
    def track_ids(self) -> tuple[str, ...]:
        """Sorted IDs of the tracks in the dataset."""
        return tuple(sorted(set(self.lap_tracks)))

    def subset(self, laps: Sequence[int]) -> BCDataset:
        """Return only the rows of the given laps (indices into lap_ids), renumbered from 0."""
        renumber = np.full(len(self.lap_ids), -1, dtype=np.int64)
        renumber[list(laps)] = np.arange(len(laps))
        keep = renumber[self.lap_of_row] >= 0
        return BCDataset(
            obs=self.obs[keep],
            actions=self.actions[keep],
            lap_of_row=renumber[self.lap_of_row[keep]],
            lap_ids=tuple(self.lap_ids[i] for i in laps),
            lap_tracks=tuple(self.lap_tracks[i] for i in laps),
        )


def load_bc_dataset(laps_dir: Path, track_ids: Collection[str] | None = None) -> BCDataset:
    """Load the completed human laps recorded under the current contract major version.

    Model laps, failed laps (crashed, timeout, stuck) and laps from another contract major
    version are skipped.

    Args:
        laps_dir: Directory holding <track_id>/<lap_id>.json and .npz files.
        track_ids: Only load these tracks; None loads every track.

    Returns:
        The laps' rows, in sorted file order.

    Raises:
        ValueError: If no lap qualifies.
        FileNotFoundError: If a qualifying lap has no .npz file.
        ContractViolationError: If a lap's .npz breaks the contract or disagrees with its .json.
        pydantic.ValidationError: If a .json file breaks the telemetry schema.
    """
    obs: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    lap_of_row: list[NDArray[np.int64]] = []
    lap_ids: list[str] = []
    lap_tracks: list[str] = []
    for json_path in sorted(laps_dir.glob("*/*.json")):
        telemetry = read_lap_telemetry(json_path)
        if track_ids is not None and telemetry.track.track_id not in track_ids:
            continue
        if major_version(telemetry.contract_version) != major_version(CONTRACT_VERSION):
            logger.warning(
                "skipping lap %s: contract %s, need %s",
                telemetry.lap_id,
                telemetry.contract_version,
                CONTRACT_VERSION,
            )
            continue
        if telemetry.driver.type is not DriverType.HUMAN or (
            telemetry.result.status is not LapStatus.COMPLETED
        ):
            continue
        arrays = read_lap_arrays(json_path.with_suffix(".npz"))
        if (arrays.sample_count, arrays.contract_version) != (
            telemetry.result.sample_count,
            telemetry.contract_version,
        ):
            raise ContractViolationError(
                f"lap {telemetry.lap_id}: .npz has {arrays.sample_count} samples under contract "
                f"{arrays.contract_version}, .json says {telemetry.result.sample_count} under "
                f"{telemetry.contract_version}"
            )
        lap_of_row.append(np.full(arrays.sample_count, len(lap_ids), dtype=np.int64))
        obs.append(arrays.obs)
        actions.append(arrays.actions)
        lap_ids.append(telemetry.lap_id)
        lap_tracks.append(telemetry.track.track_id)
    if not lap_ids:
        raise ValueError(f"no completed human laps under {laps_dir} for tracks {track_ids}")
    dataset = BCDataset(
        np.concatenate(obs),
        np.concatenate(actions),
        np.concatenate(lap_of_row),
        tuple(lap_ids),
        tuple(lap_tracks),
    )
    logger.info("loaded %d laps (%d samples) from %s", len(lap_ids), len(dataset.obs), laps_dir)
    return dataset


def split_by_lap(
    dataset: BCDataset, val_fraction: float, rng: np.random.Generator
) -> tuple[BCDataset, BCDataset | None]:
    """Split whole laps into training and validation sets, so validation laps stay unseen.

    Args:
        dataset: Laps to split.
        val_fraction: Fraction of laps to hold out, rounded up; 0 for no validation set.
        rng: Seeded generator choosing which laps are held out.

    Returns:
        (train, validation); validation is None when val_fraction is 0.

    Raises:
        ValueError: If the held-out laps would leave no training laps.
    """
    n_laps = len(dataset.lap_ids)
    n_val = math.ceil(val_fraction * n_laps)
    if n_val >= n_laps:
        raise ValueError(f"val_fraction {val_fraction} leaves no training laps out of {n_laps}")
    order = rng.permutation(n_laps)
    train = dataset.subset(sorted(order[n_val:].tolist()))
    val = dataset.subset(sorted(order[:n_val].tolist())) if n_val else None
    return train, val


def train_step(
    policy: BCPolicy, optimizer: torch.optim.Optimizer, obs: torch.Tensor, actions: torch.Tensor
) -> float:
    """Take one gradient step on the MSE between predicted and recorded actions.

    Returns:
        The batch MSE before the step.
    """
    policy.train()
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(policy(obs), actions)
    loss.backward()  # type: ignore[no-untyped-call]  # torch leaves Tensor.backward unannotated
    optimizer.step()
    return float(loss.item())


def mse(policy: BCPolicy, obs: NDArray[np.float32], actions: NDArray[np.float32]) -> float:
    """Return the policy's action MSE over a whole dataset, without gradients."""
    policy.eval()
    with torch.no_grad():
        predicted = policy(torch.from_numpy(obs))
        return float(nn.functional.mse_loss(predicted, torch.from_numpy(actions)).item())


@dataclass(frozen=True, slots=True)
class BCResult:
    """Trained policy and its per-epoch MSE on the training and validation laps."""

    policy: BCPolicy
    train_mse: tuple[float, ...]
    val_mse: tuple[float, ...]  # empty without validation laps
    train_laps: tuple[str, ...]
    val_laps: tuple[str, ...]


def train_bc(dataset: BCDataset, config: BCConfig, tracker: RunTracker | None = None) -> BCResult:
    """Train a BCPolicy on a whole-lap split of `dataset`; deterministic on CPU for a seed.

    Args:
        dataset: Recorded laps.
        config: Hyperparameters, including the seed.
        tracker: Receives train_mse and val_mse per epoch (step = epoch index).

    Returns:
        The trained policy (in eval mode) and its loss history.
    """
    logger.info("training BC policy with seed=%d", config.seed)
    torch.manual_seed(config.seed)
    batches = torch.Generator().manual_seed(config.seed)
    train, val = split_by_lap(dataset, config.val_fraction, np.random.default_rng(config.seed))
    policy = BCPolicy(config.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    obs, actions = torch.from_numpy(train.obs), torch.from_numpy(train.actions)
    history: dict[str, list[float]] = {"train_mse": [], "val_mse": []}
    for epoch in range(config.epochs):
        order = torch.randperm(len(obs), generator=batches)
        for batch in torch.split(order, config.batch_size):
            train_step(policy, optimizer, obs[batch], actions[batch])
        losses = {"train_mse": mse(policy, train.obs, train.actions)}
        if val is not None:
            losses["val_mse"] = mse(policy, val.obs, val.actions)
        for key, value in losses.items():
            history[key].append(value)
            if tracker is not None:
                tracker.log_metric(key, value, step=epoch)
        summary = " ".join(f"{key}={value:.5f}" for key, value in losses.items())
        logger.info("epoch %d/%d %s", epoch + 1, config.epochs, summary)
    policy.eval()
    return BCResult(
        policy=policy,
        train_mse=tuple(history["train_mse"]),
        val_mse=tuple(history["val_mse"]),
        train_laps=train.lap_ids,
        val_laps=val.lap_ids if val is not None else (),
    )


def run_bc_training(
    laps_dir: Path,
    output_path: Path,
    config: BCConfig,
    tracker: RunTracker,
    track_ids: Collection[str] | None = None,
) -> BCResult:
    """Load laps, train, save the weights and record the run as CLAUDE.md section 9 requires.

    Args:
        laps_dir: Directory holding <track_id>/<lap_id>.json and .npz files.
        output_path: Where to write the .pt weights, e.g. models/bc_v1.pt.
        config: Hyperparameters, including the seed.
        tracker: Run tracker (an MLflow run in production).
        track_ids: Only train on these tracks; None uses every track.

    Returns:
        The training result.
    """
    dataset = load_bc_dataset(laps_dir, track_ids)
    tracker.log_params(
        {
            **contract_params(),
            **{f"bc.{name}": value for name, value in dataclasses.asdict(config).items()},
            "contract_version": CONTRACT_VERSION,
            "reward_version": "none (behaviour cloning uses no reward)",
            "seed": config.seed,
            "track_ids": ",".join(dataset.track_ids),
            "laps": len(dataset.lap_ids),
            "samples": len(dataset.obs),
            "git_commit": git_commit(),
        }
    )
    result = train_bc(dataset, config, tracker)
    tracker.log_params({"train_laps": len(result.train_laps), "val_laps": len(result.val_laps)})
    save_policy(result.policy, output_path)
    tracker.log_artifact(output_path)
    logger.info("saved BC policy to %s", output_path)
    return result
