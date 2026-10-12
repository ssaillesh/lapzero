"""Experiment tracking for training runs (CLAUDE.md section 9): MLflow runs and required params.

Training code logs through the RunTracker protocol, so tests can record calls without MLflow.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from lapzero import contract

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

ParamValue = str | int | float | bool
_MLFLOW_PARAMS_PER_BATCH = 100  # MLflow's per-request limit for log_batch params


class RunTracker(Protocol):
    """Where a training run records its params, per-step metrics and output files."""

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Record run parameters; each key is logged once per run."""
        ...

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Record one metric value at a training step (e.g. an epoch)."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Attach an output file, such as model weights, to the run."""
        ...


def contract_params() -> dict[str, ParamValue]:
    """Return every scalar constant in lapzero.contract, keyed "contract.<NAME>".

    Returns:
        For example {"contract.OBS_DIM": 113, "contract.CONTRACT_VERSION": "1.0.0", ...}.
    """
    return {
        f"contract.{name}": value
        for name, value in sorted(vars(contract).items())
        if name.isupper() and isinstance(value, bool | int | float | str)
    }


def git_commit(repo: Path | None = None) -> str:
    """Return the checked-out commit hash, suffixed "-dirty" when there are uncommitted changes.

    Args:
        repo: Directory inside the repository; defaults to the current directory.

    Returns:
        The hash, or "unknown" when git or the repository is unavailable (logged as a warning).
    """
    try:
        commit = _git(["rev-parse", "HEAD"], repo)
        dirty = bool(_git(["status", "--porcelain"], repo))
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("could not read the git commit: %s", error)
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _git(args: list[str], repo: Path | None) -> str:
    """Run a git command and return its stripped stdout."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


class MlflowTracker:
    """RunTracker that writes to one MLflow run."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        """Wrap an open run.

        Args:
            client: Client for the run's tracking server or store.
            run_id: ID of the open run.
        """
        self._client = client
        self.run_id = run_id

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Record run parameters as strings, in batches MLflow accepts."""
        from mlflow.entities import Param

        items = [Param(key, str(value)) for key, value in params.items()]
        for start in range(0, len(items), _MLFLOW_PARAMS_PER_BATCH):
            batch = items[start : start + _MLFLOW_PARAMS_PER_BATCH]
            self._client.log_batch(self.run_id, params=batch)

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Record one metric value at a training step."""
        self._client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, path: Path) -> None:
        """Copy a file into the run's artifacts."""
        self._client.log_artifact(self.run_id, str(path))


@contextmanager
def mlflow_run(
    *, tracking_uri: str, experiment: str, run_name: str | None = None
) -> Iterator[MlflowTracker]:
    """Open an MLflow run, marked FINISHED when the block succeeds and FAILED if it raises.

    MLflow 3 only accepts a local file: store (e.g. ./mlruns) with MLFLOW_ALLOW_FILE_STORE set;
    for file: URIs it is set to true here unless the environment already sets it.

    Args:
        tracking_uri: MLflow tracking URI, e.g. a file: URI of ./mlruns or an http: server.
        experiment: Experiment name; created if missing.
        run_name: Optional display name of the run.

    Yields:
        A tracker for the open run.
    """
    from mlflow.tracking import MlflowClient  # heavy import, only when a run is opened

    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    experiment_id = found.experiment_id if found else client.create_experiment(experiment)
    run_id = client.create_run(experiment_id, run_name=run_name).info.run_id
    logger.info("MLflow run %s in experiment %r at %s", run_id, experiment, tracking_uri)
    status = "FAILED"
    try:
        yield MlflowTracker(client, run_id)
        status = "FINISHED"
    finally:
        client.set_terminated(run_id, status=status)
