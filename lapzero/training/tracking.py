"""Experiment tracking for training runs (CLAUDE.md section 9): MLflow runs, required params and
the model registry with its champion and challenger aliases.

Training code logs through the RunTracker protocol, so tests can record calls without MLflow.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from lapzero import contract
from lapzero.training.policy import BCPolicy, load_policy

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

ParamValue = str | int | float | bool
_MLFLOW_PARAMS_PER_BATCH = 100  # MLflow's per-request limit for log_batch params

REGISTERED_MODEL: Final = "lapzero-policy"
"""Default registered model name. BC and PPO versions share it, so the release gate can compare
a challenger against the champion whatever algorithm produced each."""
POLICY_FLAVOR: Final = "lapzero_policy"
"""MLmodel flavor of a registered policy: a save_policy checkpoint, loaded with load_policy."""
_POLICY_ARTIFACT = "policy"  # run artifact directory holding MLmodel and the checkpoint
_POLICY_CHECKPOINT = "policy.pt"


class ModelAlias(StrEnum):
    """Model registry aliases (CLAUDE.md section 9)."""

    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass(frozen=True, slots=True)
class RegisteredPolicy:
    """A model version created by RunTracker.register_policy, and the alias it was given."""

    name: str
    version: str
    alias: ModelAlias


def alias_for_new_version(*, has_champion: bool) -> ModelAlias:
    """Return the alias a newly trained model version gets.

    Training never replaces a champion: once one exists, only the release gate promotes, so a
    new version becomes the challenger. The first version of a registered model becomes the
    champion, giving the gate a baseline (normally the BC policy) to compare challengers with.

    Args:
        has_champion: Whether the registered model already has a champion.

    Returns:
        CHALLENGER if a champion exists, otherwise CHAMPION.
    """
    return ModelAlias.CHALLENGER if has_champion else ModelAlias.CHAMPION


class RunTracker(Protocol):
    """Where a training run records its params, per-step metrics and output files."""

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Record run parameters; each key is logged once per run."""
        ...

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Record one metric value at a training step (e.g. an epoch)."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Attach an output file, such as a plot, to the run."""
        ...

    def register_policy(
        self, checkpoint: Path, model_name: str, tags: Mapping[str, str]
    ) -> RegisteredPolicy:
        """Register a save_policy checkpoint as a new model version of `model_name`.

        The version is aliased as alias_for_new_version decides and tagged with `tags`.
        """
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

    def register_policy(
        self, checkpoint: Path, model_name: str, tags: Mapping[str, str]
    ) -> RegisteredPolicy:
        """Log a checkpoint as an MLflow Model of this run and register it as a new version.

        The run gets a "policy" artifact directory holding the checkpoint and an MLmodel file
        with the lapzero_policy flavor (contract_version included). The registered model is
        created if missing.

        Args:
            checkpoint: File written by save_policy.
            model_name: Registered model name, normally REGISTERED_MODEL.
            tags: Model version tags, e.g. {"algorithm": "ppo"}.

        Returns:
            The new version and the alias it was given.
        """
        from mlflow.exceptions import MlflowException
        from mlflow.models import Model

        with tempfile.TemporaryDirectory() as staging:
            model_dir = Path(staging)
            shutil.copyfile(checkpoint, model_dir / _POLICY_CHECKPOINT)
            model = Model(run_id=self.run_id, artifact_path=_POLICY_ARTIFACT)
            model.add_flavor(
                POLICY_FLAVOR,
                checkpoint=_POLICY_CHECKPOINT,
                contract_version=contract.CONTRACT_VERSION,
            )
            model.save(str(model_dir / "MLmodel"))
            self._client.log_artifacts(self.run_id, str(model_dir), _POLICY_ARTIFACT)
        try:
            registered = self._client.get_registered_model(model_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            registered = self._client.create_registered_model(model_name)
        alias = alias_for_new_version(has_champion=ModelAlias.CHAMPION in registered.aliases)
        source = f"runs:/{self.run_id}/{_POLICY_ARTIFACT}"
        version = self._client.create_model_version(
            model_name, source, run_id=self.run_id, tags=dict(tags)
        )
        self._client.set_registered_model_alias(model_name, alias, version.version)
        logger.info("registered %s version %s as %s", model_name, version.version, alias)
        return RegisteredPolicy(model_name, str(version.version), alias)


def mlflow_client(tracking_uri: str) -> MlflowClient:
    """Return a client for a tracking server or store, which also serves as the model registry.

    MLflow 3 only accepts a local file: store (e.g. ./mlruns) with MLFLOW_ALLOW_FILE_STORE set;
    for file: URIs it is set to true here unless the environment already sets it.

    Args:
        tracking_uri: MLflow tracking URI, e.g. a file: URI of ./mlruns or an http: server.

    Returns:
        The client.
    """
    from mlflow.tracking import MlflowClient  # heavy import, only when MLflow is used

    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def read_policy_flavor(model_dir: Path) -> dict[str, str] | None:
    """Return the lapzero_policy flavor of an MLflow Model directory, or None if it has none.

    Reads the MLmodel YAML directly: mlflow.models.Model.load needs pandas, which LapZero does
    not install.
    """
    from mlflow.utils.yaml_utils import read_yaml

    # read_yaml is unannotated; MLmodel is a YAML mapping with a "flavors" mapping.
    mlmodel: dict[str, Any] = read_yaml(str(model_dir), "MLmodel")  # type: ignore[no-untyped-call]
    flavor: dict[str, str] | None = mlmodel.get("flavors", {}).get(POLICY_FLAVOR)
    return flavor


def load_registered_policy(*, tracking_uri: str, model_name: str, alias: str) -> BCPolicy:
    """Download the model version that `alias` points to and load its policy.

    Args:
        tracking_uri: MLflow tracking URI holding the registry.
        model_name: Registered model name, normally REGISTERED_MODEL.
        alias: Version alias, e.g. ModelAlias.CHAMPION.

    Returns:
        The policy, in eval mode.

    Raises:
        mlflow.exceptions.MlflowException: If the model or alias does not exist.
        ValueError: If the version is not a lapzero_policy MLflow Model.
        ContractViolationError: If the policy was saved under another contract major version.
    """
    import mlflow.artifacts

    version = mlflow_client(tracking_uri).get_model_version_by_alias(model_name, alias)
    with tempfile.TemporaryDirectory() as download:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=version.source, dst_path=download, tracking_uri=tracking_uri
        )
        flavor = read_policy_flavor(Path(local))
        if flavor is None:
            raise ValueError(
                f"{model_name} version {version.version} ({alias}) is not a {POLICY_FLAVOR} model"
            )
        return load_policy(Path(local) / flavor["checkpoint"])


@contextmanager
def mlflow_run(
    *, tracking_uri: str, experiment: str, run_name: str | None = None
) -> Iterator[MlflowTracker]:
    """Open an MLflow run, marked FINISHED when the block succeeds and FAILED if it raises.

    Args:
        tracking_uri: MLflow tracking URI, e.g. a file: URI of ./mlruns or an http: server.
        experiment: Experiment name; created if missing.
        run_name: Optional display name of the run.

    Yields:
        A tracker for the open run.
    """
    client = mlflow_client(tracking_uri)
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
