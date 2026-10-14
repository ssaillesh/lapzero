"""Model history from the MLflow model registry: the version the champion alias points at.

The backend must not import lapzero.training, so the alias name and the run param holding the
contract version are repeated here; they match lapzero.training.tracking (ModelAlias.CHAMPION and
contract_params()). An API without the registry reads the champion from a snapshot file instead
(get_champion_from_snapshot).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from lapzero.io.champion import read_champion_snapshot

logger = logging.getLogger(__name__)

CHAMPION_ALIAS: Final = "champion"
CONTRACT_VERSION_PARAM: Final = "contract.CONTRACT_VERSION"
_NOT_FOUND: Final = "RESOURCE_DOES_NOT_EXIST"  # MlflowException.error_code of a missing entity


class ChampionNotFoundError(LookupError):
    """Raised when the model is not registered or has no champion alias."""


class RegistryUnavailableError(RuntimeError):
    """Raised when the model registry cannot be read."""


@dataclass(frozen=True, slots=True)
class ChampionVersion:
    """The registered model version the champion alias points at.

    Attributes:
        model_name: Registered model name.
        version: Registry version number, e.g. "8".
        run_id: MLflow run that trained the version, or None.
        contract_version: The run's logged contract version, or None if unknown.
        tags: Version tags, e.g. gate_result set by the release gate.
        created_at: When the version was registered (UTC).
    """

    model_name: str
    version: str
    run_id: str | None
    contract_version: str | None
    tags: dict[str, str]
    created_at: datetime


def get_champion(tracking_uri: str, model_name: str) -> ChampionVersion:
    """Read the champion version of a registered model.

    Args:
        tracking_uri: MLflow tracking URI that also serves the registry.
        model_name: Registered model name.

    Returns:
        The champion version.

    Raises:
        ChampionNotFoundError: If the model is not registered or has no champion.
        RegistryUnavailableError: If MLflow fails for another reason (e.g. server down).
    """
    from mlflow.exceptions import MlflowException  # heavy import, only when the route is used
    from mlflow.tracking import MlflowClient

    if tracking_uri.startswith("file:"):
        # MLflow 3 only reads a local file store with this set, as lapzero.training does.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        # RegisteredModel.aliases is unannotated; MLflow maps alias names to version strings.
        aliases: dict[str, str] = dict(client.get_registered_model(model_name).aliases)
        number = aliases.get(CHAMPION_ALIAS)
        if number is None:
            raise ChampionNotFoundError(f"{model_name} has no {CHAMPION_ALIAS} alias")
        version = client.get_model_version(model_name, number)
        params = client.get_run(version.run_id).data.params if version.run_id else {}
    except MlflowException as error:
        if error.error_code == _NOT_FOUND:
            raise ChampionNotFoundError(f"{model_name} champion not found: {error}") from error
        logger.error("model registry at %s failed: %s", tracking_uri, error)
        raise RegistryUnavailableError(f"model registry failed: {error.error_code}") from error
    return ChampionVersion(
        model_name=model_name,
        version=str(version.version),
        run_id=version.run_id or None,
        contract_version=params.get(CONTRACT_VERSION_PARAM),
        tags=dict(version.tags),
        created_at=datetime.fromtimestamp(version.creation_timestamp / 1000, tz=UTC),
    )


def get_champion_from_snapshot(path: Path, model_name: str) -> ChampionVersion:
    """Read the champion from a snapshot written by scripts/export_champion.py.

    Args:
        path: The CHAMPION_SNAPSHOT file.
        model_name: Registered model the API reports on; the snapshot must be of this model.

    Returns:
        The champion version the snapshot recorded.

    Raises:
        ChampionNotFoundError: If the snapshot is of another model.
        RegistryUnavailableError: If the file is missing or is not a valid snapshot.
    """
    try:
        snapshot = read_champion_snapshot(path)
    except (OSError, ValidationError) as error:
        logger.error("champion snapshot %s cannot be read: %s", path, error)
        raise RegistryUnavailableError(f"champion snapshot {path.name} cannot be read") from error
    if snapshot.model_name != model_name:
        raise ChampionNotFoundError(
            f"champion snapshot {path.name} is of {snapshot.model_name}, not {model_name}"
        )
    return ChampionVersion(
        model_name=snapshot.model_name,
        version=snapshot.version,
        run_id=snapshot.run_id,
        contract_version=snapshot.contract_version,
        tags=dict(snapshot.tags),
        created_at=snapshot.created_at,
    )
