"""Model registry operations shared by scripts and CI: the repo-root tracking store, and the
release gate's promotion of the challenger to champion (CLAUDE.md section 9).

get_tracking_uri resolves <repo root>/mlruns from this file's location rather than the working
directory, so a promotion updates the same registry the training scripts wrote to wherever it
runs from.

Training can move the challenger alias while a sweep or the gate runs. Promotion therefore checks
that both aliases still point at the versions the sweeps evaluated (promote_from_sweeps), so an
unevaluated version is never made champion.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from lapzero.io.champion import ChampionSnapshot, write_champion_snapshot
from lapzero.io.telemetry import TRACK_ID_PATTERN, ULID_PATTERN
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias, mlflow_client

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

    from lapzero.eval.sweep import SweepReport

logger = logging.getLogger(__name__)

DEFAULT_GATE_TRACK: Final = "oschersleben"
"""Track the release gate sweeps run on unless told otherwise."""
GATE_PROMOTE: Final = "promote"
"""gate_result tag value of a version the release gate promoted."""
CONTRACT_VERSION_PARAM: Final = "contract.CONTRACT_VERSION"
"""Run param holding the contract version; contract_params() logs it on every training run."""


def _utc_now() -> datetime:
    """Current UTC time to the second."""
    return datetime.now(UTC).replace(microsecond=0)


def get_tracking_uri() -> str:
    """Return the file: URI of the MLflow store at <repo root>/mlruns.

    Returns:
        For example "file:///home/me/lapzero/mlruns", whatever the working directory.
    """
    return (Path(__file__).resolve().parents[1] / "mlruns").as_uri()


def get_champion_version(
    *, tracking_uri: str | None = None, model_name: str = REGISTERED_MODEL
) -> str:
    """Return the model version the champion alias points at.

    Args:
        tracking_uri: MLflow tracking URI holding the registry; defaults to get_tracking_uri().
        model_name: Registered model name.

    Returns:
        The version number, e.g. "8".

    Raises:
        ValueError: If the model has no champion.
        mlflow.exceptions.MlflowException: If the model is not registered.
    """
    client = mlflow_client(tracking_uri or get_tracking_uri())
    return _alias_version(_aliases(client, model_name), ModelAlias.CHAMPION, model_name)


def export_champion(
    path: Path,
    *,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL,
    clock: Callable[[], datetime] = _utc_now,
) -> ChampionSnapshot:
    """Write the champion's registry metadata to a snapshot file, for an API without MLflow.

    The snapshot holds what GET /api/models/champion reports (version, run, contract version,
    tags, registration time) and when it was taken; no weights. Re-export after a promotion.

    Args:
        path: JSON file to write, e.g. backend/champion_snapshot.json; replaced if it exists.
        tracking_uri: MLflow tracking URI holding the registry; defaults to get_tracking_uri().
        model_name: Registered model name.
        clock: Returns the current UTC time; injectable for tests.

    Returns:
        The snapshot written.

    Raises:
        mlflow.exceptions.MlflowException: If the model or its champion alias does not exist.
    """
    client = mlflow_client(tracking_uri or get_tracking_uri())
    version = client.get_model_version_by_alias(model_name, ModelAlias.CHAMPION)
    params = client.get_run(version.run_id).data.params if version.run_id else {}
    snapshot = ChampionSnapshot(
        model_name=model_name,
        version=str(version.version),
        run_id=version.run_id or None,
        contract_version=params.get(CONTRACT_VERSION_PARAM),
        tags=dict(version.tags),
        created_at=datetime.fromtimestamp(version.creation_timestamp / 1000, tz=UTC),
        exported_at=clock(),
    )
    write_champion_snapshot(path, snapshot)
    logger.info(
        "wrote %s version %s (%s) to %s", model_name, snapshot.version, ModelAlias.CHAMPION, path
    )
    return snapshot


def promote_challenger(
    champion_sweep: str,
    challenger_sweep: str,
    gate_track: str = DEFAULT_GATE_TRACK,
    *,
    expected_champion_version: str | None = None,
    expected_challenger_version: str | None = None,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL,
) -> dict[str, str]:
    """Make the challenger the champion, recording the release gate's evidence on it.

    The challenger's version is tagged gate_result, gate_champion_sweep, gate_challenger_sweep,
    gate_track and replaced_champion_version; then the champion alias moves to it and the
    challenger alias is removed. This does not run the gate: call it only after a promote
    decision (lapzero.eval.gate.decide). promote_from_sweeps fills every argument from the two
    sweep reports.

    Args:
        champion_sweep: sweep_id (ULID) of the champion's friction sweep.
        challenger_sweep: sweep_id (ULID) of the challenger's friction sweep.
        gate_track: Track ID both sweeps ran on.
        expected_champion_version: If given, the version the champion alias must point at.
        expected_challenger_version: If given, the version the challenger alias must point at.
        tracking_uri: MLflow tracking URI holding the registry; defaults to get_tracking_uri().
        model_name: Registered model name.

    Returns:
        The model's aliases after promotion, alias to version, e.g. {"champion": "9"}.

    Raises:
        ValueError: If a sweep ID is not a ULID, gate_track is not a track ID, the model lacks a
            champion or a challenger, or an alias points at another version than expected. The
            registry is left unchanged.
        mlflow.exceptions.MlflowException: If the model is not registered.
    """
    _check_id("champion_sweep", champion_sweep, ULID_PATTERN)
    _check_id("challenger_sweep", challenger_sweep, ULID_PATTERN)
    _check_id("gate_track", gate_track, TRACK_ID_PATTERN)
    client = mlflow_client(tracking_uri or get_tracking_uri())
    aliases = _aliases(client, model_name)
    challenger = _alias_version(aliases, ModelAlias.CHALLENGER, model_name)
    champion = _alias_version(aliases, ModelAlias.CHAMPION, model_name)
    _check_expected(ModelAlias.CHAMPION, champion, expected_champion_version, model_name)
    _check_expected(ModelAlias.CHALLENGER, challenger, expected_challenger_version, model_name)
    tags = {
        "gate_result": GATE_PROMOTE,
        "gate_champion_sweep": champion_sweep,
        "gate_challenger_sweep": challenger_sweep,
        "gate_track": gate_track,
        "replaced_champion_version": champion,
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, challenger, key, value)
    client.set_registered_model_alias(model_name, ModelAlias.CHAMPION, challenger)
    client.delete_registered_model_alias(model_name, ModelAlias.CHALLENGER)
    logger.info(
        "promoted %s version %s to champion, replacing version %s", model_name, challenger, champion
    )
    return _aliases(client, model_name)


def promote_from_sweeps(
    champion: SweepReport,
    challenger: SweepReport,
    *,
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL,
) -> dict[str, str]:
    """Promote the challenger the gate passed, guarded by the versions its sweeps evaluated.

    Calls promote_challenger with the sweeps' IDs, their track, and their policy versions as the
    expected champion and challenger versions.

    Args:
        champion: Sweep report of the champion, as read by lapzero.eval.sweep.read_sweep.
        challenger: Sweep report of the challenger.
        tracking_uri: MLflow tracking URI holding the registry; defaults to get_tracking_uri().
        model_name: Registered model name.

    Returns:
        The model's aliases after promotion, alias to version.

    Raises:
        ValueError: If a sweep evaluated a local checkpoint rather than a registry version, the
            sweeps ran on different tracks, or promote_challenger refuses. The registry is left
            unchanged.
        mlflow.exceptions.MlflowException: If the model is not registered.
    """
    expected_champion = _swept_version(ModelAlias.CHAMPION, champion)
    expected_challenger = _swept_version(ModelAlias.CHALLENGER, challenger)
    if champion.run.track_id != challenger.run.track_id:
        raise ValueError(
            f"sweeps ran on different tracks: champion {champion.run.track_id!r}, "
            f"challenger {challenger.run.track_id!r}"
        )
    return promote_challenger(
        champion.run.sweep_id,
        challenger.run.sweep_id,
        challenger.run.track_id,
        expected_champion_version=expected_champion,
        expected_challenger_version=expected_challenger,
        tracking_uri=tracking_uri,
        model_name=model_name,
    )


def _swept_version(role: ModelAlias, sweep: SweepReport) -> str:
    """Return the registry version a sweep evaluated, or raise ValueError for a checkpoint."""
    if sweep.run.policy_version is None:
        raise ValueError(
            f"{role} sweep {sweep.run.sweep_id} evaluated {sweep.run.policy!r}, which has no "
            "registry version; promotion needs both sweeps run from the model registry"
        )
    return sweep.run.policy_version


def _check_expected(alias: ModelAlias, actual: str, expected: str | None, model_name: str) -> None:
    """Raise ValueError if `alias` points at another version than `expected` (when given)."""
    if expected is not None and actual != expected:
        raise ValueError(
            f"{model_name}@{alias} is version {actual}, expected {expected}: the registry changed "
            "after the sweeps ran, so sweep the current versions again before promoting"
        )


def _check_id(field: str, value: str, pattern: str) -> None:
    """Raise ValueError naming `field` unless `value` matches `pattern`."""
    if re.fullmatch(pattern, value) is None:
        raise ValueError(f"{field} {value!r} does not match {pattern}")


def _aliases(client: MlflowClient, model_name: str) -> dict[str, str]:
    """Return the registered model's aliases, alias to version number."""
    # RegisteredModel.aliases is unannotated; MLflow maps alias names to version strings.
    aliases: dict[str, str] = dict(client.get_registered_model(model_name).aliases)
    return aliases


def _alias_version(aliases: dict[str, str], alias: ModelAlias, model_name: str) -> str:
    """Return the version `alias` points at, or raise ValueError naming the missing alias."""
    version = aliases.get(alias)
    if version is None:
        raise ValueError(f"{model_name} has no {alias} alias (aliases: {aliases or 'none'})")
    return version
