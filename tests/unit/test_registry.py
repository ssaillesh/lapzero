"""Unit tests for lapzero.registry against throwaway MLflow file stores under tmp_path.

No test may write to the repo's real mlruns/: default-URI tests patch get_tracking_uri.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from lapzero import registry
from lapzero.eval.sweep import SweepReport
from lapzero.io.champion import read_champion_snapshot
from lapzero.registry import (
    CONTRACT_VERSION_PARAM,
    DEFAULT_GATE_TRACK,
    GATE_PROMOTE,
    export_champion,
    get_champion_version,
    get_tracking_uri,
    promote_challenger,
    promote_from_sweeps,
)
from lapzero.training.tracking import ModelAlias, contract_params, mlflow_client
from tests.conftest import make_sweep_report, register_model_versions

MODEL = "test-policy"
REPO_ROOT = Path(__file__).resolve().parents[2]
CHAMPION_SWEEP = "01M4TGPRVQY5F71QAQKBSR5RKQ"
CHALLENGER_SWEEP = "01M4TH84XCQYGAD34XR0C5BTDH"


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test
    return (tmp_path / "mlruns").as_uri()


def _registry(tracking_uri: str, versions: int, aliases: dict[str, str]) -> None:
    register_model_versions(tracking_uri, MODEL, versions, aliases)


def _aliases(tracking_uri: str) -> dict[str, str]:
    return dict(mlflow_client(tracking_uri).get_registered_model(MODEL).aliases)


def _tags(tracking_uri: str, version: str) -> dict[str, str]:
    return dict(mlflow_client(tracking_uri).get_model_version(MODEL, version).tags)


def _promote(tracking_uri: str, **overrides: str) -> dict[str, str]:
    args = {
        "champion_sweep": CHAMPION_SWEEP,
        "challenger_sweep": CHALLENGER_SWEEP,
        "gate_track": "track_a",
    }
    return promote_challenger(**(args | overrides), tracking_uri=tracking_uri, model_name=MODEL)


# Tracking URI


def test_tracking_uri_is_the_repo_root_mlruns_whatever_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_tracking_uri() == (REPO_ROOT / "mlruns").as_uri()


def test_registry_calls_default_to_the_resolved_tracking_uri(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "get_tracking_uri", lambda: tracking_uri)
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"})
    assert get_champion_version(model_name=MODEL) == "1"
    promote_challenger(CHAMPION_SWEEP, CHALLENGER_SWEEP, model_name=MODEL)
    assert _aliases(tracking_uri) == {"champion": "2"}
    assert _tags(tracking_uri, "2")["gate_track"] == DEFAULT_GATE_TRACK


# Champion lookup


def test_champion_version_is_the_version_the_champion_alias_points_at(tracking_uri: str) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "2", ModelAlias.CHALLENGER: "3"})
    assert get_champion_version(tracking_uri=tracking_uri, model_name=MODEL) == "2"


def test_champion_version_raises_when_the_model_has_no_champion(tracking_uri: str) -> None:
    _registry(tracking_uri, 1, {})
    with pytest.raises(ValueError, match="no champion alias"):
        get_champion_version(tracking_uri=tracking_uri, model_name=MODEL)


def test_champion_version_raises_when_the_model_is_not_registered(tracking_uri: str) -> None:
    with pytest.raises(MlflowException):
        get_champion_version(tracking_uri=tracking_uri, model_name=MODEL)


# Promotion


def test_promotion_moves_champion_to_the_challenger_and_removes_the_challenger_alias(
    tracking_uri: str,
) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    aliases = _promote(tracking_uri)
    assert aliases == {"champion": "3"}
    assert _aliases(tracking_uri) == aliases
    assert get_champion_version(tracking_uri=tracking_uri, model_name=MODEL) == "3"


def test_promotion_tags_the_new_champion_with_the_gate_record(tracking_uri: str) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    _promote(tracking_uri)
    assert _tags(tracking_uri, "3") == {
        "algorithm": "ppo",
        "gate_result": GATE_PROMOTE,
        "gate_champion_sweep": CHAMPION_SWEEP,
        "gate_challenger_sweep": CHALLENGER_SWEEP,
        "gate_track": "track_a",
        "replaced_champion_version": "1",
    }
    assert _tags(tracking_uri, "1") == {"algorithm": "ppo"}  # the old champion is untouched


@pytest.mark.parametrize(
    ("aliases", "missing"),
    [
        ({ModelAlias.CHAMPION: "1"}, "challenger"),
        ({ModelAlias.CHALLENGER: "2"}, "champion"),
    ],
)
def test_promotion_without_both_aliases_raises_and_changes_nothing(
    tracking_uri: str, aliases: dict[str, str], missing: str
) -> None:
    _registry(tracking_uri, 2, aliases)
    with pytest.raises(ValueError, match=f"no {missing} alias"):
        _promote(tracking_uri)
    assert _aliases(tracking_uri) == aliases
    assert all(_tags(tracking_uri, v) == {"algorithm": "ppo"} for v in ("1", "2"))


def test_a_challenger_can_only_be_promoted_once(tracking_uri: str) -> None:
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"})
    _promote(tracking_uri)
    with pytest.raises(ValueError, match="no challenger alias"):
        _promote(tracking_uri)
    assert _aliases(tracking_uri) == {"champion": "2"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("champion_sweep", f"data/eval/friction_sweep/oschersleben/{CHAMPION_SWEEP}"),
        ("challenger_sweep", ""),
        ("challenger_sweep", CHALLENGER_SWEEP.lower()),
        ("gate_track", "tracks/oschersleben"),
    ],
)
def test_promotion_refuses_values_that_are_not_sweep_or_track_ids(
    tracking_uri: str, field: str, value: str
) -> None:
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"})
    with pytest.raises(ValueError, match=field):
        _promote(tracking_uri, **{field: value})
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}
    assert _tags(tracking_uri, "2") == {"algorithm": "ppo"}


# Version guards: training may move an alias between the sweeps and the promotion


@pytest.mark.parametrize(
    ("expected", "alias"),
    [
        ({"expected_champion_version": "2"}, "champion"),
        ({"expected_challenger_version": "2"}, "challenger"),
    ],
)
def test_promotion_refuses_when_an_alias_no_longer_points_at_the_expected_version(
    tracking_uri: str, expected: dict[str, str], alias: str
) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    with pytest.raises(ValueError, match=f"{MODEL}@{alias} is version [13], expected 2"):
        _promote(tracking_uri, **expected)
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "3"}
    assert all(_tags(tracking_uri, v) == {"algorithm": "ppo"} for v in ("1", "2", "3"))


def test_promotion_proceeds_when_both_expected_versions_match(tracking_uri: str) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    aliases = _promote(tracking_uri, expected_champion_version="1", expected_challenger_version="3")
    assert aliases == {"champion": "3"}
    assert _tags(tracking_uri, "3")["replaced_champion_version"] == "1"


# Promotion from sweep reports


def _sweep(
    sweep_id: str, version: str | None, alias: str, track_id: str = "track_b"
) -> SweepReport:
    """A sweep report of `version` (None: a local checkpoint) on `track_id`."""
    return make_sweep_report(version, alias=alias, sweep_id=sweep_id, track_id=track_id)


def _promote_from_sweeps(
    tracking_uri: str, champion: SweepReport, challenger: SweepReport
) -> dict[str, str]:
    return promote_from_sweeps(champion, challenger, tracking_uri=tracking_uri, model_name=MODEL)


def test_promotion_from_sweeps_records_their_ids_and_track(tracking_uri: str) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    aliases = _promote_from_sweeps(
        tracking_uri,
        _sweep(CHAMPION_SWEEP, "1", "champion"),
        _sweep(CHALLENGER_SWEEP, "3", "challenger"),
    )
    assert aliases == {"champion": "3"}
    tags = _tags(tracking_uri, "3")
    assert tags["gate_champion_sweep"] == CHAMPION_SWEEP
    assert tags["gate_challenger_sweep"] == CHALLENGER_SWEEP
    assert tags["gate_track"] == "track_b"  # from the sweeps, not DEFAULT_GATE_TRACK


def test_promotion_from_sweeps_refuses_a_challenger_that_is_not_the_one_swept(
    tracking_uri: str,
) -> None:
    _registry(tracking_uri, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "3"})
    with pytest.raises(ValueError, match="challenger is version 3, expected 2"):
        _promote_from_sweeps(
            tracking_uri,
            _sweep(CHAMPION_SWEEP, "1", "champion"),
            _sweep(CHALLENGER_SWEEP, "2", "challenger"),
        )
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "3"}


@pytest.mark.parametrize("checkpoint_role", ["champion", "challenger"])
def test_promotion_from_sweeps_refuses_sweeps_of_local_checkpoints(
    tracking_uri: str, checkpoint_role: str
) -> None:
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"})
    champion = _sweep(CHAMPION_SWEEP, None if checkpoint_role == "champion" else "1", "champion")
    challenger = _sweep(
        CHALLENGER_SWEEP, None if checkpoint_role == "challenger" else "2", "challenger"
    )
    with pytest.raises(ValueError, match=f"{checkpoint_role} sweep .* no registry version"):
        _promote_from_sweeps(tracking_uri, champion, challenger)
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}


def test_promotion_from_sweeps_refuses_sweeps_on_different_tracks(tracking_uri: str) -> None:
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"})
    with pytest.raises(ValueError, match="different tracks"):
        _promote_from_sweeps(
            tracking_uri,
            _sweep(CHAMPION_SWEEP, "1", "champion", track_id="track_a"),
            _sweep(CHALLENGER_SWEEP, "2", "challenger", track_id="track_b"),
        )
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}


# Champion snapshot

EXPORTED_AT = datetime(2026, 10, 14, 6, 30, tzinfo=UTC)


def _version_with_run(tracking_uri: str, contract_version: str) -> str:
    """Register MODEL version 1 from a run that logged `contract_version`; return the run ID."""
    client = mlflow_client(tracking_uri)
    run_id = client.create_run(client.create_experiment("train")).info.run_id
    client.log_param(run_id, CONTRACT_VERSION_PARAM, contract_version)
    client.create_registered_model(MODEL)
    tags = {"gate_result": GATE_PROMOTE, "gate_track": "oschersleben"}
    client.create_model_version(MODEL, "file:///weights/1", run_id=run_id, tags=tags)
    client.set_registered_model_alias(MODEL, ModelAlias.CHAMPION, "1")
    return run_id


def test_the_contract_version_param_is_the_one_training_logs() -> None:
    assert CONTRACT_VERSION_PARAM in contract_params()


def test_export_writes_the_champion_version_its_run_and_its_gate_tags(
    tracking_uri: str, tmp_path: Path
) -> None:
    run_id = _version_with_run(tracking_uri, "1.1.0")
    path = tmp_path / "backend" / "champion_snapshot.json"

    snapshot = export_champion(
        path, tracking_uri=tracking_uri, model_name=MODEL, clock=lambda: EXPORTED_AT
    )

    assert read_champion_snapshot(path) == snapshot
    assert (snapshot.model_name, snapshot.version, snapshot.run_id) == (MODEL, "1", run_id)
    assert snapshot.contract_version == "1.1.0"
    assert snapshot.tags == {"gate_result": GATE_PROMOTE, "gate_track": "oschersleben"}
    assert snapshot.exported_at == EXPORTED_AT
    registered = mlflow_client(tracking_uri).get_model_version(MODEL, "1").creation_timestamp
    assert snapshot.created_at == datetime.fromtimestamp(registered / 1000, tz=UTC)


def test_export_of_a_version_without_a_run_has_no_run_or_contract(
    tracking_uri: str, tmp_path: Path
) -> None:
    _registry(tracking_uri, 2, {ModelAlias.CHAMPION: "2"})

    snapshot = export_champion(tmp_path / "c.json", tracking_uri=tracking_uri, model_name=MODEL)

    assert (snapshot.version, snapshot.run_id, snapshot.contract_version) == ("2", None, None)
    assert snapshot.tags == {"algorithm": "ppo"}


def test_export_without_a_champion_raises_and_writes_nothing(
    tracking_uri: str, tmp_path: Path
) -> None:
    _registry(tracking_uri, 1, {ModelAlias.CHALLENGER: "1"})

    with pytest.raises(MlflowException):
        export_champion(tmp_path / "c.json", tracking_uri=tracking_uri, model_name=MODEL)
    assert not (tmp_path / "c.json").exists()
