"""Unit tests for backend/app/services/models.py against throwaway MLflow file stores."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from backend.app.services.models import (
    CONTRACT_VERSION_PARAM,
    ChampionNotFoundError,
    RegistryUnavailableError,
    get_champion,
)
from lapzero.training.tracking import ModelAlias, contract_params, mlflow_client
from tests.conftest import register_model_versions

MODEL = "test-policy"


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test
    return (tmp_path / "mlruns").as_uri()


def test_alias_and_param_names_match_what_training_writes() -> None:
    assert ModelAlias.CHAMPION == "champion"
    assert CONTRACT_VERSION_PARAM in contract_params()


def test_champion_is_the_version_the_alias_points_at(tracking_uri: str) -> None:
    register_model_versions(tracking_uri, MODEL, 3, {"champion": "2", "challenger": "3"})

    champion = get_champion(tracking_uri, MODEL)

    assert (champion.model_name, champion.version) == (MODEL, "2")
    assert champion.tags == {"algorithm": "ppo"}
    assert champion.created_at.tzinfo is UTC
    assert abs((datetime.now(UTC) - champion.created_at).total_seconds()) < 60


def test_version_registered_without_a_run_has_no_run_or_contract_version(
    tracking_uri: str,
) -> None:
    register_model_versions(tracking_uri, MODEL, 1, {"champion": "1"})

    champion = get_champion(tracking_uri, MODEL)

    assert (champion.run_id, champion.contract_version) == (None, None)


def test_champion_reports_the_contract_version_its_run_logged(tracking_uri: str) -> None:
    client = mlflow_client(tracking_uri)
    run = client.create_run(client.create_experiment("train"))
    client.log_param(run.info.run_id, CONTRACT_VERSION_PARAM, "1.1.0")
    client.create_registered_model(MODEL)
    client.create_model_version(MODEL, "file:///weights", run_id=run.info.run_id)
    client.set_registered_model_alias(MODEL, "champion", "1")

    champion = get_champion(tracking_uri, MODEL)

    assert (champion.run_id, champion.contract_version) == (run.info.run_id, "1.1.0")


def test_model_without_a_champion_alias_is_not_found(tracking_uri: str) -> None:
    register_model_versions(tracking_uri, MODEL, 1, {"challenger": "1"})

    with pytest.raises(ChampionNotFoundError, match="no champion alias"):
        get_champion(tracking_uri, MODEL)


def test_unregistered_model_is_not_found(tracking_uri: str) -> None:
    with pytest.raises(ChampionNotFoundError, match=MODEL):
        get_champion(tracking_uri, MODEL)


def test_other_registry_failures_mean_the_registry_is_unavailable(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object) -> None:
        raise MlflowException("connection refused")  # error_code INTERNAL_ERROR

    monkeypatch.setattr(MlflowClient, "get_registered_model", fail)

    with pytest.raises(RegistryUnavailableError, match="INTERNAL_ERROR"):
        get_champion(tracking_uri, MODEL)


def test_file_store_is_allowed_when_the_environment_does_not_say(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE")  # the original value is restored afterwards

    with pytest.raises(ChampionNotFoundError):
        get_champion(tracking_uri, MODEL)

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
