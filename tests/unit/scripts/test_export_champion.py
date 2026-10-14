"""Unit tests for scripts/export_champion.py, run in-process on a throwaway MLflow store."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from lapzero.io.champion import read_champion_snapshot
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias
from scripts.export_champion import main
from tests.conftest import register_model_versions


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test
    return (tmp_path / "mlruns").as_uri()


def test_the_champion_is_written_to_the_snapshot_and_summarized(
    tracking_uri: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_model_versions(tracking_uri, REGISTERED_MODEL, 3, {ModelAlias.CHAMPION: "2"})
    out = tmp_path / "champion_snapshot.json"

    assert main(["--out", str(out), "--tracking-uri", tracking_uri]) == 0

    snapshot = read_champion_snapshot(out)
    assert (snapshot.model_name, snapshot.version) == (REGISTERED_MODEL, "2")
    assert f"{out}: {REGISTERED_MODEL} version 2, run -, gate -" in capsys.readouterr().out


def test_another_registered_model_can_be_exported(tracking_uri: str, tmp_path: Path) -> None:
    register_model_versions(tracking_uri, "other-policy", 1, {ModelAlias.CHAMPION: "1"})
    out = tmp_path / "c.json"

    args = ["--out", str(out), "--tracking-uri", tracking_uri, "--registered-model", "other-policy"]
    assert main(args) == 0
    assert read_champion_snapshot(out).model_name == "other-policy"


def test_a_registry_without_a_champion_fails_loudly(tracking_uri: str, tmp_path: Path) -> None:
    register_model_versions(tracking_uri, REGISTERED_MODEL, 1, {ModelAlias.CHALLENGER: "1"})

    with pytest.raises(MlflowException):
        main(["--out", str(tmp_path / "c.json"), "--tracking-uri", tracking_uri])
