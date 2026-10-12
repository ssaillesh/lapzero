"""Integration test: record laps, run scripts/train_bc.py, check the weights, the MLflow run and
the model registry."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from mlflow.tracking import MlflowClient

from lapzero.training.policy import load_policy
from lapzero.training.tracking import ModelAlias, load_registered_policy
from tests.conftest import record_lap

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_bc_script_saves_weights_and_records_a_finished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laps = tmp_path / "laps"
    for seed, track_id in enumerate(["track_a", "track_a", "track_b"]):
        record_lap(laps, track_id=track_id, steps=6, seed=seed)
    output = tmp_path / "models" / "bc_v1.pt"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    command = [
        sys.executable,
        "scripts/train_bc.py",
        "--laps-dir",
        str(laps),
        "--output",
        str(output),
        "--tracking-uri",
        tracking_uri,
        "--epochs",
        "3",
        "--batch-size",
        "4",
        "--hidden-size",
        "16",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr

    with torch.no_grad():
        assert load_policy(output)(torch.zeros(113)).shape == (2,)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("lapzero-bc")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "bc_v1"
    assert run.data.params["track_ids"] == "track_a,track_b"
    assert run.data.params["contract_version"] == "1.0.0"
    assert len(client.get_metric_history(run.info.run_id, "train_mse")) == 3
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["policy"]  # the registered MLflow Model, not a raw weights file
    registry = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    champion = registry.get_model_version_by_alias("lapzero-policy", ModelAlias.CHAMPION)
    assert champion.run_id == run.info.run_id
    assert champion.tags == {"algorithm": "bc", "contract_version": "1.0.0"}
    registered = load_registered_policy(
        tracking_uri=tracking_uri, model_name="lapzero-policy", alias=ModelAlias.CHAMPION
    )
    obs = torch.rand(4, 113)
    with torch.no_grad():
        torch.testing.assert_close(registered(obs), load_policy(output)(obs))
