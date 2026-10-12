"""Unit tests for lapzero/training/tracking.py: required params, git state and MLflow runs.

MLflow runs write to a file store in a temporary directory; no server is involved.
"""

import re
import subprocess
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from lapzero.training.tracking import contract_params, git_commit, mlflow_run


def test_contract_params_hold_every_scalar_contract_constant() -> None:
    params = contract_params()
    assert params["contract.OBS_DIM"] == 113
    assert params["contract.CONTRACT_VERSION"] == "1.0.0"
    assert params["contract.REVERSE_ENABLED"] is False
    assert "contract.OBS_LOW" not in params
    assert "contract.LIDAR_SLICE" not in params


def _git(repo: Path, *args: str) -> None:
    isolated = ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(["git", *isolated, *args], cwd=repo, check=True, capture_output=True)


def test_git_commit_reports_the_head_hash_and_uncommitted_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "file.txt").write_text("a")
    _git(tmp_path, "add", "file.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "a")
    commit = git_commit(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    (tmp_path / "file.txt").write_text("b")
    assert git_commit(tmp_path) == f"{commit}-dirty"


def test_git_commit_is_unknown_outside_a_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == "unknown"


def test_git_commit_is_unknown_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert git_commit(tmp_path) == "unknown"


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    return (tmp_path / "mlruns").as_uri()


def test_mlflow_run_records_params_metrics_and_artifacts(tracking_uri: str, tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    params = {f"param_{i}": i for i in range(150)}  # more than one MLflow batch
    with mlflow_run(tracking_uri=tracking_uri, experiment="test", run_name="run") as tracker:
        tracker.log_params(params)
        tracker.log_metric("train_mse", 0.5, step=0)
        tracker.log_metric("train_mse", 0.25, step=1)
        tracker.log_artifact(weights)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(tracker.run_id)
    history = client.get_metric_history(tracker.run_id, "train_mse")
    assert run.info.status == "FINISHED"
    assert run.data.params == {key: str(value) for key, value in params.items()}
    assert [(metric.step, metric.value) for metric in history] == [(0, 0.5), (1, 0.25)]
    assert [artifact.path for artifact in client.list_artifacts(tracker.run_id)] == ["weights.pt"]


def test_a_failing_block_marks_the_run_failed_and_reraises(tracking_uri: str) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker,
    ):
        raise RuntimeError("boom")
    run = MlflowClient(tracking_uri=tracking_uri).get_run(tracker.run_id)
    assert run.info.status == "FAILED"


def test_runs_of_the_same_experiment_share_it(tracking_uri: str) -> None:
    with mlflow_run(tracking_uri=tracking_uri, experiment="shared") as first:
        pass
    with mlflow_run(tracking_uri=tracking_uri, experiment="shared") as second:
        pass
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = {client.get_run(t.run_id).info.experiment_id for t in (first, second)}
    assert len(experiments) == 1


def test_an_explicit_file_store_refusal_is_respected(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    with (
        pytest.raises(MlflowException, match="maintenance mode"),
        mlflow_run(tracking_uri=tracking_uri, experiment="test"),
    ):
        pass
