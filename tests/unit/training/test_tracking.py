"""Unit tests for lapzero/training/tracking.py: required params, git state, MLflow runs and the
model registry.

MLflow runs and the registry write to a file store in a temporary directory; no server is involved.
"""

import re
import subprocess
from pathlib import Path

import pytest
import torch
from mlflow.exceptions import MlflowException
from mlflow.models import Model  # writing MLmodel files needs no pandas; Model.load does
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.tracking import MlflowClient

from lapzero.training.policy import BCPolicy, save_policy
from lapzero.training.tracking import (
    POLICY_FLAVOR,
    MlflowTracker,
    ModelAlias,
    RegisteredPolicy,
    alias_for_new_version,
    contract_params,
    git_commit,
    load_registered_policy,
    mlflow_client,
    mlflow_run,
    read_policy_flavor,
)

MODEL = "lapzero-policy"


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


# Model registry


@pytest.mark.parametrize(
    ("has_champion", "alias"), [(False, ModelAlias.CHAMPION), (True, ModelAlias.CHALLENGER)]
)
def test_a_new_version_is_champion_only_when_the_model_has_none(
    has_champion: bool, alias: ModelAlias
) -> None:
    assert alias_for_new_version(has_champion=has_champion) is alias


def _checkpoint(directory: Path, seed: int) -> tuple[Path, BCPolicy]:
    torch.manual_seed(seed)
    policy = BCPolicy(8)
    policy.eval()
    path = directory / f"policy_{seed}.pt"
    save_policy(policy, path)
    return path, policy


def _register(
    tracking_uri: str, checkpoint: Path, tags: dict[str, str] | None = None
) -> tuple[RegisteredPolicy, str]:
    with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
        registered = tracker.register_policy(checkpoint, MODEL, tags or {"algorithm": "bc"})
    return registered, tracker.run_id


def test_the_first_version_becomes_champion_and_later_ones_take_over_challenger(
    tracking_uri: str, tmp_path: Path
) -> None:
    registered = [_register(tracking_uri, _checkpoint(tmp_path, seed)[0])[0] for seed in range(3)]
    assert registered == [
        RegisteredPolicy(MODEL, "1", ModelAlias.CHAMPION),
        RegisteredPolicy(MODEL, "2", ModelAlias.CHALLENGER),
        RegisteredPolicy(MODEL, "3", ModelAlias.CHALLENGER),
    ]
    aliases = mlflow_client(tracking_uri).get_registered_model(MODEL).aliases
    assert aliases == {"champion": "1", "challenger": "3"}


def test_registered_policies_load_back_by_alias_with_identical_weights(
    tracking_uri: str, tmp_path: Path
) -> None:
    first_path, first = _checkpoint(tmp_path, 1)
    second_path, second = _checkpoint(tmp_path, 2)
    _register(tracking_uri, first_path)
    _register(tracking_uri, second_path)
    obs = torch.rand(4, 113)
    with torch.no_grad():
        for alias, expected in ((ModelAlias.CHAMPION, first), (ModelAlias.CHALLENGER, second)):
            loaded = load_registered_policy(
                tracking_uri=tracking_uri, model_name=MODEL, alias=alias
            )
            torch.testing.assert_close(loaded(obs), expected(obs))


def test_a_registered_version_is_a_tagged_mlflow_model_of_its_run(
    tracking_uri: str, tmp_path: Path
) -> None:
    tags = {"algorithm": "ppo", "reward_version": "1.0.0"}
    registered, run_id = _register(tracking_uri, _checkpoint(tmp_path, 1)[0], tags)
    client = mlflow_client(tracking_uri)
    version = client.get_model_version(MODEL, registered.version)
    assert version.run_id == run_id
    assert version.tags == tags
    files = sorted(artifact.path for artifact in client.list_artifacts(run_id, "policy"))
    assert files == ["policy/MLmodel", "policy/policy.pt"]
    mlmodel = Path(client.download_artifacts(run_id, "policy/MLmodel", str(tmp_path)))
    flavor = read_policy_flavor(mlmodel.parent)
    assert flavor == {"checkpoint": "policy.pt", "contract_version": "1.0.0"}


def test_loading_an_alias_that_was_never_set_raises(tracking_uri: str, tmp_path: Path) -> None:
    _register(tracking_uri, _checkpoint(tmp_path, 1)[0])  # champion only
    with pytest.raises(MlflowException, match="challenger"):
        load_registered_policy(
            tracking_uri=tracking_uri, model_name=MODEL, alias=ModelAlias.CHALLENGER
        )


def test_a_version_that_is_not_a_lapzero_policy_model_is_refused(
    tracking_uri: str, tmp_path: Path
) -> None:
    model_dir = tmp_path / "other"
    model_dir.mkdir()
    with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
        Model(run_id=tracker.run_id).add_flavor("other_flavor").save(str(model_dir / "MLmodel"))
    client = mlflow_client(tracking_uri)
    client.log_artifacts(tracker.run_id, str(model_dir), "other")
    client.create_registered_model(MODEL)
    version = client.create_model_version(MODEL, f"runs:/{tracker.run_id}/other", tracker.run_id)
    client.set_registered_model_alias(MODEL, ModelAlias.CHAMPION, version.version)
    with pytest.raises(ValueError, match=f"not a {POLICY_FLAVOR} model"):
        load_registered_policy(
            tracking_uri=tracking_uri, model_name=MODEL, alias=ModelAlias.CHAMPION
        )


class UnavailableRegistry:
    """Client stand-in whose registry fails with a server error, not a missing model."""

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None:
        pass

    def get_registered_model(self, name: str) -> None:
        raise MlflowException("registry unavailable", error_code=INTERNAL_ERROR)


def test_registry_errors_other_than_a_missing_model_are_raised(tmp_path: Path) -> None:
    tracker = MlflowTracker(UnavailableRegistry(), run_id="run")  # type: ignore[arg-type]
    with pytest.raises(MlflowException, match="registry unavailable"):
        tracker.register_policy(_checkpoint(tmp_path, 1)[0], MODEL, {})
