"""Integration smoke test of BC then PPO: scripts/train_bc.py registers the champion, then
scripts/train_ppo.py fine-tunes it on F1TENTH Gym and registers the challenger.

PPO uses a synthetic 20 m x 20 m walled room centred on the origin (inner wall faces 9.7 m away)
with a 5 m radius circular centreline, so no map download is needed.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient

from lapzero.sim.reward import REWARD_VERSION
from lapzero.training.policy import load_policy
from lapzero.training.tracking import ModelAlias, load_registered_policy
from tests.conftest import record_lap

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_room(directory: Path) -> Path:
    """Write a ROS map (PGM + yaml) of an empty room with 0.3 m thick walls."""
    size_px, wall_px = 400, 6  # 0.05 m per pixel
    pixels = np.full((size_px, size_px), 255, dtype=np.uint8)
    pixels[:wall_px, :] = pixels[-wall_px:, :] = 0
    pixels[:, :wall_px] = pixels[:, -wall_px:] = 0
    header = f"P5\n{size_px} {size_px}\n255\n".encode()
    (directory / "room.pgm").write_bytes(header + pixels.tobytes())
    (directory / "room.yaml").write_text(
        "image: room.pgm\nresolution: 0.05\norigin: [-10.0, -10.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    return directory / "room.yaml"


def _write_circle_centreline(path: Path, radius: float = 5.0, points: int = 72) -> Path:
    """Write a CCW circular centreline starting at (0, -radius), in f1tenth_racetracks format."""
    angles = -math.pi / 2 + np.linspace(0.0, 2 * math.pi, points, endpoint=False)
    rows = "\n".join(f"{radius * math.cos(a):.4f}, {radius * math.sin(a):.4f}" for a in angles)
    path.write_text(f"# x_m, y_m\n{rows}\n")
    return path


def _run(script: str, *args: str) -> None:
    command = [sys.executable, f"scripts/{script}", *args]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr


def test_ppo_fine_tunes_the_bc_champion_and_registers_a_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laps = tmp_path / "laps"
    for seed in range(3):
        record_lap(laps, steps=6, seed=seed)
    bc_path = tmp_path / "models" / "bc_v1.pt"
    output = tmp_path / "models" / "ppo_v1.pt"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    _run(
        "train_bc.py",
        *("--laps-dir", str(laps), "--output", str(bc_path), "--tracking-uri", tracking_uri),
        *("--epochs", "2", "--batch-size", "4"),
    )
    room = _write_room(tmp_path)
    centreline = _write_circle_centreline(tmp_path / "centreline.csv")
    args = [
        "--map",
        str(room),
        "--start",
        "0",
        "-5",
        "0",
        "--centreline",
        str(centreline),
        "--track-id",
        "room",
        "--bc-weights",
        str(bc_path),
        "--output",
        str(output),
        "--tracking-uri",
        tracking_uri,
        "--laps",
        "1",
        "--max-episode-s",
        "1.0",
        "--total-timesteps",
        "64",
        "--n-steps",
        "32",
        "--batch-size",
        "16",
        "--n-epochs",
        "2",
        "--critic-warmup-rollouts",
        "1",
    ]
    _run("train_ppo.py", *args)

    actor = load_policy(output)
    assert actor.hidden_size == 256
    with torch.no_grad():
        actions = actor(torch.zeros(113))
    assert actions.shape == (2,)
    assert torch.isfinite(actions).all()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("lapzero-ppo")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "ppo_v1"
    params = run.data.params
    assert params["track_ids"] == "room"
    assert params["contract_version"] == "1.0.0"
    assert params["reward_version"] == REWARD_VERSION
    assert params["warm_start"] == "bc_v1.pt"
    assert params["seed"] == "0"
    assert params["ppo.critic_warmup_rollouts"] == "1"
    for metric in ("train/approx_kl", "train/value_loss", "race/episode_reward"):
        assert client.get_metric_history(run.info.run_id, metric), f"{metric} not logged"
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["policy"]  # the registered MLflow Model, not a raw weights file

    registry = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    champion = registry.get_model_version_by_alias("lapzero-policy", ModelAlias.CHAMPION)
    challenger = registry.get_model_version_by_alias("lapzero-policy", ModelAlias.CHALLENGER)
    assert (str(champion.version), champion.tags["algorithm"]) == ("1", "bc")
    assert (str(challenger.version), challenger.tags["algorithm"]) == ("2", "ppo")
    assert challenger.run_id == run.info.run_id
    assert challenger.tags["reward_version"] == REWARD_VERSION
    obs = torch.rand(4, 113)
    with torch.no_grad():
        for alias, path in ((ModelAlias.CHAMPION, bc_path), (ModelAlias.CHALLENGER, output)):
            loaded = load_registered_policy(
                tracking_uri=tracking_uri, model_name="lapzero-policy", alias=alias
            )
            torch.testing.assert_close(loaded(obs), load_policy(path)(obs))
