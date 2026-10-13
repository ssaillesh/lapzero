"""Integration smoke tests of BC then PPO: scripts/train_bc.py registers the champion, then
scripts/train_ppo.py fine-tunes it on F1TENTH Gym (one track, or several at once with one worker
process each) and registers the challenger; scripts/eval.py scores a registered or local policy
noise-free, or under friction and sensor noise with exports.

PPO uses a synthetic 20 m x 20 m walled room centred on the origin (inner wall faces 9.7 m away)
with a 5 m radius circular centreline, so no map download is needed. Multi-track runs add a
10 m x 10 m room with a 2.5 m radius centreline.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient

from lapzero.contract import LIDAR_SLICE
from lapzero.sim.env import EnvConfig
from lapzero.sim.reward import REWARD_VERSION, RewardConfig
from lapzero.training.policy import BCPolicy, load_policy, save_policy
from lapzero.training.tracking import ModelAlias, load_registered_policy, mlflow_run
from lapzero.training.vec_env import TrackEnvFactory, make_track_vec_env, track_files
from tests.conftest import record_lap

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_room(directory: Path, name: str = "room", size_px: int = 400) -> Path:
    """Write a ROS map (PGM + yaml) of an empty room centred on the origin, 0.3 m thick walls."""
    wall_px = 6  # 0.05 m per pixel
    pixels = np.full((size_px, size_px), 255, dtype=np.uint8)
    pixels[:wall_px, :] = pixels[-wall_px:, :] = 0
    pixels[:, :wall_px] = pixels[:, -wall_px:] = 0
    header = f"P5\n{size_px} {size_px}\n255\n".encode()
    (directory / f"{name}.pgm").write_bytes(header + pixels.tobytes())
    half = size_px * 0.05 / 2
    (directory / f"{name}.yaml").write_text(
        f"image: {name}.pgm\nresolution: 0.05\norigin: [{-half}, {-half}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    return directory / f"{name}.yaml"


def _write_circle_centreline(path: Path, radius: float = 5.0, points: int = 72) -> Path:
    """Write a CCW circular centreline starting at (0, -radius), in f1tenth_racetracks format."""
    angles = -math.pi / 2 + np.linspace(0.0, 2 * math.pi, points, endpoint=False)
    rows = "\n".join(f"{radius * math.cos(a):.4f}, {radius * math.sin(a):.4f}" for a in angles)
    path.write_text(f"# x_m, y_m\n{rows}\n")
    return path


def _write_track(tracks_dir: Path, track_id: str, size_px: int) -> None:
    """Write <tracks_dir>/<track_id>/ as train_ppo.py --tracks reads it: a room map and a circular
    centreline halfway between the room's centre and its walls."""
    directory = tracks_dir / track_id
    directory.mkdir(parents=True)
    _write_room(directory, track_id, size_px)
    _write_circle_centreline(directory / "centreline.csv", radius=size_px * 0.05 / 4)


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
    assert params["contract_version"] == "1.1.0"
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


def test_ppo_with_a_friction_range_logs_a_fresh_friction_for_each_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bc_path = tmp_path / "bc.pt"
    torch.manual_seed(0)
    save_policy(BCPolicy(16), bc_path)
    room = _write_room(tmp_path)
    centreline = _write_circle_centreline(tmp_path / "centreline.csv")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    _run(
        "train_ppo.py",
        *("--map", str(room), "--start", "0", "-5", "0", "--centreline", str(centreline)),
        *("--track-id", "room", "--bc-weights", str(bc_path), "--output", str(tmp_path / "o.pt")),
        *("--tracking-uri", tracking_uri, "--laps", "1", "--max-episode-s", "1.0"),
        *("--total-timesteps", "128", "--n-steps", "32", "--batch-size", "16"),
        *("--n-epochs", "1", "--critic-warmup-rollouts", "1", "--friction-range", "0.5", "1.05"),
    )

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("lapzero-ppo")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.params["env.friction_range"] == "(0.5, 1.05)"
    frictions = [
        m.value for m in client.get_metric_history(run.info.run_id, "race/episode_friction_mu")
    ]
    episodes = client.get_metric_history(run.info.run_id, "race/episode_collision")
    assert len(frictions) == len(episodes) >= 3  # 1 s episodes are 25 control steps
    assert len(set(frictions)) == len(frictions)
    assert all(0.5 <= mu <= 1.05 for mu in frictions)


def test_each_vec_env_worker_scans_the_map_of_its_own_track(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks"
    _write_track(tracks, "big", size_px=400)  # 20 m room
    _write_track(tracks, "small", size_px=200)  # 10 m room
    factories = [
        TrackEnvFactory(
            track_id,
            *track_files(tracks, track_id),
            seed=0,
            env_config=EnvConfig(),
            reward_config=RewardConfig(),
        )
        for track_id in ("big", "small")
    ]
    vec_env = make_track_vec_env(factories)
    try:
        observations = vec_env.reset()
    finally:
        vec_env.close()

    # Built alone, one after the other in this process, each env scans its own map.
    for factory, obs in zip(factories, observations, strict=True):
        env = factory()
        try:
            expected, _ = env.reset()
        finally:
            env.close()
        np.testing.assert_array_equal(obs, expected)
    assert observations[0][LIDAR_SLICE].mean() > observations[1][LIDAR_SLICE].mean()


def test_ppo_on_several_tracks_logs_each_track_and_saves_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracks = tmp_path / "tracks"
    _write_track(tracks, "big", size_px=400)
    _write_track(tracks, "small", size_px=200)
    bc_path = tmp_path / "bc.pt"
    torch.manual_seed(0)
    save_policy(BCPolicy(16), bc_path)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    _run(
        "train_ppo.py",
        *("--tracks", "big", "small", "--tracks-dir", str(tracks)),
        *("--bc-weights", str(bc_path), "--output", str(tmp_path / "models" / "multi.pt")),
        *("--tracking-uri", tracking_uri, "--laps", "1", "--max-episode-s", "1.0"),
        *("--total-timesteps", "128", "--n-steps", "32", "--batch-size", "16", "--n-epochs", "1"),
        *("--critic-warmup-rollouts", "1", "--checkpoint-interval", "64"),
    )

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("lapzero-ppo")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    params = run.data.params
    assert (params["track_ids"], params["n_envs"]) == ("big,small", "2")
    assert params["checkpoint_interval"] == "64"
    run_id = run.info.run_id
    pooled = client.get_metric_history(run_id, "race/episode_collision")
    per_track = [
        client.get_metric_history(run_id, f"race/{track}/episode_collision")
        for track in ("big", "small")
    ]
    assert all(per_track), "a track logged no episodes"  # 1 s episodes are 25 control steps
    assert len(pooled) == len(per_track[0]) + len(per_track[1])

    # 2 envs x 32 steps per rollout: the warm-up rollout ends at step 64, the next at 128.
    checkpoints = sorted((tmp_path / "models" / "checkpoints").iterdir())
    assert [path.name for path in checkpoints] == ["multi_step_128.pt", "multi_step_64.pt"]
    assert all(load_policy(path).hidden_size == 16 for path in checkpoints)


def test_train_ppo_rejects_tracks_combined_with_single_track_flags(tmp_path: Path) -> None:
    command = [sys.executable, "scripts/train_ppo.py", "--tracks", "big", "--map", "room.yaml"]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 2
    assert "--tracks cannot be combined" in completed.stderr


def _eval_output(*args: str) -> str:
    command = [sys.executable, "scripts/eval.py", *args]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_eval_script_scores_a_registry_alias_and_a_local_checkpoint(tmp_path: Path) -> None:
    room = _write_room(tmp_path)
    centreline = _write_circle_centreline(tmp_path / "centreline.csv")
    checkpoint = tmp_path / "policy.pt"
    torch.manual_seed(0)
    save_policy(BCPolicy(16), checkpoint)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
        tracker.register_policy(checkpoint, "lapzero-policy", {"algorithm": "bc"})  # champion
    track = ["--map", str(room), "--start", "0", "-5", "0", "--centreline", str(centreline)]
    runs = {
        "lapzero-policy@champion": ["--tracking-uri", tracking_uri],
        str(checkpoint): ["--checkpoint", str(checkpoint)],
    }
    outputs = {
        label: _eval_output(*track, *args, "--episodes", "2") for label, args in runs.items()
    }
    for label, stdout in outputs.items():
        assert f"{label} on room, 2 noise-free episodes" in stdout
        assert "seed   0" in stdout and "seed   1" in stdout
        assert "completed" in stdout.splitlines()[-1]
    # Same weights, same seeds, no exploration noise: identical results either way.
    episode_lines = [[ln for ln in out.splitlines() if "seed" in ln] for out in outputs.values()]
    assert episode_lines[0] == episode_lines[1]


def test_eval_script_exports_traces_under_friction_and_observation_noise(tmp_path: Path) -> None:
    room = _write_room(tmp_path)
    centreline = _write_circle_centreline(tmp_path / "centreline.csv")
    checkpoint = tmp_path / "policy.pt"
    torch.manual_seed(0)
    save_policy(BCPolicy(16), checkpoint)
    out = tmp_path / "eval"
    stdout = _eval_output(
        *("--map", str(room), "--centreline", str(centreline), "--checkpoint", str(checkpoint)),
        *("--episodes", "2", "--friction", "0.5", "--lidar-noise", "0.05", "--out", str(out)),
    )
    assert "on room, 2 noisy" in stdout
    (run_dir,) = (out / "room").iterdir()
    assert f"traces and summary.json: {run_dir}" in stdout
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["run"]["friction_mu"] == 0.5
    assert summary["run"]["obs_noise"]["lidar_std_m"] == 0.05
    assert summary["run"]["policy_version"] is None
    assert summary["run"]["start_pose"][:2] == [0.0, -5.0]  # the centreline's first point
    assert [episode["seed"] for episode in summary["episodes"]] == [0, 1]
    for episode in summary["episodes"]:
        lines = (run_dir / episode["trace_csv"]).read_text().splitlines()
        assert lines[0].startswith("t,x,y,theta,v_long,")
        assert len(lines) == episode["steps"] + 1


def test_eval_sweep_script_writes_one_level_per_friction_from_jittered_starts(
    tmp_path: Path,
) -> None:
    room = _write_room(tmp_path)
    centreline = _write_circle_centreline(tmp_path / "centreline.csv")
    checkpoint = tmp_path / "policy.pt"
    torch.manual_seed(0)
    save_policy(BCPolicy(16), checkpoint)
    out = tmp_path / "sweep"
    command = [
        *(sys.executable, "scripts/eval_sweep.py", "--map", str(room)),
        *("--centreline", str(centreline), "--checkpoint", str(checkpoint)),
        *("--frictions", "0.5", "1.0", "0.5", "--episodes", "2", "--out", str(out)),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert "2 episodes per level, jittered start" in completed.stdout

    (sweep_dir,) = (out / "room").iterdir()
    sweep = json.loads((sweep_dir / "sweep.json").read_text())
    assert sweep["run"]["randomize_start"] is True
    assert [level["friction_mu"] for level in sweep["levels"]] == [0.5, 1.0]
    for level in sweep["levels"]:
        assert level["episodes"] == 2
        summary = json.loads((sweep_dir / level["eval_dir"] / "summary.json").read_text())
        assert summary["run"]["friction_mu"] == level["friction_mu"]
        assert [episode["seed"] for episode in summary["episodes"]] == [0, 1]
    crashed = sum(level["crashed"] for level in sweep["levels"])
    assert len(sweep["crashes"]) == crashed
