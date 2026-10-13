"""Unit tests for lapzero/training/vec_env.py: track file lookup and the per-track env factory.

make_env is replaced by a LapZeroEnv on the FakeSimulator, so F1TENTH never starts; the
SubprocVecEnv workers on real maps run in tests/integration/test_ppo_pipeline.py.
"""

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor

from lapzero.sim.env import EnvConfig, LapZeroEnv, RewardFn
from lapzero.sim.reward import RacingReward, RewardConfig
from lapzero.sim.simulator import Pose
from lapzero.sim.track import Centreline
from lapzero.training import vec_env
from lapzero.training.vec_env import TrackEnvFactory, make_track_vec_env, track_files
from tests.conftest import FakeSimulator

ENV_CONFIG = EnvConfig(max_episode_s=0.2)  # 5 control steps per episode


def _write_track(tracks_dir: Path, track_id: str) -> Path:
    """Write <track_id>/<track_id>.yaml and a square centreline starting at (-5, -5) heading +x."""
    directory = tracks_dir / track_id
    directory.mkdir(parents=True)
    (directory / f"{track_id}.yaml").write_text(f"image: {track_id}.png\n")
    rows = "-5.0, -5.0\n5.0, -5.0\n5.0, 5.0\n-5.0, 5.0\n"
    (directory / "centreline.csv").write_text(f"# x_m, y_m\n{rows}")
    return directory


@pytest.fixture
def make_env_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace vec_env.make_env with a FakeSimulator env and record the arguments of each call."""
    calls: list[dict[str, Any]] = []

    def fake_make_env(
        map_yaml: Path,
        start_pose: Pose,
        *,
        seed: int,
        reward_fn: RewardFn | None = None,
        config: EnvConfig | None = None,
        centreline: Centreline | None = None,
    ) -> LapZeroEnv:
        calls.append(
            {"map_yaml": map_yaml, "start_pose": start_pose, "seed": seed, "config": config}
        )
        return LapZeroEnv(FakeSimulator(), start_pose, config=config, centreline=centreline)

    monkeypatch.setattr(vec_env, "make_env", fake_make_env)
    return calls


def _factory(tracks_dir: Path, track_id: str = "track_a", **overrides: Any) -> TrackEnvFactory:
    """Write track `track_id` under `tracks_dir` and return a factory for it."""
    _write_track(tracks_dir, track_id)
    fields: dict[str, Any] = {
        "seed": 3,
        "env_config": ENV_CONFIG,
        "reward_config": RewardConfig(collision_penalty=7.0),
    }
    return TrackEnvFactory(track_id, *track_files(tracks_dir, track_id), **(fields | overrides))


def test_track_files_are_the_map_yaml_and_centreline_in_the_tracks_own_directory(
    tmp_path: Path,
) -> None:
    directory = _write_track(tmp_path, "spielberg")
    assert track_files(tmp_path, "spielberg") == (
        directory / "spielberg.yaml",
        directory / "centreline.csv",
    )


@pytest.mark.parametrize("missing", ["track_a.yaml", "centreline.csv"])
def test_a_missing_track_file_is_named_in_the_error(tmp_path: Path, missing: str) -> None:
    (_write_track(tmp_path, "track_a") / missing).unlink()
    with pytest.raises(FileNotFoundError, match=rf"'track_a'.*{missing}"):
        track_files(tmp_path, "track_a")


def test_an_unknown_track_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nowhere"):
        track_files(tmp_path, "nowhere")


def test_the_factory_builds_a_monitored_racing_env_starting_on_the_centreline(
    tmp_path: Path, make_env_calls: list[dict[str, Any]]
) -> None:
    factory = _factory(tmp_path)
    env = factory()
    assert isinstance(env, Monitor)
    assert isinstance(env.env, RacingReward)
    assert env.env.config == RewardConfig(collision_penalty=7.0)
    assert isinstance(env.unwrapped, LapZeroEnv)
    (call,) = make_env_calls
    assert call["map_yaml"] == tmp_path / "track_a" / "track_a.yaml"
    assert call["seed"] == 3
    assert call["config"] == ENV_CONFIG
    # First centreline point, facing the second: (-5, -5) towards (5, -5) is +x.
    assert call["start_pose"] == Pose(-5.0, -5.0, 0.0)


def test_an_explicit_start_pose_replaces_the_centreline_start(
    tmp_path: Path, make_env_calls: list[dict[str, Any]]
) -> None:
    start = Pose(1.0, 2.0, math.pi / 2)
    _factory(tmp_path, start_pose=start)()
    assert make_env_calls[0]["start_pose"] == start


def test_the_factory_env_reports_monitor_episode_stats(
    tmp_path: Path, make_env_calls: list[dict[str, Any]]
) -> None:
    env = _factory(tmp_path)()
    env.reset(seed=0)
    done, info = False, {}
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        done = terminated or truncated
    assert info["episode"]["l"] == 5


def test_the_factory_survives_pickling_into_a_worker_process(tmp_path: Path) -> None:
    factory = _factory(tmp_path, start_pose=Pose(1.0, 2.0, 0.5))
    assert pickle.loads(pickle.dumps(factory)) == factory


def test_a_vec_env_needs_at_least_one_track() -> None:
    with pytest.raises(ValueError, match="at least one track"):
        make_track_vec_env([])
