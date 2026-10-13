"""Unit tests for lapzero/training/callbacks.py: SB3 metric forwarding, episode and lap metrics
(pooled and per track), and periodic actor checkpoints.

The callback is stepped by hand with SB3-style locals; no model is trained here.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lapzero.training.callbacks import (
    PolicyCheckpointCallback,
    RaceMetricsCallback,
    TrackerOutputFormat,
)
from lapzero.training.policy import BCPolicy, load_policy
from tests.unit.training.test_bc import FakeTracker


class FakeModel:
    """Stands in for the SB3 model, which a callback reads num_timesteps from."""

    num_timesteps = 0


def _callback(tracker: FakeTracker) -> RaceMetricsCallback:
    callback = RaceMetricsCallback(tracker)
    callback.model = FakeModel()
    return callback


def _step(
    callback: RaceMetricsCallback, timestep: int, infos: list[dict[str, Any]], dones: list[bool]
) -> bool:
    """Run the callback as SB3 does after one vectorized env step."""
    callback.model.num_timesteps = timestep
    callback.update_locals({"infos": infos, "dones": np.array(dones)})
    return callback.on_step()


def _info(t: float, lap_count: int = 0, collision: bool = False, **extra: Any) -> dict[str, Any]:
    return {"t": t, "lap_count": lap_count, "collision": collision, **extra}


def _values(tracker: FakeTracker, key: str) -> list[float]:
    return [value for name, value, _ in tracker.metrics if name == key]


def test_numeric_sb3_values_are_logged_at_the_dump_step() -> None:
    tracker = FakeTracker()
    values = {"train/value_loss": 0.5, "time/iterations": 3, "train/std": np.float32(0.25)}
    TrackerOutputFormat(tracker).write(values, {}, step=64)
    assert tracker.metrics == [
        ("time/iterations", 3.0, 64),
        ("train/std", 0.25, 64),
        ("train/value_loss", 0.5, 64),
    ]


def test_non_finite_and_non_numeric_sb3_values_are_skipped() -> None:
    tracker = FakeTracker()
    values = {
        "train/explained_variance": float("nan"),
        "train/loss": float("inf"),
        "note": "text",
        "missing": None,
        "train/approx_kl": 0.01,
    }
    TrackerOutputFormat(tracker).write(values, {}, step=1)
    assert tracker.metrics == [("train/approx_kl", 0.01, 1)]


def test_closing_the_output_format_logs_nothing() -> None:
    tracker = FakeTracker()
    TrackerOutputFormat(tracker).close()
    assert tracker.metrics == []


def test_lap_times_are_measured_from_the_end_of_the_previous_lap() -> None:
    tracker = FakeTracker()
    callback = _callback(tracker)
    _step(callback, 1, [_info(0.04)], [False])
    _step(callback, 2, [_info(10.0, lap_count=1)], [False])
    _step(callback, 3, [_info(10.04, lap_count=1)], [False])
    _step(callback, 4, [_info(18.5, lap_count=2)], [False])
    assert _values(tracker, "race/lap_time") == [10.0, 8.5]
    assert tracker.steps("race/lap_time") == [2, 4]


def test_a_finished_episode_logs_its_monitor_stats_and_collision() -> None:
    tracker = FakeTracker()
    callback = _callback(tracker)
    episode = {"r": 12.5, "l": 300, "t": 1.0}
    _step(callback, 300, [_info(12.0, collision=True, episode=episode)], [True])
    assert tracker.metrics == [
        ("race/episode_reward", 12.5, 300),
        ("race/episode_length", 300.0, 300),
        ("race/episode_collision", 1.0, 300),
    ]


def test_a_finished_episode_without_monitor_stats_logs_only_the_collision_flag() -> None:
    tracker = FakeTracker()
    callback = _callback(tracker)
    _step(callback, 5, [_info(0.2)], [True])
    assert tracker.metrics == [("race/episode_collision", 0.0, 5)]


def test_a_lap_finishing_on_the_last_step_is_logged_and_timing_restarts_after_reset() -> None:
    tracker = FakeTracker()
    callback = _callback(tracker)
    _step(callback, 1, [_info(10.0, lap_count=1)], [False])
    _step(callback, 2, [_info(19.0, lap_count=2)], [True])
    _step(callback, 3, [_info(0.04)], [False])  # new episode: t and lap_count restart
    _step(callback, 4, [_info(9.0, lap_count=1)], [False])
    assert _values(tracker, "race/lap_time") == [10.0, 9.0, 9.0]


def test_each_env_is_timed_separately() -> None:
    tracker = FakeTracker()
    callback = _callback(tracker)
    _step(callback, 2, [_info(5.0), _info(7.0, lap_count=1)], [False, False])
    _step(callback, 4, [_info(10.0, lap_count=1), _info(7.04, lap_count=1)], [False, True])
    _step(callback, 6, [_info(10.04, lap_count=1), _info(6.0, lap_count=1)], [False, False])
    assert _values(tracker, "race/lap_time") == [7.0, 10.0, 6.0]


def test_the_callback_never_stops_training() -> None:
    callback = _callback(FakeTracker())
    assert _step(callback, 1, [_info(0.04, collision=True)], [True]) is True


class FakeVecEnv:
    """Stands in for the SB3 VecEnv; `frictions[i]` is env i's current episode friction mu.

    With `frictions` None the envs have no friction attribute, like a plain Gymnasium env.
    """

    def __init__(self, frictions: list[float] | None) -> None:
        self.frictions = frictions

    def has_attr(self, attr_name: str) -> bool:
        return attr_name == "friction" and self.frictions is not None

    def get_attr(self, attr_name: str, indices: list[int] | None = None) -> list[float]:
        assert attr_name == "friction" and self.frictions is not None
        return [self.frictions[i] for i in indices or range(len(self.frictions))]


class FakeModelWithEnv(FakeModel):
    def __init__(self, env: FakeVecEnv) -> None:
        self.env = env

    def get_env(self) -> FakeVecEnv:
        return self.env


def _started_callback(tracker: FakeTracker, env: FakeVecEnv) -> RaceMetricsCallback:
    """A callback on `env`, after SB3 has reset the envs and started training."""
    callback = RaceMetricsCallback(tracker)
    callback.model = FakeModelWithEnv(env)
    callback.on_training_start({}, {})
    return callback


def test_each_episode_logs_the_friction_it_started_with() -> None:
    tracker = FakeTracker()
    env = FakeVecEnv([0.6, 0.9])
    callback = _started_callback(tracker, env)
    env.frictions = [0.7, 0.9]  # env 0 finished and the VecEnv already reset it with a new mu
    _step(callback, 2, [_info(0.2), _info(0.2)], [True, False])
    env.frictions = [0.55, 0.8]  # both finished and were reset
    _step(callback, 4, [_info(0.3), _info(0.5)], [True, True])
    assert _values(tracker, "race/episode_friction_mu") == [0.6, 0.7, 0.9]
    assert tracker.steps("race/episode_friction_mu") == [2, 4, 4]


def test_no_friction_is_logged_when_the_envs_have_none() -> None:
    tracker = FakeTracker()
    callback = _started_callback(tracker, FakeVecEnv(None))
    _step(callback, 5, [_info(0.2)], [True])
    assert tracker.metrics == [("race/episode_collision", 0.0, 5)]


class FakeTrackVecEnv(FakeVecEnv):
    """A FakeVecEnv that also reports its number of envs, which track ids are checked against."""

    def __init__(self, num_envs: int, frictions: list[float] | None = None) -> None:
        super().__init__(frictions)
        self.num_envs = num_envs


def _track_callback(
    tracker: FakeTracker, track_ids: list[str], env: FakeTrackVecEnv
) -> RaceMetricsCallback:
    callback = RaceMetricsCallback(tracker, track_ids)
    callback.model = FakeModelWithEnv(env)
    callback.on_training_start({}, {})
    return callback


def test_with_track_ids_every_metric_is_also_logged_under_its_track() -> None:
    tracker = FakeTracker()
    callback = _track_callback(tracker, ["spielberg", "moscowraceway"], FakeTrackVecEnv(2))
    episode = {"r": 12.5, "l": 300, "t": 1.0}
    infos = [_info(12.0, collision=True, episode=episode), _info(9.5, lap_count=1)]
    _step(callback, 300, infos, [True, False])
    assert tracker.metrics == [
        ("race/episode_reward", 12.5, 300),
        ("race/spielberg/episode_reward", 12.5, 300),
        ("race/episode_length", 300.0, 300),
        ("race/spielberg/episode_length", 300.0, 300),
        ("race/episode_collision", 1.0, 300),
        ("race/spielberg/episode_collision", 1.0, 300),
        ("race/lap_time", 9.5, 300),
        ("race/moscowraceway/lap_time", 9.5, 300),
    ]


def test_each_tracks_friction_is_logged_under_that_track() -> None:
    tracker = FakeTracker()
    env = FakeTrackVecEnv(2, frictions=[0.6, 0.9])
    callback = _track_callback(tracker, ["track_a", "track_b"], env)
    env.frictions = [0.6, 0.7]  # env 1 finished and was reset with a new mu
    _step(callback, 2, [_info(0.2), _info(0.2)], [False, True])
    assert _values(tracker, "race/track_b/episode_friction_mu") == [0.9]
    assert _values(tracker, "race/track_a/episode_friction_mu") == []


def test_track_ids_must_name_one_track_per_env() -> None:
    callback = RaceMetricsCallback(FakeTracker(), ["track_a"])
    callback.model = FakeModelWithEnv(FakeTrackVecEnv(2))
    with pytest.raises(ValueError, match="1 track_ids for 2 envs"):
        callback.on_training_start({}, {})


class ActorSource:
    """Hands out a small BCPolicy and counts how often a checkpoint asked for one."""

    def __init__(self) -> None:
        self.calls = 0
        torch.manual_seed(0)
        self.policy = BCPolicy(16)

    def __call__(self) -> BCPolicy:
        self.calls += 1
        return self.policy


def _run_checkpoints(
    directory: Path, interval: int, timesteps: list[int], actor: ActorSource
) -> PolicyCheckpointCallback:
    """Step a checkpoint callback through `timesteps`, as SB3 reports them after each env step."""
    callback = PolicyCheckpointCallback(interval, directory, "ppo_multitrack_v1", actor)
    callback.model = FakeModel()
    for timestep in timesteps:
        callback.model.num_timesteps = timestep
        assert callback.on_step() is True
    return callback


def test_the_actor_is_saved_each_time_the_timestep_count_reaches_a_multiple_of_the_interval(
    tmp_path: Path,
) -> None:
    actor = ActorSource()
    directory = tmp_path / "models" / "checkpoints"
    callback = _run_checkpoints(directory, 4, [2, 4, 6, 8, 10], actor)  # 2 envs per step
    names = ["ppo_multitrack_v1_step_4.pt", "ppo_multitrack_v1_step_8.pt"]
    assert callback.saved == [directory / name for name in names]
    assert sorted(path.name for path in directory.iterdir()) == names
    assert actor.calls == 2
    loaded = load_policy(directory / names[0])
    for key, value in actor.policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_a_step_past_several_multiples_saves_once_under_the_last_one(tmp_path: Path) -> None:
    callback = _run_checkpoints(tmp_path, 2, [3, 6, 7], ActorSource())  # 3 envs per step
    assert [path.name for path in callback.saved] == [
        "ppo_multitrack_v1_step_2.pt",
        "ppo_multitrack_v1_step_6.pt",
    ]


def test_nothing_is_saved_before_the_first_interval(tmp_path: Path) -> None:
    actor = ActorSource()
    callback = _run_checkpoints(tmp_path / "checkpoints", 100, [2, 50, 98], actor)
    assert callback.saved == []
    assert actor.calls == 0
    assert not (tmp_path / "checkpoints").exists()


def test_a_checkpoint_interval_below_one_timestep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="interval"):
        PolicyCheckpointCallback(0, tmp_path, "ppo", ActorSource())
