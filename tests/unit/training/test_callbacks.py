"""Unit tests for lapzero/training/callbacks.py: SB3 metric forwarding, episode and lap metrics.

The callback is stepped by hand with SB3-style locals; no model is trained here.
"""

from typing import Any

import numpy as np

from lapzero.training.callbacks import RaceMetricsCallback, TrackerOutputFormat
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
