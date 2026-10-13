"""Unit tests for lapzero/eval/sweep.py: the friction grid, sweeping levels and sweep.json.

Sweeps run on the FakeSimulator, where every episode crashes on its second control step.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from lapzero.eval.runner import EpisodeResult, EvalSummary
from lapzero.eval.sweep import (
    CrashPoint,
    FrictionLevel,
    SweepRun,
    friction_grid,
    read_sweep,
    sweep_friction,
    write_sweep,
)
from lapzero.io.telemetry import ULID_PATTERN, LapStatus
from lapzero.sim.simulator import Pose, SimState
from tests.conftest import START_POSE, FakeSimulator
from tests.unit.eval.test_export import _trace
from tests.unit.eval.test_runner import ConstantPolicy

# Friction grid


def test_the_default_grid_runs_from_0_50_to_1_05_in_twelve_levels() -> None:
    assert friction_grid(0.50, 1.05, 0.05) == (
        *(0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
        *(0.8, 0.85, 0.9, 0.95, 1.0, 1.05),
    )


def test_a_grid_from_a_value_to_itself_has_one_level() -> None:
    assert friction_grid(0.7, 0.7, 0.05) == (0.7,)


@pytest.mark.parametrize(
    ("low", "high", "step"),
    [(0.5, 1.0, 0.3), (1.0, 0.5, 0.05), (0.0, 1.0, 0.05), (0.5, 1.0, 0.0), (0.5, math.nan, 0.05)],
)
def test_invalid_grids_are_rejected(low: float, high: float, step: float) -> None:
    with pytest.raises(ValueError, match="friction grid"):
        friction_grid(low, high, step)


# Sweeping


class RecordingSimulator(FakeSimulator):
    """Records the friction in force at every reset; each episode crashes on control step 2."""

    def __init__(self) -> None:
        super().__init__()
        self.collision_at_step = 8  # physics steps
        self.reset_frictions: list[float] = []

    def reset(self, pose: Pose) -> SimState:
        self.reset_frictions.append(self.friction)
        return super().reset(pose)


def test_each_level_runs_its_episodes_at_its_friction_from_the_same_start_poses() -> None:
    sim = RecordingSimulator()
    levels = sweep_friction(
        sim,
        START_POSE,
        ConstantPolicy(throttle=1.0),
        frictions=(0.6, 0.9),
        episodes=3,
        seed=5,
        randomize_start=True,
        centreline=None,
    )
    assert [level.friction_mu for level in levels] == [0.6, 0.9]
    assert sim.reset_frictions == [0.6, 0.6, 0.6, 0.9, 0.9, 0.9]
    for level in levels:
        assert [e.seed for e in level.summary.episodes] == [5, 6, 7]
        assert all(e.status is LapStatus.CRASHED for e in level.summary.episodes)
    assert sim.reset_poses[:3] == sim.reset_poses[3:]
    assert len(set(sim.reset_poses[:3])) == 3
    assert not sim.closed  # the caller owns the simulator


@pytest.mark.parametrize("frictions", [(), (0.6, 0.6)])
def test_a_sweep_needs_distinct_frictions(frictions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="frictions"):
        sweep_friction(
            FakeSimulator(),
            START_POSE,
            ConstantPolicy(),
            frictions=frictions,
            episodes=1,
            seed=0,
            randomize_start=False,
            centreline=None,
        )


# Export


def _sweep_run(**overrides: Any) -> SweepRun:
    fields: dict[str, Any] = {
        "git_commit": "abc123-dirty",
        "policy": "lapzero-policy@champion",
        "policy_version": "5",
        "track_id": "oschersleben",
        "map_yaml": "data/tracks/oschersleben/oschersleben.yaml",
        "start_pose": (0.0, 0.0, 2.857),
        "seed": 0,
        "episodes_per_level": 2,
        "randomize_start": True,
    }
    return SweepRun(**(fields | overrides))


def _level(friction_mu: float, *, second_crashes: bool) -> FrictionLevel:
    """Two episodes: seed 0 completes in 46.561 s; seed 1 crashes or completes in 47.0 s."""
    first = EpisodeResult(0, LapStatus.COMPLETED, 46.561, 1.0, 6.5, 4, _trace([0.1, 0.0, 0.2, 0.1]))
    if second_crashes:  # _trace ends at x 2.0, y -0.5
        second = EpisodeResult(
            1, LapStatus.CRASHED, None, 0.54632, 5.0, 2, _trace([0.5, 1.1], crash=True)
        )
    else:
        second = EpisodeResult(1, LapStatus.COMPLETED, 47.0, 1.0, 6.4, 4, _trace([0.0] * 4))
    return FrictionLevel(friction_mu, EvalSummary((first, second)))


def _levels() -> list[FrictionLevel]:
    return [_level(0.5, second_crashes=True), _level(1.05, second_crashes=False)]


def test_a_sweep_writes_sweep_json_and_one_export_per_level(tmp_path: Path) -> None:
    run = _sweep_run()
    directory = write_sweep(tmp_path, run, _levels())
    assert re.match(ULID_PATTERN, run.sweep_id)
    assert directory == tmp_path / "oschersleben" / run.sweep_id
    assert sorted(path.name for path in directory.iterdir()) == ["mu_0.5", "mu_1.05", "sweep.json"]
    level_files = sorted(path.name for path in (directory / "mu_0.5").iterdir())
    assert level_files == ["seed_0.csv", "seed_1.csv", "summary.json"]
    summary = json.loads((directory / "mu_0.5" / "summary.json").read_text())
    assert summary["run"]["friction_mu"] == 0.5
    assert summary["run"]["randomize_start"] is True
    assert summary["run"]["policy_version"] == "5"


def test_sweep_json_aggregates_each_level_and_locates_every_crash(tmp_path: Path) -> None:
    report = read_sweep(write_sweep(tmp_path, _sweep_run(), _levels()))
    low, high = report.levels
    assert (low.friction_mu, low.episodes, low.completed, low.crashed) == (0.5, 2, 1, 1)
    assert (low.completion_rate, low.mean_lap_time_s, low.eval_dir) == (0.5, 46.56, "mu_0.5")
    assert (high.completed, high.crashed, high.mean_lap_time_s) == (2, 0, 46.78)
    assert report.crashes == (
        CrashPoint(friction_mu=0.5, seed=1, x=2.0, y=-0.5, lap_fraction=0.5463),
    )
    assert report.level(1.05) is high
    assert report.level(0.7) is None


def test_an_existing_sweep_directory_is_never_overwritten(tmp_path: Path) -> None:
    run = _sweep_run()
    write_sweep(tmp_path, run, _levels())
    with pytest.raises(FileExistsError):
        write_sweep(tmp_path, run, _levels())


def test_levels_sharing_a_friction_write_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct frictions"):
        write_sweep(tmp_path, _sweep_run(), [_level(0.5, second_crashes=True)] * 2)
    assert not any(tmp_path.iterdir())


def test_an_episode_without_a_trace_writes_nothing(tmp_path: Path) -> None:
    untraced = EpisodeResult(0, LapStatus.TIMEOUT, None, 0.2, 1.0, 3)
    with pytest.raises(ValueError, match="trace"):
        write_sweep(tmp_path, _sweep_run(), [FrictionLevel(0.5, EvalSummary((untraced,)))])
    assert not any(tmp_path.iterdir())
