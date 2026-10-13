"""Unit tests for lapzero/eval/export.py: per-episode CSV traces and the run's summary.json."""

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from lapzero.contract import CONTRACT_VERSION
from lapzero.eval.export import EvalReport, EvalRun, write_eval, write_trace_csv
from lapzero.eval.perturb import ObservationNoise
from lapzero.eval.runner import EpisodeResult, EpisodeTrace, EvalSummary
from lapzero.io.telemetry import ULID_PATTERN, LapStatus

HEADER = "t,x,y,theta,v_long,steer,throttle,steer_rad,v_target,progress,cte_m,collision"


def _trace(cte_m: list[float], *, crash: bool = False) -> EpisodeTrace:
    rows = len(cte_m)
    collision = np.zeros(rows, dtype=bool)
    collision[-1] = crash
    return EpisodeTrace(
        t=np.arange(1, rows + 1) * 0.04,
        x=np.linspace(1.23456, 2.0, rows),
        y=np.full(rows, -0.5),
        theta=np.full(rows, 0.123456),
        v_long=np.linspace(1.0, 3.0, rows),
        steer=np.full(rows, -0.25),
        throttle=np.full(rows, 0.98765),
        steer_rad=np.full(rows, -0.104725),
        v_target=np.linspace(1.6, 3.6, rows),
        progress=np.linspace(0.01, 0.02, rows),
        cte_m=np.array(cte_m),
        collision=collision,
    )


def _run(**overrides: Any) -> EvalRun:
    fields: dict[str, Any] = {
        "git_commit": "abc123-dirty",
        "policy": "lapzero-policy@champion",
        "policy_version": "5",
        "track_id": "oschersleben",
        "map_yaml": "data/tracks/oschersleben/oschersleben.yaml",
        "start_pose": (0.0, 0.0, 2.857),
        "seed": 3,
        "randomize_start": False,
        "friction_mu": 0.7,
        "obs_noise": ObservationNoise(lidar_std_m=0.05),
    }
    return EvalRun(**(fields | overrides))


def _summary() -> EvalSummary:
    completed = _trace([0.1, -0.3, 0.2, 0.0])
    crashed = _trace([0.5, 1.1], crash=True)
    return EvalSummary(
        (
            EpisodeResult(3, LapStatus.COMPLETED, 30.12, 1.0, 6.5, 4, completed),
            EpisodeResult(4, LapStatus.CRASHED, None, 0.42, 5.0, 2, crashed),
        )
    )


def test_a_run_writes_summary_json_and_one_csv_per_episode(tmp_path: Path) -> None:
    run = _run()
    directory = write_eval(tmp_path, run, _summary())
    assert directory == tmp_path / "oschersleben" / run.eval_id
    assert sorted(p.name for p in directory.iterdir()) == [
        "seed_3.csv",
        "seed_4.csv",
        "summary.json",
    ]


def test_the_csv_has_a_header_then_one_row_per_step_at_contract_precision(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    write_trace_csv(path, _trace([0.12345, -1.0], crash=True))
    lines = path.read_text().splitlines()
    assert lines[0] == HEADER
    assert len(lines) == 3
    assert lines[1] == "0.04,1.235,-0.500,0.1235,1.000,-0.250,0.988,-0.1047,1.600,0.0100,0.123,0"
    assert lines[2].endswith(",-1.000,1")  # cte_m, then the crash
    np.testing.assert_allclose(np.loadtxt(path, delimiter=",", skiprows=1)[:, 10], [0.123, -1.0])


def test_the_summary_records_the_run_its_episodes_and_the_cross_track_error(
    tmp_path: Path,
) -> None:
    run = _run()
    report = EvalReport.model_validate_json(
        (write_eval(tmp_path, run, _summary()) / "summary.json").read_text()
    )
    assert report.run == run
    assert report.run.obs_noise == ObservationNoise(lidar_std_m=0.05)
    assert (report.completion_rate, report.crash_rate) == (0.5, 0.5)
    assert report.mean_lap_time_s == pytest.approx(30.12)
    assert report.mean_abs_cte_m == pytest.approx(2.2 / 6, abs=1e-3)  # over all 6 steps
    assert report.max_abs_cte_m == pytest.approx(1.1)
    completed, crashed = report.episodes
    assert (completed.seed, completed.status, completed.trace_csv) == (3, "completed", "seed_3.csv")
    assert completed.lap_time_s == pytest.approx(30.12)
    assert completed.max_abs_cte_m == pytest.approx(0.3)
    assert completed.rms_cte_m == pytest.approx(np.sqrt(0.14 / 4), abs=1e-3)
    assert completed.max_speed_mps == pytest.approx(3.0)
    assert (crashed.status, crashed.lap_time_s, crashed.lap_fraction) == ("crashed", None, 0.42)


def test_the_summary_is_plain_json_with_the_contract_version(tmp_path: Path) -> None:
    summary = json.loads((write_eval(tmp_path, _run(), _summary()) / "summary.json").read_text())
    assert summary["run"]["contract_version"] == CONTRACT_VERSION
    assert summary["run"]["start_pose"] == [0.0, 0.0, 2.857]
    assert summary["run"]["obs_noise"] == {
        "lidar_std_m": 0.05,
        "speed_std_mps": 0.0,
        "yaw_rate_std_radps": 0.0,
    }


def test_a_run_gets_a_unique_ulid_and_a_utc_timestamp_by_default() -> None:
    first, second = _run(), _run()
    assert re.fullmatch(ULID_PATTERN, first.eval_id)
    assert first.eval_id != second.eval_id
    assert first.created_at.utcoffset() == timedelta(0)


def test_an_episode_without_a_trace_writes_nothing(tmp_path: Path) -> None:
    summary = EvalSummary((EpisodeResult(0, LapStatus.STUCK, None, 0.0, 0.0, 75),))
    with pytest.raises(ValueError, match="trace"):
        write_eval(tmp_path, _run(), summary)
    assert not any(tmp_path.iterdir())


def test_episodes_sharing_a_seed_write_nothing(tmp_path: Path) -> None:
    episode = _summary().episodes[0]
    with pytest.raises(ValueError, match="unique"):
        write_eval(tmp_path, _run(), EvalSummary((episode, episode)))
    assert not any(tmp_path.iterdir())


def test_an_existing_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    run = _run()
    write_eval(tmp_path, run, _summary())
    with pytest.raises(FileExistsError):
        write_eval(tmp_path, run, _summary())


@pytest.mark.parametrize(
    "overrides",
    [
        {"track_id": "../escape"},
        {"friction_mu": 0.0},
        {"start_pose": (0.0, float("nan"), 0.0)},
        {"eval_id": "not-a-ulid"},
        {"unknown_field": 1},
    ],
)
def test_invalid_run_info_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _run(**overrides)
