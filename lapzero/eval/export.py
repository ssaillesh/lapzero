"""Evaluation exports for trajectory analysis: one CSV trace per episode and a summary.json.

A run is written to <root>/<track_id>/<eval_id>/: summary.json (what was evaluated, under which
conditions, per-episode results and aggregates) and seed_<seed>.csv for every episode (the
EpisodeTrace columns at CTRL_HZ). These are analysis files, not lap telemetry (contract
Section 5): they carry values the contract does not define, such as the cross-track error, and
the backend never reads them.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from lapzero.contract import (
    CONTRACT_VERSION,
    DECIMALS_ACTION,
    DECIMALS_ANGLE,
    DECIMALS_POSITION,
    DECIMALS_PROGRESS,
    DECIMALS_SPEED,
    DECIMALS_TIME,
)
from lapzero.eval.perturb import ObservationNoise
from lapzero.eval.runner import EpisodeResult, EpisodeTrace, EvalSummary
from lapzero.io.telemetry import TRACK_ID_PATTERN, ULID_PATTERN, LapStatus
from lapzero.record.ulid import new_ulid

logger = logging.getLogger(__name__)

EVAL_DIR: Final = "data/eval"
"""Default root for evaluation exports (gitignored with the rest of data/)."""
_SUMMARY_FILE = "summary.json"
_CSV_DECIMALS: Final = {
    "t": DECIMALS_TIME,
    "x": DECIMALS_POSITION,
    "y": DECIMALS_POSITION,
    "theta": DECIMALS_ANGLE,
    "v_long": DECIMALS_SPEED,
    "steer": DECIMALS_ACTION,
    "throttle": DECIMALS_ACTION,
    "steer_rad": DECIMALS_ANGLE,
    "v_target": DECIMALS_SPEED,
    "progress": DECIMALS_PROGRESS,
    "cte_m": DECIMALS_POSITION,
}
"""Decimal places per CSV column: lap telemetry precision (contract Section 5.2); cte_m is a
position. collision is written as 0 or 1."""


def _utc_now() -> datetime:
    """Current UTC time to the second."""
    return datetime.now(UTC).replace(microsecond=0)


class _Strict(BaseModel):
    """Base for export models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class EvalRun(_Strict):
    """What was evaluated, and under which conditions.

    Attributes:
        eval_id: ULID naming the run directory; new by default.
        created_at: UTC time the run was described; now by default.
        contract_version: State space contract version the policy ran under.
        git_commit: Commit of the evaluating code, "-dirty" if it had uncommitted changes.
        policy: Registry "<model>@<alias>" or checkpoint path.
        policy_version: Registry model version; None for a checkpoint.
        track_id: Track the run drove on; also the directory name.
        map_yaml: Map file the simulator loaded.
        start_pose: Fixed start pose x (metres), y (metres), theta (rad).
        seed: Seed of the first episode; episode i used seed + i.
        randomize_start: Start pose jitter was on (a diagnostic, not the contract's eval setup).
        friction_mu: Tyre-road friction coefficient the simulator ran with.
        obs_noise: Sensor noise added to what the policy saw.
    """

    eval_id: str = Field(default_factory=new_ulid, pattern=ULID_PATTERN)
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    contract_version: str = CONTRACT_VERSION
    git_commit: str
    policy: str
    policy_version: str | None
    track_id: str = Field(pattern=TRACK_ID_PATTERN)
    map_yaml: str
    start_pose: tuple[float, float, float]
    seed: int
    randomize_start: bool
    friction_mu: float = Field(gt=0.0)
    obs_noise: ObservationNoise


class EpisodeRecord(_Strict):
    """One episode's result and trace statistics, as written to summary.json.

    Speeds are m/s, distances metres, lap_time_s seconds; trace_csv is the trace file's name.
    """

    seed: int
    status: LapStatus
    lap_time_s: float | None
    lap_fraction: float
    steps: int
    mean_speed_mps: float
    max_speed_mps: float
    mean_abs_cte_m: float
    max_abs_cte_m: float
    rms_cte_m: float
    trace_csv: str


class EvalReport(_Strict):
    """Contents of summary.json: the run, its aggregates and every episode.

    mean_abs_cte_m and max_abs_cte_m are over every control step of every episode, metres.
    """

    run: EvalRun
    completion_rate: float
    crash_rate: float
    mean_lap_time_s: float | None
    mean_abs_cte_m: float
    max_abs_cte_m: float
    episodes: tuple[EpisodeRecord, ...]


def write_trace_csv(path: Path, trace: EpisodeTrace) -> None:
    """Write a trace as CSV: a header of column names, then one row per control step.

    Args:
        path: CSV file to write.
        trace: Episode trace; see EpisodeTrace for the columns and their units.
    """
    names = [column.name for column in fields(trace)]
    formats = [f"%.{_CSV_DECIMALS[name]}f" if name in _CSV_DECIMALS else "%d" for name in names]
    data = np.column_stack([getattr(trace, name) for name in names])
    np.savetxt(path, data, fmt=formats, delimiter=",", header=",".join(names), comments="")


def _record(episode: EpisodeResult, trace: EpisodeTrace, trace_csv: str) -> EpisodeRecord:
    """Summarize one episode and its trace at telemetry precision."""
    lap_time = episode.lap_time_s
    return EpisodeRecord(
        seed=episode.seed,
        status=episode.status,
        lap_time_s=round(lap_time, DECIMALS_TIME) if lap_time is not None else None,
        lap_fraction=round(episode.lap_fraction, DECIMALS_PROGRESS),
        steps=episode.steps,
        mean_speed_mps=round(episode.mean_speed_mps, DECIMALS_SPEED),
        max_speed_mps=round(trace.max_speed_mps, DECIMALS_SPEED),
        mean_abs_cte_m=round(trace.mean_abs_cte_m, DECIMALS_POSITION),
        max_abs_cte_m=round(trace.max_abs_cte_m, DECIMALS_POSITION),
        rms_cte_m=round(trace.rms_cte_m, DECIMALS_POSITION),
        trace_csv=trace_csv,
    )


def write_eval(root: Path, run: EvalRun, summary: EvalSummary) -> Path:
    """Write summary.json and one CSV trace per episode to root/<track_id>/<eval_id>/.

    Args:
        root: Export root, normally EVAL_DIR.
        run: What was evaluated and under which conditions.
        summary: The evaluation's episodes, each with its trace.

    Returns:
        The run directory.

    Raises:
        ValueError: If an episode has no trace or two episodes share a seed; nothing is written.
        FileExistsError: If the run directory already exists.
    """
    directory = root / run.track_id / run.eval_id
    write_eval_to(directory, run, summary)
    return directory


def write_eval_to(directory: Path, run: EvalRun, summary: EvalSummary) -> None:
    """Write summary.json and one CSV trace per episode to `directory`, which must not exist.

    Args:
        directory: Run directory to create.
        run: What was evaluated and under which conditions.
        summary: The evaluation's episodes, each with its trace.

    Raises:
        ValueError: If an episode has no trace or two episodes share a seed; nothing is written.
        FileExistsError: If `directory` already exists.
    """
    traced = [(e, e.trace) for e in summary.episodes if e.trace is not None]
    if len(traced) != len(summary.episodes):
        missing = [e.seed for e in summary.episodes if e.trace is None]
        raise ValueError(f"episodes with seeds {missing} have no trace to export")
    seeds = [episode.seed for episode in summary.episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"episode seeds must be unique to name their CSV files, got {seeds}")
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for episode, trace in traced:
        name = f"seed_{episode.seed}.csv"
        write_trace_csv(directory / name, trace)
        records.append(_record(episode, trace, name))
    all_cte = np.abs(np.concatenate([trace.cte_m for _, trace in traced]))
    mean_lap_time = summary.mean_lap_time_s
    report = EvalReport(
        run=run,
        completion_rate=summary.completion_rate,
        crash_rate=summary.crash_rate,
        mean_lap_time_s=round(mean_lap_time, DECIMALS_TIME) if mean_lap_time is not None else None,
        mean_abs_cte_m=round(float(all_cte.mean()), DECIMALS_POSITION),
        max_abs_cte_m=round(float(all_cte.max()), DECIMALS_POSITION),
        episodes=tuple(records),
    )
    (directory / _SUMMARY_FILE).write_text(report.model_dump_json(indent=2) + "\n")
    logger.info("wrote %s and %d episode traces to %s", _SUMMARY_FILE, len(records), directory)
