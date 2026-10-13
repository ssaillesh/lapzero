"""Friction sweep: one policy evaluated at each level of a tyre-road friction grid.

One simulator runs the whole sweep. Each level drives a LapZeroEnv whose friction_range is
(mu, mu), so the friction is set on every reset by the same code as training's domain
randomization, and runs one-lap episodes seeded seed + i (lapzero.eval.runner). Every level uses
the same seeds, so with randomize_start all levels, and any two policies swept with the same
settings, start from the same jittered poses.

A sweep is written to <root>/<track_id>/<sweep_id>/: sweep.json (what was swept, each level's
completion and mean lap time, and where every crash happened) and one mu_<mu>/ directory per
level with its summary.json and per-episode CSV traces (lapzero.eval.export). The release gate
(lapzero.eval.gate) compares two sweep.json reports.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from torch import nn

from lapzero.contract import CONTRACT_VERSION, DECIMALS_POSITION, DECIMALS_PROGRESS, DECIMALS_TIME
from lapzero.eval.export import EvalRun, write_eval_to
from lapzero.eval.perturb import ObservationNoise
from lapzero.eval.runner import EvalSummary, evaluate_policy
from lapzero.io.telemetry import TRACK_ID_PATTERN, ULID_PATTERN, LapStatus
from lapzero.record.ulid import new_ulid
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.simulator import Pose, Simulator
from lapzero.sim.track import Centreline

logger = logging.getLogger(__name__)

FRICTION_SWEEP_DIR: Final = "data/eval/friction_sweep"
"""Default root for sweep exports (gitignored with the rest of data/)."""
SWEEP_FILE: Final = "sweep.json"
_MU_DECIMALS: Final = 4


def friction_grid(low: float, high: float, step: float) -> tuple[float, ...]:
    """Return the friction levels low, low + step, ..., high, rounded to 4 decimals.

    Args:
        low: Lowest friction coefficient mu (dimensionless).
        high: Highest friction coefficient mu, reached in a whole number of steps from low.
        step: Spacing between levels.

    Returns:
        The levels in increasing order.

    Raises:
        ValueError: If a value is not finite and > 0, high < low, or (high - low) is not a whole
            number of steps.
    """
    for name, value in (("low", low), ("high", high), ("step", step)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"friction grid {name} must be finite and > 0, got {value}")
    steps = (high - low) / step
    if steps < 0.0 or not math.isclose(steps, round(steps), abs_tol=1e-6):
        raise ValueError(f"friction grid {low}..{high} is not a whole number of {step} steps")
    return tuple(round(low + i * step, _MU_DECIMALS) for i in range(round(steps) + 1))


@dataclass(frozen=True, slots=True)
class FrictionLevel:
    """One level of a sweep: the friction mu it ran with and every episode's result."""

    friction_mu: float
    summary: EvalSummary


def sweep_friction(
    simulator: Simulator,
    start_pose: Pose,
    policy: nn.Module,
    *,
    frictions: Sequence[float],
    episodes: int,
    seed: int,
    randomize_start: bool,
    centreline: Centreline | None,
) -> tuple[FrictionLevel, ...]:
    """Evaluate `policy` at every friction level, `episodes` episodes seeded seed + i each.

    The simulator stays open; the caller closes it.

    Args:
        simulator: Physics backend shared by every level.
        start_pose: Fixed start pose of the track.
        policy: Deterministic policy, see lapzero.eval.runner.run_episode.
        frictions: Friction coefficients mu (dimensionless), each finite and > 0.
        episodes: Episodes per level, at least 1.
        seed: Seed of every level's first episode; logged.
        randomize_start: Jitter the start pose per episode seed (a diagnostic, not the
            contract's eval setup).
        centreline: Track centreline for progress and cross-track error; optional.

    Returns:
        One FrictionLevel per friction, in the given order.

    Raises:
        ValueError: If `frictions` is empty or repeats a value, or a friction or `episodes` is
            invalid.
    """
    if not frictions or len(set(frictions)) != len(frictions):
        raise ValueError(f"frictions must be non-empty and distinct, got {list(frictions)}")
    logger.info(
        "friction sweep over mu %s: %d episodes per level from seed=%d, randomize_start=%s",
        list(frictions),
        episodes,
        seed,
        randomize_start,
    )
    levels = []
    for mu in frictions:
        config = EnvConfig(randomize_start=randomize_start, friction_range=(mu, mu))
        # Not closed: closing the env would close the shared simulator.
        env = LapZeroEnv(simulator, start_pose, config=config, centreline=centreline)
        summary = evaluate_policy(env, policy, episodes=episodes, seed=seed)
        logger.info(
            "mu %.4g: completed %.0f%%, mean lap time %s s",
            mu,
            100.0 * summary.completion_rate,
            summary.mean_lap_time_s,
        )
        levels.append(FrictionLevel(mu, summary))
    return tuple(levels)


def _utc_now() -> datetime:
    """Current UTC time to the second."""
    return datetime.now(UTC).replace(microsecond=0)


class _Strict(BaseModel):
    """Base for sweep models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class SweepRun(_Strict):
    """What was swept, and under which conditions.

    Attributes:
        sweep_id: ULID naming the sweep directory; new by default.
        created_at: UTC time the sweep was described; now by default.
        contract_version: State space contract version the policy ran under.
        git_commit: Commit of the evaluating code, "-dirty" if it had uncommitted changes.
        policy: Registry "<model>@<alias>" or checkpoint path.
        policy_version: Registry model version; None for a checkpoint.
        track_id: Track the sweep drove on; also a directory name.
        map_yaml: Map file the simulator loaded.
        start_pose: Fixed start pose x (metres), y (metres), theta (rad).
        seed: Seed of each level's first episode; episode i used seed + i.
        episodes_per_level: Episodes run at each friction level.
        randomize_start: Start pose jitter was on.
    """

    sweep_id: str = Field(default_factory=new_ulid, pattern=ULID_PATTERN)
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    contract_version: str = CONTRACT_VERSION
    git_commit: str
    policy: str
    policy_version: str | None
    track_id: str = Field(pattern=TRACK_ID_PATTERN)
    map_yaml: str
    start_pose: tuple[float, float, float]
    seed: int
    episodes_per_level: int = Field(ge=1)
    randomize_start: bool


class LevelReport(_Strict):
    """One friction level's aggregates; eval_dir holds its summary.json and traces.

    mean_lap_time_s is over completed episodes, seconds; None if none completed.
    """

    friction_mu: float = Field(gt=0.0)
    episodes: int = Field(ge=1)
    completed: int = Field(ge=0)
    crashed: int = Field(ge=0)
    completion_rate: float = Field(ge=0.0, le=1.0)
    mean_lap_time_s: float | None
    eval_dir: str


class CrashPoint(_Strict):
    """Where an episode crashed: world-frame x, y in metres, and the fraction of the lap driven."""

    friction_mu: float
    seed: int
    x: float
    y: float
    lap_fraction: float


class SweepReport(_Strict):
    """Contents of sweep.json: the sweep, every level's aggregates and every crash."""

    run: SweepRun
    levels: tuple[LevelReport, ...]
    crashes: tuple[CrashPoint, ...]

    def level(self, friction_mu: float) -> LevelReport | None:
        """Return the level run at `friction_mu` (to 1e-9), or None if there is none."""
        for level in self.levels:
            if math.isclose(level.friction_mu, friction_mu, rel_tol=0.0, abs_tol=1e-9):
                return level
        return None


def _level_report(level: FrictionLevel, eval_dir: str) -> LevelReport:
    """Aggregate one level's episodes."""
    episodes = level.summary.episodes
    mean_lap_time = level.summary.mean_lap_time_s
    return LevelReport(
        friction_mu=level.friction_mu,
        episodes=len(episodes),
        completed=sum(e.status is LapStatus.COMPLETED for e in episodes),
        crashed=sum(e.status is LapStatus.CRASHED for e in episodes),
        completion_rate=level.summary.completion_rate,
        mean_lap_time_s=round(mean_lap_time, DECIMALS_TIME) if mean_lap_time is not None else None,
        eval_dir=eval_dir,
    )


def _crash_points(level: FrictionLevel) -> list[CrashPoint]:
    """Where each crashed episode of `level` ended."""
    return [
        CrashPoint(
            friction_mu=level.friction_mu,
            seed=e.seed,
            x=round(float(e.trace.x[-1]), DECIMALS_POSITION),
            y=round(float(e.trace.y[-1]), DECIMALS_POSITION),
            lap_fraction=round(e.lap_fraction, DECIMALS_PROGRESS),
        )
        for e in level.summary.episodes
        if e.status is LapStatus.CRASHED and e.trace is not None
    ]


def write_sweep(root: Path, run: SweepRun, levels: Sequence[FrictionLevel]) -> Path:
    """Write sweep.json and every level's export to root/<track_id>/<sweep_id>/.

    Args:
        root: Export root, normally FRICTION_SWEEP_DIR.
        run: What was swept and under which conditions.
        levels: The sweep's levels, each episode with its trace.

    Returns:
        The sweep directory.

    Raises:
        ValueError: If there are no levels, two levels share a friction, or an episode has no
            trace; nothing is written.
        FileExistsError: If the sweep directory already exists.
    """
    frictions = [level.friction_mu for level in levels]
    if not frictions or len(set(frictions)) != len(frictions):
        raise ValueError(f"levels must be non-empty with distinct frictions, got {frictions}")
    if any(e.trace is None for level in levels for e in level.summary.episodes):
        raise ValueError("every episode needs a trace to export")
    directory = root / run.track_id / run.sweep_id
    directory.mkdir(parents=True, exist_ok=False)
    reports, crashes = [], []
    for level in levels:
        eval_dir = f"mu_{level.friction_mu:g}"
        eval_run = EvalRun(
            git_commit=run.git_commit,
            policy=run.policy,
            policy_version=run.policy_version,
            track_id=run.track_id,
            map_yaml=run.map_yaml,
            start_pose=run.start_pose,
            seed=run.seed,
            randomize_start=run.randomize_start,
            friction_mu=level.friction_mu,
            obs_noise=ObservationNoise(),
        )
        write_eval_to(directory / eval_dir, eval_run, level.summary)
        reports.append(_level_report(level, eval_dir))
        crashes += _crash_points(level)
    report = SweepReport(run=run, levels=tuple(reports), crashes=tuple(crashes))
    (directory / SWEEP_FILE).write_text(report.model_dump_json(indent=2) + "\n")
    logger.info("wrote %s and %d friction levels to %s", SWEEP_FILE, len(reports), directory)
    return directory


def read_sweep(directory: Path) -> SweepReport:
    """Read the sweep.json written by write_sweep.

    Args:
        directory: Sweep directory.

    Returns:
        The validated report.

    Raises:
        FileNotFoundError: If the directory has no sweep.json.
        pydantic.ValidationError: If sweep.json does not match SweepReport.
    """
    return SweepReport.model_validate_json((directory / SWEEP_FILE).read_text())
