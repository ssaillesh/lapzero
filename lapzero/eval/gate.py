"""Release gate: decides whether a challenger policy replaces the champion.

Robustness-aware promotion rule (README "Release gate"; changing it needs approval). Both
policies are swept over friction with the same settings (lapzero.eval.sweep). The challenger
replaces the champion only if both hold:

1. Robustness: it completes at least MIN_COMPLETION of all episodes at friction levels with mu
   in ROBUSTNESS_MU, pooled across those levels.
2. Pace: its mean lap time at nominal grip (NOMINAL_MU) is at most MAX_SLOWDOWN slower than
   the champion's.

The gate refuses to compare policies whose contract_version major numbers differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from lapzero.contract import ContractViolationError
from lapzero.eval.sweep import SweepReport

MIN_COMPLETION: Final = 0.98
"""Fraction of episodes in the robustness range the challenger must complete."""
ROBUSTNESS_MU: Final = (0.60, 1.05)
"""Friction range (inclusive) whose episodes count towards the completion rate."""
NOMINAL_MU: Final = 1.05
"""Nominal grip for the pace check: the sweep's top level, F1TENTH's default mu 1.0489 rounded."""
MAX_SLOWDOWN: Final = 0.02
"""Largest allowed relative increase of the nominal-grip lap time over the champion's."""
_MU_TOL: Final = 1e-9


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The gate's verdict and the numbers behind it.

    Attributes:
        promote: True if the challenger replaces the champion.
        reasons: One line per check, saying whether and why it passed.
        completion_rate: Challenger's pooled completion rate over ROBUSTNESS_MU.
        champion_lap_time_s: Champion's mean lap time at NOMINAL_MU, s; None if no lap.
        challenger_lap_time_s: Challenger's mean lap time at NOMINAL_MU, s; None if no lap.
    """

    promote: bool
    reasons: tuple[str, ...]
    completion_rate: float
    champion_lap_time_s: float | None
    challenger_lap_time_s: float | None


def _major(version: str) -> str:
    return version.split(".")[0]


def _check_comparable(champion: SweepReport, challenger: SweepReport) -> None:
    """Raise unless both sweeps ran under the same contract major and the same settings."""
    if _major(champion.run.contract_version) != _major(challenger.run.contract_version):
        raise ContractViolationError(
            f"cannot compare contract_version {champion.run.contract_version} (champion) with "
            f"{challenger.run.contract_version} (challenger): major versions differ"
        )
    for name in ("track_id", "seed", "episodes_per_level", "randomize_start", "start_pose"):
        ours, theirs = getattr(champion.run, name), getattr(challenger.run, name)
        if ours != theirs:
            raise ValueError(f"sweeps differ in {name}: champion {ours}, challenger {theirs}")
    frictions = [[level.friction_mu for level in sweep.levels] for sweep in (champion, challenger)]
    if frictions[0] != frictions[1]:
        raise ValueError(f"sweeps differ in friction levels: {frictions[0]} vs {frictions[1]}")
    for mu in (ROBUSTNESS_MU[0], NOMINAL_MU):
        if champion.level(mu) is None:
            raise ValueError(f"sweeps have no friction level at mu {mu}")


def _robustness(challenger: SweepReport) -> tuple[bool, float, str]:
    """Check the challenger's pooled completion rate over ROBUSTNESS_MU."""
    low, high = ROBUSTNESS_MU
    levels = [
        level for level in challenger.levels if low - _MU_TOL <= level.friction_mu <= high + _MU_TOL
    ]
    completed = sum(level.completed for level in levels)
    episodes = sum(level.episodes for level in levels)
    rate = completed / episodes
    passed = rate >= MIN_COMPLETION - _MU_TOL
    reason = (
        f"robustness {'passed' if passed else 'failed'}: challenger completed "
        f"{completed}/{episodes} episodes ({rate:.1%}) with mu in [{low:.2f}, {high:.2f}], "
        f"needs >= {MIN_COMPLETION:.0%}"
    )
    return passed, rate, reason


def _nominal_lap_time_s(sweep: SweepReport) -> float | None:
    """Mean lap time at NOMINAL_MU, s; None if no lap was completed there."""
    level = sweep.level(NOMINAL_MU)
    return level.mean_lap_time_s if level is not None else None


def _pace(champion_s: float | None, challenger_s: float | None) -> tuple[bool, str]:
    """Check the challenger's nominal-grip lap time against the champion's."""
    where = f"at nominal grip (mu {NOMINAL_MU:.2f})"
    if challenger_s is None:
        return False, f"pace failed: challenger completed no lap {where}"
    if champion_s is None:
        return False, f"pace failed: champion completed no lap {where}, so there is no baseline"
    slowdown = challenger_s / champion_s - 1.0
    passed = challenger_s <= champion_s * (1.0 + MAX_SLOWDOWN) + _MU_TOL
    reason = (
        f"pace {'passed' if passed else 'failed'}: challenger {challenger_s:.2f} s vs champion "
        f"{champion_s:.2f} s {where} ({slowdown:+.2%}), allowed +{MAX_SLOWDOWN:.1%}"
    )
    return passed, reason


def decide(champion: SweepReport, challenger: SweepReport) -> GateDecision:
    """Apply the robustness-aware promotion rule to two friction sweeps.

    Args:
        champion: Sweep of the current champion.
        challenger: Sweep of the challenger, with the same settings and friction levels.

    Returns:
        The decision, with one reason per check.

    Raises:
        ContractViolationError: If the contract_version major numbers differ.
        ValueError: If the sweeps differ in track, seeds, episodes, start or friction levels,
            or lack a level at ROBUSTNESS_MU's lower end or at NOMINAL_MU.
    """
    _check_comparable(champion, challenger)
    robust, rate, robustness_reason = _robustness(challenger)
    champion_s = _nominal_lap_time_s(champion)
    challenger_s = _nominal_lap_time_s(challenger)
    fast_enough, pace_reason = _pace(champion_s, challenger_s)
    return GateDecision(
        promote=robust and fast_enough,
        reasons=(robustness_reason, pace_reason),
        completion_rate=rate,
        champion_lap_time_s=champion_s,
        challenger_lap_time_s=challenger_s,
    )
