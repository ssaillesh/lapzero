"""Unit tests for lapzero/eval/gate.py: the robustness-aware promotion rule.

Sweeps are built directly as SweepReports on the 0.50 to 1.05 grid with 10 episodes per level,
so the robustness range (mu 0.60 to 1.05) holds 100 episodes.
"""

import math
from typing import Any

import pytest

from lapzero.contract import ContractViolationError
from lapzero.eval.gate import decide
from lapzero.eval.sweep import LevelReport, SweepReport, SweepRun, friction_grid

GRID = friction_grid(0.50, 1.05, 0.05)


def _sweep(
    completed: dict[float, int] | None = None,
    nominal_lap_s: float | None = 46.56,
    *,
    grid: tuple[float, ...] = GRID,
    episodes: int = 10,
    **run_overrides: Any,
) -> SweepReport:
    """A sweep where every level completes all episodes unless `completed` says otherwise.

    Laps at mu 1.05 take `nominal_lap_s`; elsewhere 48.0 s.
    """
    levels = []
    for mu in grid:
        done = (completed or {}).get(mu, episodes)
        lap = nominal_lap_s if math.isclose(mu, 1.05) else 48.0
        levels.append(
            LevelReport(
                friction_mu=mu,
                episodes=episodes,
                completed=done,
                crashed=episodes - done,
                completion_rate=done / episodes,
                mean_lap_time_s=lap if done else None,
                eval_dir=f"mu_{mu:g}",
            )
        )
    fields: dict[str, Any] = {
        "git_commit": "abc123",
        "policy": "lapzero-policy@champion",
        "policy_version": "5",
        "track_id": "oschersleben",
        "map_yaml": "data/tracks/oschersleben/oschersleben.yaml",
        "start_pose": (0.0, 0.0, 2.857),
        "seed": 0,
        "episodes_per_level": episodes,
        "randomize_start": True,
    }
    run = SweepRun(**(fields | run_overrides))
    return SweepReport(run=run, levels=tuple(levels), crashes=())


V5_LIKE = _sweep({0.5: 0, 0.55: 0, 0.6: 0, 0.65: 1}, 46.56)  # fast but fragile below mu 0.7


def test_a_robust_challenger_within_two_percent_of_the_champion_is_promoted() -> None:
    decision = decide(V5_LIKE, _sweep({0.5: 3}, 47.30))
    assert decision.promote
    assert decision.completion_rate == 1.0
    assert (decision.champion_lap_time_s, decision.challenger_lap_time_s) == (46.56, 47.30)
    robustness, pace = decision.reasons
    assert robustness.startswith("robustness passed")
    assert "100/100" in robustness
    assert pace.startswith("pace passed")
    assert "+1.59%" in pace


def test_crashes_below_the_robustness_range_do_not_count() -> None:
    decision = decide(V5_LIKE, _sweep({0.5: 0, 0.55: 0}, 46.56))
    assert decision.promote
    assert decision.completion_rate == 1.0


@pytest.mark.parametrize(("completed_at_0_60", "promote"), [(8, True), (7, False)])
def test_the_challenger_must_complete_98_percent_from_mu_0_60_to_1_05(
    completed_at_0_60: int, promote: bool
) -> None:
    decision = decide(V5_LIKE, _sweep({0.6: completed_at_0_60}, 46.56))
    assert decision.promote is promote
    assert decision.completion_rate == pytest.approx((90 + completed_at_0_60) / 100)


@pytest.mark.parametrize(("challenger_s", "promote"), [(51.0, True), (51.01, False)])
def test_the_challenger_may_be_at_most_two_percent_slower_at_nominal_grip(
    challenger_s: float, promote: bool
) -> None:
    decision = decide(_sweep(nominal_lap_s=50.0), _sweep(nominal_lap_s=challenger_s))
    assert decision.promote is promote
    assert decision.reasons[1].startswith("pace passed" if promote else "pace failed")


def test_a_faster_but_fragile_challenger_is_rejected() -> None:
    decision = decide(_sweep(), _sweep({0.65: 5}, 45.0))
    assert not decision.promote
    assert decision.reasons[0].startswith("robustness failed")
    assert decision.reasons[1].startswith("pace passed")


def test_a_challenger_without_a_nominal_lap_is_rejected() -> None:
    decision = decide(_sweep(), _sweep({1.05: 0}, None))
    assert not decision.promote
    assert decision.challenger_lap_time_s is None
    assert "challenger completed no lap" in decision.reasons[1]


def test_a_champion_without_a_nominal_lap_leaves_no_baseline_and_no_promotion() -> None:
    decision = decide(_sweep({1.05: 0}, None), _sweep())
    assert not decision.promote
    assert "no baseline" in decision.reasons[1]


def test_minor_contract_versions_may_differ() -> None:
    assert decide(_sweep(contract_version="1.0.0"), _sweep(contract_version="1.2.0")).promote


def test_the_gate_refuses_different_contract_major_versions() -> None:
    with pytest.raises(ContractViolationError, match="major"):
        decide(_sweep(contract_version="1.0.0"), _sweep(contract_version="2.0.0"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"track_id": "spielberg"},
        {"seed": 1},
        {"episodes_per_level": 20},
        {"randomize_start": False},
        {"start_pose": (1.0, 0.0, 2.857)},
    ],
)
def test_sweeps_with_different_settings_cannot_be_compared(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=f"differ in {next(iter(overrides))}"):
        decide(_sweep(), _sweep(**overrides))


def test_sweeps_over_different_friction_levels_cannot_be_compared() -> None:
    with pytest.raises(ValueError, match="friction levels"):
        decide(_sweep(), _sweep(grid=friction_grid(0.6, 1.05, 0.05)))


@pytest.mark.parametrize("grid", [friction_grid(0.65, 1.05, 0.05), friction_grid(0.5, 1.0, 0.05)])
def test_sweeps_must_include_mu_0_60_and_nominal_grip(grid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="no friction level"):
        decide(_sweep(grid=grid), _sweep(grid=grid))
