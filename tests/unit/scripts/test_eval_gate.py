"""Unit tests for scripts/eval_gate.py, run in-process on temporary sweep directories and a
throwaway MLflow store holding lapzero-policy versions 1 (champion) and 2 (challenger).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lapzero.registry import GATE_PROMOTE
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias, mlflow_client
from scripts.eval_gate import main
from tests.conftest import make_sweep_report, register_model_versions, write_sweep_report

CHAMPION_SWEEP = "01M4TGPRVQY5F71QAQKBSR5RKQ"
CHALLENGER_SWEEP = "01M4TH84XCQYGAD34XR0C5BTDH"
CHAMPION_LAP_S = 46.56
FASTER_LAP_S = 46.20  # passes pace
SLOWER_LAP_S = 48.00  # +3.1%, fails pace (allowed +2.0%)


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test
    uri = (tmp_path / "mlruns").as_uri()
    register_model_versions(
        uri, REGISTERED_MODEL, 3, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"}
    )
    return uri


def _gate(
    tmp_path: Path,
    tracking_uri: str,
    *,
    challenger_lap_s: float,
    challenger_version: str | None = "2",
    promote: bool = False,
) -> int:
    """Write the champion (v1) and challenger sweeps on oschersleben and run the gate on them.

    challenger_version None writes a sweep of a local checkpoint.
    """
    root = tmp_path / "friction_sweep"
    champion = make_sweep_report("1", sweep_id=CHAMPION_SWEEP, nominal_lap_s=CHAMPION_LAP_S)
    challenger = make_sweep_report(
        challenger_version,
        alias="challenger",
        sweep_id=CHALLENGER_SWEEP,
        nominal_lap_s=challenger_lap_s,
    )
    argv = [
        "--champion-sweep",
        str(write_sweep_report(root, champion)),
        "--challenger-sweep",
        str(write_sweep_report(root, challenger)),
        "--tracking-uri",
        tracking_uri,
    ]
    return main([*argv, "--promote"] if promote else argv)


def _aliases(tracking_uri: str) -> dict[str, str]:
    return dict(mlflow_client(tracking_uri).get_registered_model(REGISTERED_MODEL).aliases)


def _tags(tracking_uri: str, version: str) -> dict[str, str]:
    return dict(mlflow_client(tracking_uri).get_model_version(REGISTERED_MODEL, version).tags)


def test_a_passing_gate_without_promote_reports_promote_and_leaves_the_registry(
    tmp_path: Path, tracking_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(tmp_path, tracking_uri, challenger_lap_s=FASTER_LAP_S) == 0
    out = capsys.readouterr().out
    assert "Release gate on oschersleben" in out
    assert f"version 1, sweep {CHAMPION_SWEEP}" in out
    assert f"version 2, sweep {CHALLENGER_SWEEP}" in out
    assert "robustness passed: challenger completed 100/100 episodes (100.0%)" in out
    assert "pace passed: challenger 46.20 s vs champion 46.56 s" in out
    assert "verdict: PROMOTE" in out
    assert "--promote" in out
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}


def test_promote_makes_the_swept_challenger_champion_when_the_gate_passes(
    tmp_path: Path, tracking_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(tmp_path, tracking_uri, challenger_lap_s=FASTER_LAP_S, promote=True) == 0
    assert "promoted lapzero-policy version 2 to champion, replacing version 1" in (
        capsys.readouterr().out
    )
    assert _aliases(tracking_uri) == {"champion": "2"}
    assert _tags(tracking_uri, "2") == {
        "algorithm": "ppo",
        "gate_result": GATE_PROMOTE,
        "gate_champion_sweep": CHAMPION_SWEEP,
        "gate_challenger_sweep": CHALLENGER_SWEEP,
        "gate_track": "oschersleben",
        "replaced_champion_version": "1",
    }


def test_a_rejected_challenger_is_never_promoted(
    tmp_path: Path, tracking_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate(tmp_path, tracking_uri, challenger_lap_s=SLOWER_LAP_S, promote=True) == 1
    out = capsys.readouterr().out
    assert "pace failed: challenger 48.00 s vs champion 46.56 s" in out
    assert "verdict: REJECT" in out
    assert "promoted" not in out
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}
    assert _tags(tracking_uri, "2") == {"algorithm": "ppo"}


def test_promote_refuses_when_training_moved_the_challenger_after_the_sweep(
    tmp_path: Path, tracking_uri: str
) -> None:
    client = mlflow_client(tracking_uri)
    client.set_registered_model_alias(REGISTERED_MODEL, ModelAlias.CHALLENGER, "3")
    with pytest.raises(ValueError, match="challenger is version 3, expected 2"):
        _gate(tmp_path, tracking_uri, challenger_lap_s=FASTER_LAP_S, promote=True)
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "3"}


def test_promote_refuses_a_challenger_swept_from_a_local_checkpoint(
    tmp_path: Path, tracking_uri: str
) -> None:
    with pytest.raises(ValueError, match=r"challenger sweep .* no registry version"):
        _gate(
            tmp_path,
            tracking_uri,
            challenger_lap_s=FASTER_LAP_S,
            challenger_version=None,  # a checkpoint sweep
            promote=True,
        )
    assert _aliases(tracking_uri) == {"champion": "1", "challenger": "2"}
