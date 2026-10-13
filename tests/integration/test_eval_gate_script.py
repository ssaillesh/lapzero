"""Integration test: scripts/eval_gate.py run as a CLI on sweep directories written to disk and a
throwaway MLflow registry, checking its exit status, output and the promotion it makes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias, mlflow_client
from tests.conftest import make_sweep_report, register_model_versions, write_sweep_report

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("challenger_lap_s", "returncode", "verdict", "aliases"),
    [
        (46.20, 0, "PROMOTE", {"champion": "2"}),
        (48.00, 1, "REJECT", {"champion": "1", "challenger": "2"}),
    ],
)
def test_eval_gate_script_promotes_only_a_challenger_that_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    challenger_lap_s: float,
    returncode: int,
    verdict: str,
    aliases: dict[str, str],
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    register_model_versions(
        tracking_uri, REGISTERED_MODEL, 2, {ModelAlias.CHAMPION: "1", ModelAlias.CHALLENGER: "2"}
    )
    root = tmp_path / "friction_sweep"
    champion = write_sweep_report(root, make_sweep_report("1", nominal_lap_s=46.56))
    challenger = write_sweep_report(
        root, make_sweep_report("2", alias="challenger", nominal_lap_s=challenger_lap_s)
    )
    command = [
        sys.executable,
        "scripts/eval_gate.py",
        "--champion-sweep",
        str(champion),
        "--challenger-sweep",
        str(challenger),
        "--tracking-uri",
        tracking_uri,
        "--promote",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

    assert completed.returncode == returncode, completed.stderr
    assert f"verdict: {verdict}" in completed.stdout
    registry = mlflow_client(tracking_uri).get_registered_model(REGISTERED_MODEL)
    assert dict(registry.aliases) == aliases
