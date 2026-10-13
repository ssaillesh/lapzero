"""Unit tests for scripts/check_sim_params.py, run in-process with the simulator replaced by a
fixed list of baseline values (the real simulator runs in tests/integration).
"""

from __future__ import annotations

import pytest

import scripts.check_sim_params
from lapzero.contract import CONTRACT_VERSION
from lapzero.sim.f1tenth import BaselineValue
from scripts.check_sim_params import main

MATCHING = [
    BaselineValue("SIM_DT", 0.01, 0.01),
    BaselineValue("STEER_MAX (s_max)", 0.4189, 0.4189),
    BaselineValue("N_BEAMS (RAW_BEAMS / BEAM_BIN)", 108.0, 108.0),
]


def _installed(monkeypatch: pytest.MonkeyPatch, values: list[BaselineValue]) -> None:
    monkeypatch.setattr(scripts.check_sim_params, "installed_baseline", lambda: values)


def test_matching_simulator_prints_every_value_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _installed(monkeypatch, MATCHING)
    assert main([]) == 0
    out = capsys.readouterr().out
    assert f"baseline vs contract {CONTRACT_VERSION}" in out
    for name in ("SIM_DT", "STEER_MAX (s_max)", "N_BEAMS (RAW_BEAMS / BEAM_BIN)"):
        assert name in out
    assert "0.4189      0.4189  ok" in out
    assert "MISMATCH" not in out
    assert "all 3 values match" in out


def test_differing_simulator_names_every_mismatch_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _installed(
        monkeypatch,
        [
            BaselineValue("SIM_DT", 0.01, 0.01),
            BaselineValue("STEER_MAX (s_max)", 0.36, 0.4189),
            BaselineValue("SIM_V_MAX", 25.0, 20.0),
        ],
    )
    assert main([]) == 1
    out = capsys.readouterr().out
    assert "0.36      0.4189  MISMATCH" in out
    assert "25          20  MISMATCH" in out
    assert "2 value(s) differ: STEER_MAX (s_max), SIM_V_MAX" in out
    assert "bump contract_version" in out
