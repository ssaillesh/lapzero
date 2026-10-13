"""Integration test: scripts/check_sim_params.py against the installed F1TENTH Gym (contract
Section 8), starting the real simulator on its temporary blank map.
"""

import pytest

from scripts.check_sim_params import main

pytestmark = pytest.mark.integration


def test_installed_simulator_matches_the_contract_baseline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in (
        "SIM_DT",
        "RAW_BEAMS",
        "LIDAR_FOV",
        "LIDAR_MAX",
        "STEER_MAX (s_max)",
        "STEER_MAX (s_min)",
        "SIM_V_MIN",
        "SIM_V_MAX",
        "N_BEAMS (RAW_BEAMS / BEAM_BIN)",
    ):
        assert name in out
    assert "all 9 values match" in out
