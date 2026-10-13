"""Unit tests for scripts/eval_rollout.py, run in-process with the rollout replaced by a stub, so
neither the simulator, MLflow nor the API is involved (tests/unit/eval/test_rollout.py covers
the rollout itself).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from mlflow.exceptions import MlflowException

import scripts.eval_rollout
from lapzero.eval.rollout import DEFAULT_API_URL, LapUploadError
from scripts.eval_rollout import main

LAP_ID = "01J9ZK8V3Q7M2T6C1R4X5N8B0A"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Keyword arguments of every evaluate_champion_on_track call; each returns LAP_ID."""
    recorded: list[dict[str, Any]] = []

    def evaluate(**kwargs: Any) -> str:
        recorded.append(kwargs)
        return LAP_ID

    monkeypatch.setattr(scripts.eval_rollout, "evaluate_champion_on_track", evaluate)
    return recorded


def test_an_uploaded_lap_exits_zero_and_logs_its_id(
    calls: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert main(["--track", "spielberg"]) == 0
    assert calls == [{"track_id": "spielberg", "api_url": DEFAULT_API_URL}]
    assert f"champion lap {LAP_ID} on spielberg uploaded" in caplog.text


def test_the_api_url_is_passed_through(calls: list[dict[str, Any]]) -> None:
    assert main(["--track", "oschersleben", "--api", "http://api:9000/api"]) == 0
    assert calls == [{"track_id": "oschersleben", "api_url": "http://api:9000/api"}]


@pytest.mark.parametrize(
    "error",
    [
        LapUploadError("POST http://localhost:8000/api/laps answered 409", 409),
        MlflowException("lapzero-policy has no champion"),
        FileNotFoundError("track 'nowhere' has no map file"),
        ValueError("track_id '../x' does not match"),
    ],
)
def test_a_failed_rollout_exits_one_and_logs_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    def evaluate(**kwargs: Any) -> str:
        raise error

    monkeypatch.setattr(scripts.eval_rollout, "evaluate_champion_on_track", evaluate)
    assert main(["--track", "spielberg"]) == 1
    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.getMessage().startswith("champion rollout on spielberg failed:")
    assert error.args[0] in record.getMessage()


def test_the_track_is_required(calls: list[dict[str, Any]]) -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2
    assert calls == []
