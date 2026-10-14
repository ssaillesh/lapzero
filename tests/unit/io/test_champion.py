"""Unit tests for lapzero/io/champion.py: the champion snapshot model and its JSON file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lapzero.io.champion import ChampionSnapshot, read_champion_snapshot, write_champion_snapshot


def snapshot_fields(**overrides: Any) -> dict[str, Any]:
    """Fields of a valid snapshot of lapzero-policy version 8, with overrides."""
    return {
        "model_name": "lapzero-policy",
        "version": "8",
        "run_id": "b1e5e003993e476e95481fd35aa2ac3c",
        "contract_version": "1.0.0",
        "tags": {"gate_result": "promote", "gate_track": "oschersleben"},
        "created_at": datetime(2026, 10, 13, 11, 16, 2, tzinfo=UTC),
        "exported_at": datetime(2026, 10, 14, 6, 30, tzinfo=UTC),
    } | overrides


def test_a_snapshot_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    snapshot = ChampionSnapshot(**snapshot_fields())
    path = tmp_path / "nested" / "champion.json"

    write_champion_snapshot(path, snapshot)

    assert read_champion_snapshot(path) == snapshot
    assert path.read_text().endswith("}\n")


def test_a_version_registered_without_a_run_has_no_run_or_contract(tmp_path: Path) -> None:
    snapshot = ChampionSnapshot(**snapshot_fields(run_id=None, contract_version=None, tags={}))
    write_champion_snapshot(tmp_path / "c.json", snapshot)

    assert read_champion_snapshot(tmp_path / "c.json") == snapshot


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"model_name": ""}, "model_name"),
        ({"version": "v8"}, "version"),
        ({"contract_version": "1.0"}, "contract_version"),
        ({"created_at": datetime(2026, 10, 13, 11, 16, 2)}, "created_at"),
        ({"exported_at": datetime(2026, 10, 14, tzinfo=timezone(timedelta(hours=8)))}, "UTC"),
        ({"weights": "not allowed"}, "weights"),
    ],
)
def test_an_invalid_snapshot_is_rejected(overrides: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        ChampionSnapshot(**snapshot_fields(**overrides))


def test_a_missing_snapshot_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_champion_snapshot(tmp_path / "missing.json")


def test_a_snapshot_cannot_be_changed_after_loading() -> None:
    snapshot = ChampionSnapshot(**snapshot_fields())

    with pytest.raises(ValidationError):
        snapshot.version = "9"  # type: ignore[misc]
