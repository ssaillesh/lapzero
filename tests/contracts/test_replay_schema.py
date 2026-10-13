"""Contract Section 8 / test_replay_schema: replay JSON matches its schema file.

A replay built from lap telemetry validates against schemas/replay.schema.json, is sampled at
hz = 10, has equal-length points arrays and holds no NaN or Infinity (Section 6). A replay that
breaks a Section 6 rule is rejected by both the schema file and the Replay model, so the two cannot
drift apart.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from lapzero.contract import REPLAY_HZ, REPLAY_SCHEMA
from lapzero.io.replay import Replay, ReplayPoints, build_replay
from lapzero.io.telemetry import (
    SEMVER_PATTERN,
    TRACK_ID_PATTERN,
    ULID_PATTERN,
    LapTelemetry,
    read_lap_telemetry,
)
from tests.conftest import human_driver_json, lap_telemetry_document, record_lap

SCHEMA = json.loads((Path(__file__).parents[2] / "schemas" / "replay.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
MISSING = object()  # _changed deletes the key instead of setting a value


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not valid JSON")


def _strict_json(text: str) -> Any:
    """Parse JSON, rejecting the NaN, Infinity and -Infinity tokens Python's json module allows."""
    return json.loads(text, parse_constant=_reject_constant)


def _replay_json(telemetry: dict[str, Any]) -> dict[str, Any]:
    """The replay of a lap telemetry document, as GET /api/replays/{lap_id} serves it."""
    replay = build_replay(LapTelemetry.model_validate(telemetry))
    document: dict[str, Any] = _strict_json(replay.model_dump_json(by_alias=True))
    return document


def _changed(changes: Mapping[tuple[str | int, ...], object]) -> dict[str, Any]:
    """The 11-point replay of a valid 1 s lap with the item at each path set (or removed)."""
    document = _replay_json(lap_telemetry_document())
    for path, value in changes.items():
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        if value is MISSING:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return document


def _field_names(model: type[BaseModel]) -> set[str]:
    return {field.alias or name for name, field in model.model_fields.items()}


def test_schema_file_is_a_valid_draft_2020_12_schema() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_schema_has_exactly_the_fields_of_the_replay_model() -> None:
    points = SCHEMA["properties"]["points"]
    assert set(SCHEMA["properties"]) == set(SCHEMA["required"]) == _field_names(Replay)
    assert set(points["properties"]) == set(points["required"]) == _field_names(ReplayPoints)


def test_schema_constants_and_patterns_are_the_contract_values() -> None:
    properties = SCHEMA["properties"]
    assert properties["schema"]["const"] == REPLAY_SCHEMA
    assert properties["hz"]["const"] == REPLAY_HZ == 10
    assert properties["lap_id"]["pattern"] == ULID_PATTERN
    assert properties["schema_version"]["pattern"] == SEMVER_PATTERN
    assert properties["track_id"]["pattern"] == TRACK_ID_PATTERN


@pytest.mark.parametrize(
    "recording",
    [pytest.param({}, id="completed lap"), pytest.param({"crash": True}, id="crashed lap")],
)
def test_replay_of_a_recorded_lap_validates_at_10_hz_with_equal_length_points(
    tmp_path: Path, recording: dict[str, Any]
) -> None:
    files = record_lap(tmp_path, steps=60, **recording)
    replay = build_replay(read_lap_telemetry(files.json_path))
    document = _strict_json(replay.model_dump_json(by_alias=True))
    VALIDATOR.validate(document)
    assert document["hz"] == 10
    lengths = {name: len(column) for name, column in document["points"].items()}
    assert set(lengths.values()) == {len(document["points"]["t"])}, lengths
    assert len(document["points"]["t"]) > 1


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(_replay_json(lap_telemetry_document()), id="model lap"),
        pytest.param(
            _replay_json(lap_telemetry_document(driver=human_driver_json())), id="human lap"
        ),
        pytest.param(_replay_json(lap_telemetry_document(lap_time_s=None)), id="crashed lap"),
        pytest.param(_replay_json(lap_telemetry_document(frames=1)), id="single point"),
        pytest.param(
            _changed({("points", "steer", 0): -1.0, ("points", "steer", 1): 1.0}),
            id="steer at +-1",
        ),
    ],
)
def test_valid_replay_passes_both_schema_and_model(document: dict[str, Any]) -> None:
    VALIDATOR.validate(document)
    Replay.model_validate(document)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(_changed({("hz",): 25}), id="telemetry rate instead of 10 Hz"),
        pytest.param(_changed({("points", "steer", 3): 1.5}), id="steer above +1"),
        pytest.param(_changed({("points", "steer", 3): -1.5}), id="steer below -1"),
        pytest.param(_changed({("points", "t", 0): -0.1}), id="negative t"),
        pytest.param(
            _changed({("points",): {name: [] for name in ("t", "x", "y", "theta", "v", "steer")}}),
            id="no points",
        ),
        pytest.param(_changed({("points", "v"): MISSING}), id="missing points column"),
        pytest.param(_changed({("points", "a_x"): [0.0] * 11}), id="points column not in contract"),
        pytest.param(
            _changed({("points",): [{"t": 0.0, "x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0}]}),
            id="row-wise points",
        ),
        pytest.param(_changed({("lap_time_s",): 0.0}), id="zero lap time"),
        pytest.param(_changed({("lap_time_s",): MISSING}), id="missing lap time"),
        pytest.param(_changed({("driver_label",): ""}), id="empty driver label"),
        pytest.param(_changed({("driver_id",): "sam"}), id="field not in contract"),
        pytest.param(_changed({("lap_id",): "lap-1"}), id="lap_id not a ULID"),
        pytest.param(_changed({("track_id",): "../etc"}), id="track_id with a path"),
        pytest.param(_changed({("schema",): "lapzero.lap_telemetry"}), id="wrong schema name"),
        pytest.param(_changed({("schema_version",): "1"}), id="schema_version not semver"),
    ],
)
def test_replay_breaking_a_section_6_rule_is_rejected_by_schema_and_model(
    document: dict[str, Any],
) -> None:
    assert not VALIDATOR.is_valid(document)
    with pytest.raises(ValidationError):
        Replay.model_validate(document)
