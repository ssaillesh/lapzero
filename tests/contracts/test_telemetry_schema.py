"""Contract Section 8 / test_telemetry_schema: lap telemetry JSON matches its schema file.

A recorded lap validates against schemas/lap_telemetry.schema.json, every frames array has
result.sample_count items, and the file holds no NaN or Infinity (Section 5). A document that
breaks a Section 5 rule is rejected by both the schema file and the LapTelemetry model, so the two
cannot drift apart.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from lapzero.contract import ACT_HIGH, ACT_LOW, SIM_V_MAX, SIM_V_MIN, STEER_MAX, TELEMETRY_SCHEMA
from lapzero.io.telemetry import (
    SEMVER_PATTERN,
    TRACK_ID_PATTERN,
    ULID_PATTERN,
    DriverInfo,
    DriverType,
    LapFrames,
    LapResult,
    LapTelemetry,
    SimInfo,
    TrackInfo,
)
from tests.conftest import human_driver_json, lap_telemetry_document, model_driver_json, record_lap

SCHEMA = json.loads(
    (Path(__file__).parents[2] / "schemas" / "lap_telemetry.schema.json").read_text()
)
VALIDATOR = Draft202012Validator(SCHEMA)
MODEL_DRIVER = DriverInfo(
    type=DriverType.MODEL,
    name="lapzero-ppo",
    model_version="7",
    mlflow_run_id="3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
)
MISSING = object()  # _changed deletes the key instead of setting a value


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not valid JSON")


def _strict_json(text: str) -> Any:
    """Parse JSON, rejecting the NaN, Infinity and -Infinity tokens Python's json module allows."""
    return json.loads(text, parse_constant=_reject_constant)


def _changed(
    changes: Mapping[tuple[str | int, ...], object], document: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A copy of `document` (default: a valid lap upload) with the item at each path set (or
    removed). The JSON round trip unshares frames columns that alias one list in the builder."""
    document = json.loads(json.dumps(lap_telemetry_document() if document is None else document))
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


def test_schema_has_exactly_the_fields_of_the_telemetry_model() -> None:
    defs = SCHEMA["$defs"]
    assert set(SCHEMA["properties"]) == _field_names(LapTelemetry)
    assert set(SCHEMA["required"]) == _field_names(LapTelemetry)
    for key, model in [
        ("track", TrackInfo),
        ("driver", DriverInfo),
        ("sim", SimInfo),
        ("result", LapResult),
        ("frames", LapFrames),
    ]:
        assert set(defs[key]["properties"]) == _field_names(model), key
        assert set(defs[key]["required"]) == _field_names(model), key


def test_schema_bounds_and_patterns_are_the_contract_values() -> None:
    defs = SCHEMA["$defs"]
    frames = defs["frames"]["properties"]
    assert SCHEMA["properties"]["schema"]["const"] == TELEMETRY_SCHEMA
    assert defs["ulid"]["pattern"] == ULID_PATTERN
    assert defs["semver"]["pattern"] == SEMVER_PATTERN
    assert defs["track_id"]["pattern"] == TRACK_ID_PATTERN
    assert (defs["normalized"]["minimum"], defs["normalized"]["maximum"]) == (
        ACT_LOW[0],
        ACT_HIGH[0],
    )
    steer_rad = frames["steer_rad"]["items"]
    assert (steer_rad["minimum"], steer_rad["maximum"]) == (-STEER_MAX, STEER_MAX)
    v_target = frames["v_target"]["items"]
    assert (v_target["minimum"], v_target["maximum"]) == (SIM_V_MIN, SIM_V_MAX)
    assert defs["sim"]["properties"]["v_max_train"]["maximum"] == SIM_V_MAX


@pytest.mark.parametrize(
    "recording",
    [
        pytest.param({}, id="completed human lap"),
        pytest.param({"crash": True}, id="crashed human lap"),
        pytest.param({"driver": MODEL_DRIVER}, id="completed model lap"),
    ],
)
def test_recorded_lap_validates_with_equal_length_frames_and_no_nan(
    tmp_path: Path, recording: dict[str, Any]
) -> None:
    files = record_lap(tmp_path, steps=40, **recording)
    document = _strict_json(files.json_path.read_text())
    VALIDATOR.validate(document)
    lengths = {name: len(column) for name, column in document["frames"].items()}
    assert set(lengths.values()) == {document["result"]["sample_count"]}, lengths


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(lap_telemetry_document(), id="completed model lap"),
        pytest.param(lap_telemetry_document(driver=human_driver_json()), id="human lap"),
        pytest.param(lap_telemetry_document(lap_time_s=None), id="crashed lap"),
        pytest.param(
            _changed({("result", "status"): "timeout"}, lap_telemetry_document(lap_time_s=None)),
            id="timeout",
        ),
        pytest.param(lap_telemetry_document(contract_version="1.0.0"), id="older contract"),
        pytest.param(
            _changed({("frames", "steer_rad", 0): -0.4189, ("frames", "steer_rad", 1): 0.4189}),
            id="steer_rad at +-STEER_MAX",
        ),
        pytest.param(
            _changed({("frames", "v_target", 0): -5.0, ("frames", "v_target", 1): 20.0}),
            id="v_target at the simulator limits",
        ),
        pytest.param(
            _changed({("frames", "throttle", 0): -1.0, ("frames", "throttle", 1): 1.0}),
            id="throttle at +-1",
        ),
    ],
)
def test_valid_document_passes_both_schema_and_model(document: dict[str, Any]) -> None:
    VALIDATOR.validate(document)
    LapTelemetry.model_validate(document)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(_changed({("frames", "steer", 3): 1.5}), id="steer above +1"),
        pytest.param(_changed({("frames", "throttle", 3): -1.2}), id="throttle below -1"),
        pytest.param(_changed({("frames", "steer_rad", 3): 0.42}), id="steer_rad over STEER_MAX"),
        pytest.param(_changed({("frames", "v_target", 3): 20.5}), id="v_target over SIM_V_MAX"),
        pytest.param(_changed({("frames", "v_target", 3): -5.5}), id="v_target under SIM_V_MIN"),
        pytest.param(_changed({("frames", "progress", 3): 1.01}), id="progress above 1"),
        pytest.param(_changed({("frames", "t", 0): -0.04}), id="negative t"),
        pytest.param(
            _changed({("frames", "yaw_rate"): [0.0] * 26}), id="frames column not in contract"
        ),
        pytest.param(_changed({("frames", "v_target"): MISSING}), id="missing frames column"),
        pytest.param(_changed({("result", "status"): "finished"}), id="unknown status"),
        pytest.param(
            _changed({("result", "lap_time_s"): None}), id="completed lap without lap time"
        ),
        pytest.param(_changed({("result", "status"): "crashed"}), id="crashed lap with lap time"),
        pytest.param(_changed({("result", "lap_time_s"): 0.0}), id="zero lap time"),
        pytest.param(_changed({("result", "sample_count"): 0}), id="zero samples"),
        pytest.param(
            lap_telemetry_document(driver=human_driver_json() | {"model_version": "7"}),
            id="human with model fields",
        ),
        pytest.param(
            lap_telemetry_document(driver=human_driver_json() | {"human_input_device": None}),
            id="human without input device",
        ),
        pytest.param(
            lap_telemetry_document(driver=model_driver_json() | {"mlflow_run_id": None}),
            id="model without run id",
        ),
        pytest.param(
            lap_telemetry_document(driver=model_driver_json() | {"human_input_device": "gamepad"}),
            id="model with input device",
        ),
        pytest.param(_changed({("driver", "type"): "robot"}), id="unknown driver type"),
        pytest.param(_changed({("sim", "v_max_train"): 25.0}), id="v_max_train over SIM_V_MAX"),
        pytest.param(_changed({("lap_index",): 0}), id="lap_index zero"),
        pytest.param(_changed({("lap_id",): "lap-1"}), id="lap_id not a ULID"),
        pytest.param(_changed({("track", "track_id"): "../etc"}), id="track_id with a path"),
        pytest.param(_changed({("track", "direction"): "left"}), id="unknown direction"),
        pytest.param(_changed({("schema",): "lapzero.replay"}), id="wrong schema name"),
        pytest.param(_changed({("contract_version",): "1.1"}), id="contract_version not semver"),
        pytest.param(
            _changed({("created_at",): "2026-10-20T14:32:05"}), id="created_at without timezone"
        ),
        pytest.param(_changed({("obs",): []}), id="unknown top-level field"),
        pytest.param(_changed({("result",): MISSING}), id="missing result"),
    ],
)
def test_document_breaking_a_section_5_rule_is_rejected_by_schema_and_model(
    document: dict[str, Any],
) -> None:
    assert not VALIDATOR.is_valid(document)
    with pytest.raises(ValidationError):
        LapTelemetry.model_validate(document)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_or_infinity_in_a_telemetry_file_is_rejected(bad: float) -> None:
    text = json.dumps(_changed({("frames", "x", 3): bad}))
    with pytest.raises(ValueError, match="not valid JSON"):
        _strict_json(text)
    with pytest.raises(ValidationError):
        LapTelemetry.model_validate_json(text)
