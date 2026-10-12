"""Unit tests for lapzero/io/telemetry.py: the contract Section 5 lap telemetry schema."""

import ast
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import lapzero
from lapzero.io.telemetry import DriverType, LapStatus, LapTelemetry, read_lap_telemetry


def contract_example() -> dict[str, Any]:
    """The contract Section 5.1 example, with sample_count matching its three frames."""
    return {
        "schema": "lapzero.lap_telemetry",
        "schema_version": "1.0.0",
        "contract_version": "1.0.0",
        "lap_id": "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
        "episode_id": "01J9ZK8T9E0Y3H2P6W1D7F4G5S",
        "lap_index": 1,
        "created_at": "2026-10-20T14:32:05Z",
        "track": {"track_id": "track_02", "direction": "ccw", "length_m": 82.4},
        "driver": {
            "type": "model",
            "name": "lapzero-ppo",
            "model_version": "7",
            "mlflow_run_id": "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
            "human_input_device": None,
        },
        "sim": {
            "simulator": "f1tenth_gym",
            "sim_version": "0.2.1",
            "sim_dt": 0.01,
            "action_repeat": 4,
            "ctrl_hz": 25,
            "v_max_train": 8.0,
            "seed": 42,
        },
        "result": {
            "status": "completed",
            "lap_time_s": 35.97,
            "collision": False,
            "off_track_events": 0,
            "max_speed_mps": 7.84,
            "mean_speed_mps": 6.12,
            "sample_count": 3,
        },
        "frames": {
            "t": [0.00, 0.04, 0.08],
            "x": [0.000, 0.051, 0.118],
            "y": [0.000, 0.002, 0.006],
            "theta": [0.0000, 0.0031, 0.0074],
            "v_long": [0.00, 0.58, 1.17],
            "steer": [0.000, 0.040, 0.062],
            "throttle": [1.000, 1.000, 0.950],
            "steer_rad": [0.0000, 0.0168, 0.0260],
            "v_target": [0.60, 1.18, 1.74],
            "progress": [0.0000, 0.0006, 0.0014],
        },
    }


def _with(dotted: str, value: Any) -> dict[str, Any]:
    """The contract example with one field (dotted path, list indices allowed) replaced."""
    document = contract_example()
    *parents, last = dotted.split(".")
    target: Any = document
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return document


HUMAN_DRIVER = {
    "type": "human",
    "name": "tester",
    "model_version": None,
    "mlflow_run_id": None,
    "human_input_device": "keyboard",
}


def test_contract_example_validates() -> None:
    telemetry = LapTelemetry.model_validate(contract_example())
    assert telemetry.schema_ == "lapzero.lap_telemetry"
    assert telemetry.result.status is LapStatus.COMPLETED
    assert telemetry.driver.type is DriverType.MODEL


def test_json_round_trip_reproduces_the_contract_field_names_and_values(tmp_path: Path) -> None:
    original = LapTelemetry.model_validate(contract_example())
    path = tmp_path / "lap.json"
    path.write_text(original.model_dump_json(by_alias=True))
    assert json.loads(path.read_text()) == contract_example()
    assert read_lap_telemetry(path) == original


def test_human_driver_with_input_device_and_null_model_fields_validates() -> None:
    telemetry = LapTelemetry.model_validate(_with("driver", HUMAN_DRIVER))
    assert telemetry.driver.type is DriverType.HUMAN


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("driver.model_version", "7"),
        ("driver.human_input_device", None),
        ("driver.human_input_device", "joystick"),
    ],
)
def test_human_driver_with_model_fields_or_without_a_device_is_rejected(
    field: str, value: Any
) -> None:
    document = _with("driver", HUMAN_DRIVER)
    document["driver"][field.split(".")[1]] = value
    with pytest.raises(ValidationError):
        LapTelemetry.model_validate(document)


def test_failed_lap_validates_only_with_null_lap_time() -> None:
    crashed = _with("result.status", "crashed")
    with pytest.raises(ValidationError, match="lap_time_s"):
        LapTelemetry.model_validate(crashed)
    crashed["result"]["lap_time_s"] = None
    assert LapTelemetry.model_validate(crashed).result.lap_time_s is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frames.v_long.1", math.nan),
        ("frames.x.2", math.inf),
        ("frames.x", [0.0, 0.051]),
        ("result.sample_count", 4),
        ("result.lap_time_s", None),
        ("result.status", "finished"),
        ("driver.human_input_device", "keyboard"),
        ("created_at", "2026-10-20T16:32:05+02:00"),
        ("created_at", "2026-10-20T14:32:05"),
        ("schema", "lapzero.replay"),
        ("contract_version", "1.0"),
        ("lap_id", "01J9ZK8V3Q7M2T6C1R4X5N8B0U"),
        ("episode_id", "short"),
        ("lap_index", 0),
        ("track.track_id", "../escape"),
        ("track.direction", "north"),
        ("frames.steer.0", 1.5),
        ("frames.steer_rad.0", 0.5),
        ("frames.progress.0", -0.1),
        ("sim.v_max_train", 25.0),
        ("unexpected_field", 1),
    ],
)
def test_documents_breaking_the_contract_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        LapTelemetry.model_validate(_with(field, value))


def test_created_at_is_written_as_utc_with_z_suffix() -> None:
    utc_offset = _with("created_at", "2026-10-20T14:32:05+00:00")
    dumped = json.loads(LapTelemetry.model_validate(utc_offset).model_dump_json(by_alias=True))
    assert dumped["created_at"] == "2026-10-20T14:32:05Z"


def test_reading_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_lap_telemetry(tmp_path / "missing.json")


def test_io_package_never_imports_sim_or_training() -> None:
    io_root = Path(lapzero.__file__).parent / "io"
    for path in io_root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                assert not module.startswith(("lapzero.sim", "lapzero.training")), (
                    f"{path.name} imports {module}"
                )
