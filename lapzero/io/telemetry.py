"""Lap telemetry JSON (contract Section 5) as Pydantic models, plus a validating reader.

Shared by the lap recorder (writes files) and the backend (validates uploads), so this module must
never import lapzero.sim or lapzero.training.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from lapzero.contract import (
    SIM_V_MAX,
    SIM_V_MIN,
    STEER_MAX,
    TELEMETRY_SCHEMA,
    TELEMETRY_SCHEMA_VERSION,
)

ULID_PATTERN = r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$"  # 26 Crockford base32 chars, 128 bits
SEMVER_PATTERN = r"^\d+\.\d+\.\d+$"
TRACK_ID_PATTERN = r"^[A-Za-z0-9_-]+$"  # also used as a directory name, so no path separators


class DriverType(StrEnum):
    """Who produced the actions in a lap."""

    HUMAN = "human"
    MODEL = "model"


class InputDevice(StrEnum):
    """Input device of a human driver."""

    KEYBOARD = "keyboard"
    GAMEPAD = "gamepad"


class LapStatus(StrEnum):
    """How a recorded lap ended."""

    COMPLETED = "completed"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    STUCK = "stuck"


class TrackDirection(StrEnum):
    """Racing direction around the track."""

    CW = "cw"
    CCW = "ccw"


class _Strict(BaseModel):
    """Base for telemetry models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class TrackInfo(_Strict):
    """Track the lap was driven on. length_m is the centreline length in metres."""

    track_id: str = Field(pattern=TRACK_ID_PATTERN)
    direction: TrackDirection
    length_m: float = Field(gt=0.0)


class DriverInfo(_Strict):
    """Driver of the lap. Model fields are set only for models, the device only for humans."""

    type: DriverType
    name: str = Field(min_length=1)
    model_version: str | None = None
    mlflow_run_id: str | None = None
    human_input_device: InputDevice | None = None

    @model_validator(mode="after")
    def _fields_match_driver_type(self) -> Self:
        model_fields_set = self.model_version is not None and self.mlflow_run_id is not None
        model_fields_null = self.model_version is None and self.mlflow_run_id is None
        if self.type is DriverType.HUMAN and not (model_fields_null and self.human_input_device):
            raise ValueError(
                "human driver needs human_input_device and null model_version/mlflow_run_id"
            )
        if self.type is DriverType.MODEL and not (
            model_fields_set and self.human_input_device is None
        ):
            raise ValueError(
                "model driver needs model_version and mlflow_run_id and null human_input_device"
            )
        return self


class SimInfo(_Strict):
    """Simulator settings the lap was recorded with (sim_dt in s, v_max_train in m/s)."""

    simulator: str = Field(min_length=1)
    sim_version: str = Field(min_length=1)
    sim_dt: float = Field(gt=0.0)
    action_repeat: int = Field(ge=1)
    ctrl_hz: int = Field(ge=1)
    v_max_train: float = Field(gt=0.0, le=SIM_V_MAX)
    seed: int


class LapResult(_Strict):
    """Lap outcome. lap_time_s (s) is set only for completed laps; speeds are in m/s."""

    status: LapStatus
    lap_time_s: float | None = Field(default=None, gt=0.0)
    collision: bool
    off_track_events: int = Field(ge=0)
    max_speed_mps: float
    mean_speed_mps: float
    sample_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _lap_time_only_when_completed(self) -> Self:
        if (self.status is LapStatus.COMPLETED) != (self.lap_time_s is not None):
            raise ValueError("lap_time_s must be set for completed laps and null otherwise")
        return self


_Normalized = Annotated[float, Field(ge=-1.0, le=1.0)]


class LapFrames(_Strict):
    """Columnar per-control-step samples; every array has the same length.

    steer and throttle are the normalized policy action; steer_rad (rad) and v_target (m/s) are
    the simulator command it was mapped to. t is seconds since reset; x, y in metres; theta in
    rad; v_long in m/s; progress is the 0..1 fraction of the centreline.
    """

    t: list[Annotated[float, Field(ge=0.0)]]
    x: list[float]
    y: list[float]
    theta: list[float]
    v_long: list[float]
    steer: list[_Normalized]
    throttle: list[_Normalized]
    steer_rad: list[Annotated[float, Field(ge=-STEER_MAX, le=STEER_MAX)]]
    v_target: list[Annotated[float, Field(ge=SIM_V_MIN, le=SIM_V_MAX)]]
    progress: list[Annotated[float, Field(ge=0.0, le=1.0)]]

    @model_validator(mode="after")
    def _columns_have_equal_length(self) -> Self:
        lengths = {name: len(getattr(self, name)) for name in type(self).model_fields}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"frames arrays must all have the same length, got {lengths}")
        return self


class LapTelemetry(_Strict):
    """One recorded lap: the full contract Section 5 JSON document."""

    schema_: str = Field(
        default=TELEMETRY_SCHEMA, alias="schema", pattern=f"^{re.escape(TELEMETRY_SCHEMA)}$"
    )
    schema_version: str = Field(default=TELEMETRY_SCHEMA_VERSION, pattern=SEMVER_PATTERN)
    contract_version: str = Field(pattern=SEMVER_PATTERN)
    lap_id: str = Field(pattern=ULID_PATTERN)
    episode_id: str = Field(pattern=ULID_PATTERN)
    lap_index: int = Field(ge=1)  # 1 for the first lap of an episode
    created_at: AwareDatetime
    track: TrackInfo
    driver: DriverInfo
    sim: SimInfo
    result: LapResult
    frames: LapFrames

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"created_at must be UTC, got offset {value.utcoffset()}")
        return value

    @field_serializer("created_at")
    def _created_at_as_iso_z(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    @model_validator(mode="after")
    def _frames_match_sample_count(self) -> Self:
        if len(self.frames.t) != self.result.sample_count:
            raise ValueError(
                f"frames have {len(self.frames.t)} samples, result.sample_count is "
                f"{self.result.sample_count}"
            )
        return self


def read_lap_telemetry(path: Path) -> LapTelemetry:
    """Read and validate a lap telemetry JSON file.

    Args:
        path: Path to a <lap_id>.json file.

    Returns:
        The validated telemetry.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the content breaks the contract schema.
    """
    return LapTelemetry.model_validate_json(path.read_text())
