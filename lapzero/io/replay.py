"""Replay trajectory JSON (contract Section 6): the 10 Hz lap the browser draws as a ghost.

Built from a lap's telemetry (Section 5) when the backend ingests it, and served by
GET /api/replays/{lap_id}. The backend imports this module, so it must never import lapzero.sim
or lapzero.training.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lapzero.contract import (
    DECIMALS_ACTION,
    DECIMALS_ANGLE,
    DECIMALS_POSITION,
    DECIMALS_SPEED,
    DECIMALS_TIME,
    REPLAY_HZ,
    REPLAY_SCHEMA,
    REPLAY_SCHEMA_VERSION,
    ContractViolationError,
)
from lapzero.io.telemetry import (
    SEMVER_PATTERN,
    TRACK_ID_PATTERN,
    ULID_PATTERN,
    DriverType,
    LapTelemetry,
)


class _Strict(BaseModel):
    """Base for replay models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class ReplayPoints(_Strict):
    """Columnar 10 Hz samples of one lap; every array has the same length.

    t is seconds since the lap's first frame; x, y in metres (world frame); theta in rad; v is the
    longitudinal speed in m/s; steer is the normalized steering action in [-1, 1].
    """

    t: list[Annotated[float, Field(ge=0.0)]] = Field(min_length=1)
    x: list[float]
    y: list[float]
    theta: list[float]
    v: list[float]
    steer: list[Annotated[float, Field(ge=-1.0, le=1.0)]]

    @model_validator(mode="after")
    def _columns_have_equal_length(self) -> Self:
        lengths = {name: len(getattr(self, name)) for name in type(self).model_fields}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"points arrays must all have the same length, got {lengths}")
        return self


class Replay(_Strict):
    """One lap's replay: the full contract Section 6 JSON document.

    lap_time_s (s) is null for a lap that did not complete, as in the telemetry (Section 5.2).
    """

    schema_: str = Field(
        default=REPLAY_SCHEMA, alias="schema", pattern=f"^{re.escape(REPLAY_SCHEMA)}$"
    )
    schema_version: str = Field(default=REPLAY_SCHEMA_VERSION, pattern=SEMVER_PATTERN)
    lap_id: str = Field(pattern=ULID_PATTERN)
    track_id: str = Field(pattern=TRACK_ID_PATTERN)
    driver_label: str = Field(min_length=1)
    lap_time_s: float | None = Field(gt=0.0)
    hz: int = Field(default=REPLAY_HZ, ge=REPLAY_HZ, le=REPLAY_HZ)
    points: ReplayPoints


def driver_label(driver_type: DriverType, name: str, model_version: str | None) -> str:
    """Return the label the dashboard shows for a driver.

    Args:
        driver_type: Human or model.
        name: Driver name from the telemetry.
        model_version: Registry version of a model driver; None for humans.

    Returns:
        "AI v<model_version>" for a model with a version, otherwise the driver's name.
    """
    if driver_type == DriverType.MODEL and model_version is not None:
        return f"AI v{model_version}"
    return name


def build_replay(telemetry: LapTelemetry) -> Replay:
    """Downsample a lap's 25 Hz telemetry to the contract Section 6 replay.

    Samples are taken every 1 / REPLAY_HZ s from the lap's first frame, interpolating linearly on
    t between telemetry frames; theta is interpolated the short way round so a heading crossing
    +-pi does not spin. Replay t restarts at 0 on every lap, so laps from any point of an episode
    align by t for the ghost-lap comparison.

    Args:
        telemetry: A validated lap telemetry document.

    Returns:
        The replay, rounded to the Section 5.2 precision.

    Raises:
        ContractViolationError: If frames.t does not strictly increase.
    """
    frames = telemetry.frames
    t = np.asarray(frames.t, dtype=np.float64)
    not_increasing = np.flatnonzero(np.diff(t) <= 0.0)
    if not_increasing.size:
        i = int(not_increasing[0]) + 1
        raise ContractViolationError(
            f"frames.t must strictly increase, but frames.t[{i}] = {t[i]} s follows {t[i - 1]} s"
        )
    elapsed = t - t[0]
    # Round before flooring so float error (0.3 * 10 = 2.9999...) does not drop the last sample.
    count = math.floor(round(float(elapsed[-1]) * REPLAY_HZ, 6)) + 1
    grid = np.arange(count, dtype=np.float64) / REPLAY_HZ

    def resample(values: list[float]) -> NDArray[np.float64]:
        return np.interp(grid, elapsed, np.asarray(values, dtype=np.float64))

    theta = np.interp(grid, elapsed, np.unwrap(np.asarray(frames.theta, dtype=np.float64)))
    driver = telemetry.driver
    return Replay(
        lap_id=telemetry.lap_id,
        track_id=telemetry.track.track_id,
        driver_label=driver_label(driver.type, driver.name, driver.model_version),
        lap_time_s=telemetry.result.lap_time_s,
        points=ReplayPoints(
            t=_rounded(grid, DECIMALS_TIME),
            x=_rounded(resample(frames.x), DECIMALS_POSITION),
            y=_rounded(resample(frames.y), DECIMALS_POSITION),
            theta=_rounded(_wrap_angle(theta), DECIMALS_ANGLE),
            v=_rounded(resample(frames.v_long), DECIMALS_SPEED),
            steer=_rounded(resample(frames.steer), DECIMALS_ACTION),
        ),
    )


def _wrap_angle(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles in rad to (-pi, pi]."""
    return np.pi - np.mod(np.pi - angles, 2.0 * np.pi)


def _rounded(values: NDArray[np.float64], decimals: int) -> list[float]:
    """Return `values` rounded to `decimals` places as a list of Python floats."""
    return [round(float(value), decimals) for value in values]
