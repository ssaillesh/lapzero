"""Request and response of POST /api/laps."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import AwareDatetime

from backend.app.schemas.base import ResponseModel
from lapzero.io.telemetry import DriverType, InputDevice, LapStatus, LapTelemetry

LapSubmitRequest: TypeAlias = LapTelemetry
"""The upload body is the lap telemetry JSON document itself (contract Section 5)."""


class LapResponse(ResponseModel):
    """A stored lap: the driver and result fields of its telemetry (times in s, speeds in m/s).

    driver_label is what the dashboard shows, e.g. "AI v7" for model version 7.
    """

    lap_id: str
    episode_id: str
    lap_index: int
    track_id: str
    driver_type: DriverType
    driver_name: str
    driver_label: str
    model_version: str | None
    mlflow_run_id: str | None
    human_input_device: InputDevice | None
    status: LapStatus
    lap_time_s: float | None
    collision: bool
    off_track_events: int
    max_speed_mps: float
    mean_speed_mps: float
    sample_count: int
    contract_version: str
    created_at: AwareDatetime
