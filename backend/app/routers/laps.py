"""POST /api/laps: upload one lap's telemetry (contract Section 5)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.dependencies import SessionDep, SettingsDep
from backend.app.schemas.laps import LapResponse, LapSubmitRequest
from backend.app.services import laps
from lapzero.contract import ContractViolationError

router = APIRouter(prefix="/api/laps", tags=["laps"])


@router.post("", status_code=status.HTTP_201_CREATED)
def submit_lap(
    telemetry: LapSubmitRequest, session: SessionDep, settings: SettingsDep
) -> LapResponse:
    """Validate and store a lap, its telemetry file and its replay.

    Responds 422 if the body breaks the telemetry schema or the contract (major version, frame
    times), and 409 if the lap was already uploaded.
    """
    try:
        lap = laps.ingest_lap(session, telemetry, settings.laps_dir)
    except laps.LapExistsError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    except ContractViolationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return LapResponse.model_validate(lap)
