"""Response of GET /api/models/champion."""

from __future__ import annotations

from pydantic import AwareDatetime

from backend.app.schemas.base import ResponseModel


class ChampionResponse(ResponseModel):
    """The model version the registry's champion alias points at.

    Attributes:
        model_name: Registered model name.
        version: Registry version number, e.g. "8".
        run_id: MLflow run that trained the version; None if it was registered without one.
        contract_version: Contract version the run logged; None if unknown.
        tags: Version tags, e.g. the release gate's gate_result.
        created_at: When the version was registered (UTC).
    """

    model_name: str
    version: str
    run_id: str | None
    contract_version: str | None
    tags: dict[str, str]
    created_at: AwareDatetime
