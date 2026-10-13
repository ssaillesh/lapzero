"""Response of GET /api/health."""

from __future__ import annotations

from typing import Literal

from backend.app.schemas.base import ResponseModel
from lapzero.contract import CONTRACT_VERSION


class HealthResponse(ResponseModel):
    """The API is up, and the state space contract version it validates uploads against."""

    status: Literal["ok"] = "ok"
    contract_version: str = CONTRACT_VERSION
