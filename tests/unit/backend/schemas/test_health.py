"""Unit tests for backend/app/schemas/health.py."""

from backend.app.schemas.health import HealthResponse
from lapzero.contract import CONTRACT_VERSION


def test_health_reports_ok_and_the_contract_version() -> None:
    assert HealthResponse().model_dump() == {"status": "ok", "contract_version": CONTRACT_VERSION}
