"""Unit tests for backend/app/routers/health.py."""

from fastapi.testclient import TestClient

from lapzero.contract import CONTRACT_VERSION


def test_health_reports_ok_and_the_contract_version(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "contract_version": CONTRACT_VERSION}
