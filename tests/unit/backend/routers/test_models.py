"""Unit tests for backend/app/routers/models.py: GET /api/models/champion."""

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.services import models
from tests.conftest import register_model_versions


@pytest.fixture(autouse=True)
def allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test


def test_champion_is_read_from_the_configured_registry(
    client: TestClient, api_settings: Settings
) -> None:
    register_model_versions(
        api_settings.mlflow_tracking_uri, api_settings.mlflow_model_name, 2, {"champion": "2"}
    )

    response = client.get("/api/models/champion")

    assert response.status_code == 200
    body = response.json()
    assert (body["model_name"], body["version"]) == ("test-policy", "2")
    assert (body["run_id"], body["contract_version"]) == (None, None)
    assert body["tags"] == {"algorithm": "ppo"}
    assert body["created_at"].endswith("Z")


def test_missing_champion_is_not_found(client: TestClient) -> None:
    response = client.get("/api/models/champion")

    assert response.status_code == 404
    assert "test-policy" in response.json()["detail"]


def test_unreadable_registry_is_service_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_tracking_uri: str, _model_name: str) -> models.ChampionVersion:
        raise models.RegistryUnavailableError("model registry failed: INTERNAL_ERROR")

    monkeypatch.setattr(models, "get_champion", unavailable)

    response = client.get("/api/models/champion")

    assert response.status_code == 503
    assert response.json()["detail"] == "model registry failed: INTERNAL_ERROR"
