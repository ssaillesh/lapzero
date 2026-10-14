"""Unit tests for backend/app/routers/models.py: GET /api/models/champion."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.dependencies import get_session
from backend.app.main import create_app
from backend.app.services import models
from lapzero.io.champion import ChampionSnapshot, write_champion_snapshot
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


@pytest.fixture
def snapshot_path(tmp_path: Path) -> Path:
    """Where the snapshot-serving client's CHAMPION_SNAPSHOT points; tests write it or not."""
    return tmp_path / "champion_snapshot.json"


@pytest.fixture
def snapshot_client(
    api_settings: Settings, session: Session, snapshot_path: Path
) -> Iterator[TestClient]:
    """A client whose API serves the champion from CHAMPION_SNAPSHOT, with no registry at all."""
    settings = api_settings.model_copy(update={"champion_snapshot": snapshot_path})
    app = create_app(settings)
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield client


def test_champion_is_served_from_the_snapshot_when_one_is_configured(
    snapshot_client: TestClient, snapshot_path: Path
) -> None:
    snapshot = ChampionSnapshot(
        model_name="test-policy",
        version="8",
        run_id="b1e5e003993e476e95481fd35aa2ac3c",
        contract_version="1.0.0",
        tags={"gate_result": "promote"},
        created_at=datetime(2026, 10, 13, 11, 16, 2, tzinfo=UTC),
        exported_at=datetime(2026, 10, 14, 6, 30, tzinfo=UTC),
    )
    write_champion_snapshot(snapshot_path, snapshot)

    response = snapshot_client.get("/api/models/champion")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-policy",
        "version": "8",
        "run_id": "b1e5e003993e476e95481fd35aa2ac3c",
        "contract_version": "1.0.0",
        "tags": {"gate_result": "promote"},
        "created_at": "2026-10-13T11:16:02Z",
    }


def test_a_configured_snapshot_that_is_missing_is_service_unavailable(
    snapshot_client: TestClient,
) -> None:
    response = snapshot_client.get("/api/models/champion")

    assert response.status_code == 503
    assert "champion_snapshot.json cannot be read" in response.json()["detail"]
