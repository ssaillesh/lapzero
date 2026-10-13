"""Unit tests for backend/app/main.py: router assembly, CORS and the engine lifecycle."""

import json
import logging
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app import main
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.models.base import Base
from backend.app.models.domain import Lap
from lapzero.contract import CONTRACT_VERSION
from tests.conftest import lap_telemetry_document


def test_every_router_is_mounted_under_api(api_settings: Settings) -> None:
    paths = create_app(api_settings).openapi()["paths"]

    assert {
        (path, method.upper()) for path, operations in paths.items() for method in operations
    } == {
        ("/api/health", "GET"),
        ("/api/tracks", "GET"),
        ("/api/tracks/{track_id}", "GET"),
        ("/api/laps", "POST"),
        ("/api/replays/{lap_id}", "GET"),
        ("/api/leaderboard/{track_id}", "GET"),
        ("/api/models/champion", "GET"),
    }


def test_app_defaults_to_the_process_settings(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: api_settings)

    assert create_app().state.settings is api_settings


def test_requests_use_sessions_on_the_configured_database(
    api_settings: Settings, tmp_path: Path
) -> None:
    url = f"sqlite:///{tmp_path / 'api.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)  # tests only: real databases are migrated by Alembic
    app = create_app(api_settings.model_copy(update={"database_url": url}))

    with TestClient(app) as client:
        assert client.post("/api/laps", json=lap_telemetry_document()).status_code == 201

    with Session(engine) as session:
        assert session.get(Lap, "01J9ZK8V3Q7M2T6C1R4X5N8B0A") is not None
    engine.dispose()


def test_startup_and_shutdown_are_logged(
    api_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=main.__name__)

    with TestClient(create_app(api_settings)):
        pass

    assert f"environment=test contract_version={CONTRACT_VERSION}" in caplog.text
    assert "LapZero API stopped" in caplog.text


def test_invalid_request_lists_each_problem_without_echoing_the_input(client: TestClient) -> None:
    document = lap_telemetry_document()
    document["frames"]["x"][3] = math.nan
    del document["result"]

    response = client.post(
        "/api/laps", content=json.dumps(document), headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    issues = response.json()["detail"]
    assert all(set(issue) == {"loc", "msg", "type"} for issue in issues)
    assert {tuple(issue["loc"]) for issue in issues} == {
        ("body", "frames", "x", 3),
        ("body", "result"),
    }


@pytest.mark.parametrize(
    ("origin", "allowed"), [("http://localhost:5173", True), ("https://evil.example", False)]
)
def test_cors_allows_only_the_configured_origins(
    client: TestClient, origin: str, allowed: bool
) -> None:
    response = client.options(
        "/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
    )

    assert (response.headers.get("access-control-allow-origin") == origin) is allowed
