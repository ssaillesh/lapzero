"""Backend unit test setup: an in-memory SQLite database and a TestClient on it.

Every test runs in a transaction that is rolled back afterwards; services' commits only release
a savepoint inside it. The PostgreSQL schema and migrations are covered by
tests/integration/backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.config import Environment, Settings
from backend.app.dependencies import get_session
from backend.app.main import create_app
from backend.app.models.base import Base
from tests.conftest import lap_telemetry_document


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    """One in-memory SQLite database with every table, shared by all backend unit tests."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection: Any, _record: Any) -> None:
        # pysqlite's own transaction handling breaks SAVEPOINT; let SQLAlchemy emit BEGIN.
        dbapi_connection.isolation_level = None
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    @event.listens_for(engine, "begin")
    def _begin(connection: Any) -> None:
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)  # tests only: real databases are migrated by Alembic
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session whose work, committed or not, is rolled back when the test ends."""
    with engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        yield session
        session.close()
        transaction.rollback()


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    """Settings that keep every file the API writes or reads under tmp_path."""
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        mlflow_model_name="test-policy",
        laps_dir=tmp_path / "laps",
        tracks_dir=tmp_path / "tracks",
        environment=Environment.TEST,
    )


@pytest.fixture
def client(api_settings: Settings, session: Session) -> Iterator[TestClient]:
    """A TestClient whose requests share the test's rolled-back session."""
    app = create_app(api_settings)
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield client


@pytest.fixture
def upload_lap(client: TestClient) -> Callable[..., Response]:
    """POST /api/laps with lap_telemetry_document(**kwargs) as the body."""

    def _upload(**kwargs: Any) -> Response:
        return client.post("/api/laps", json=lap_telemetry_document(**kwargs))

    return _upload
