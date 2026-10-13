"""Integration tests of the backend on PostgreSQL: the Alembic migration and the API end to end.

Needs the docker-compose db service (`docker compose up -d db`) and DATABASE_URL pointing at it
(.env, see .env.example). The module creates a throwaway database on that server, migrates it
with Alembic and drops it afterwards; every API test runs in a transaction that is rolled back.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, inspect, make_url, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.config import REPO_ROOT, Environment, Settings
from backend.app.dependencies import get_session
from backend.app.main import create_app
from backend.app.models.base import Base
from tests.conftest import fixed_ulid, human_driver_json, lap_telemetry_document, model_driver_json

pytestmark = pytest.mark.integration

ALEMBIC_INI = REPO_ROOT / "backend" / "alembic.ini"
TABLES = {"alembic_version", "tracks", "laps", "replays"}


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    """URL of a new, empty database on the DATABASE_URL server; dropped after the module."""
    server = make_url(Settings().database_url)
    assert server.get_backend_name() == "postgresql", f"DATABASE_URL is not PostgreSQL: {server}"
    name = f"lapzero_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(server.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        yield server.set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.dispose()


@pytest.fixture(scope="module")
def alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@pytest.fixture(scope="module")
def engine(database_url: str, alembic_config: Config) -> Iterator[Engine]:
    """Engine on the throwaway database, migrated to head by Alembic."""
    command.upgrade(alembic_config, "head")
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def client(engine: Engine, tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient on the migrated database whose requests share one rolled-back transaction."""
    settings = Settings(
        _env_file=None,
        database_url=engine.url.render_as_string(hide_password=False),
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        laps_dir=tmp_path / "laps",
        tracks_dir=tmp_path / "tracks",
        environment=Environment.TEST,
    )
    with engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        app = create_app(settings)
        app.dependency_overrides[get_session] = lambda: session
        with TestClient(app) as client:
            yield client
        session.close()
        transaction.rollback()


def test_migrated_schema_matches_the_models(engine: Engine) -> None:
    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_migration_names_its_enum_check_constraints_like_the_models(engine: Engine) -> None:
    checks = {check["name"] for check in inspect(engine).get_check_constraints("laps")}

    assert checks == {"ck_laps_driver_type", "ck_laps_human_input_device", "ck_laps_status"}


def test_database_rejects_a_lap_status_outside_the_contract(engine: Engine) -> None:
    with engine.connect() as connection:
        transaction = connection.begin()
        connection.execute(text("INSERT INTO tracks VALUES ('t', 't', 82.4, '{}', now())"))
        with pytest.raises(IntegrityError, match="ck_laps_status"):
            connection.execute(
                text(
                    "INSERT INTO laps VALUES ('01J9ZK8V3Q7M2T6C1R4X5N8B0A', "
                    "'01J9ZK8T9E0Y3H2P6W1D7F4G5S', 1, 't', 'human', 'sam', NULL, NULL, "
                    "'keyboard', 'abandoned', NULL, false, 0, 1.0, 1.0, 1, '1.1.0', 'p', now())"
                )
            )
        transaction.rollback()


def test_uploaded_laps_rank_and_replay_on_postgresql(client: TestClient) -> None:
    laps = [
        (1, model_driver_json("7"), 35.5),
        (2, human_driver_json("sam"), 36.0),
        (3, human_driver_json("sam"), 37.0),
    ]
    for n, driver, lap_time_s in laps:
        document = lap_telemetry_document(fixed_ulid(n), driver=driver, lap_time_s=lap_time_s)
        assert client.post("/api/laps", json=document).status_code == 201

    board = client.get("/api/leaderboard/track_02").json()
    replay = client.get(f"/api/replays/{fixed_ulid(1)}").json()

    assert [(e["rank"], e["driver_label"], e["lap_time_s"]) for e in board["entries"]] == [
        (1, "AI v7", 35.5),
        (2, "sam", 36.0),
    ]
    assert board["entries"][0]["created_at"] == "2026-10-20T14:32:05Z"
    assert (replay["schema"], replay["hz"], replay["points"]["t"][-1]) == (
        "lapzero.replay",
        10,
        1.0,
    )
    assert [track["track_id"] for track in client.get("/api/tracks").json()] == ["track_02"]


def test_duplicate_upload_is_a_conflict_and_the_api_keeps_working(client: TestClient) -> None:
    assert client.post("/api/laps", json=lap_telemetry_document()).status_code == 201

    assert client.post("/api/laps", json=lap_telemetry_document()).status_code == 409
    assert client.post("/api/laps", json=lap_telemetry_document(fixed_ulid(9))).status_code == 201
    assert len(client.get("/api/leaderboard/track_02").json()["entries"]) == 1


def test_api_tests_leave_no_rows_behind(engine: Engine) -> None:
    with engine.connect() as connection:
        counts = [
            connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            for table in ("tracks", "laps", "replays")
        ]

    assert counts == [0, 0, 0]


def test_downgrade_drops_every_table_and_upgrade_restores_them(
    engine: Engine, alembic_config: Config
) -> None:
    command.downgrade(alembic_config, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}

    command.upgrade(alembic_config, "head")
    assert set(inspect(engine).get_table_names()) == TABLES
