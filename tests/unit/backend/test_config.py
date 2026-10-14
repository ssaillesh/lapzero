"""Unit tests for backend/app/config.py: defaults, environment variables and .env files."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import make_url

from backend.app.config import REPO_ROOT, Environment, Settings, get_settings

_KEYS = (
    "DATABASE_URL",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_MODEL_NAME",
    "CHAMPION_SNAPSHOT",
    "CORS_ORIGINS",
    "ENVIRONMENT",
    "LAPS_DIR",
    "TRACKS_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own settings so defaults are tested."""
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_database_is_local_postgres_with_no_password_in_code() -> None:
    url = make_url(Settings(_env_file=None).database_url)

    assert url.drivername == "postgresql+psycopg"
    assert url.password is None
    assert (url.host, url.port, url.database) == ("localhost", 5432, "lapzero")


def test_default_stores_are_the_repo_directories_the_rest_of_lapzero_uses() -> None:
    settings = Settings(_env_file=None)

    assert Path(__file__).resolve().parents[3] == REPO_ROOT
    assert settings.mlflow_tracking_uri == (REPO_ROOT / "mlruns").as_uri()
    assert settings.mlflow_model_name == "lapzero-policy"
    assert settings.laps_dir == REPO_ROOT / "data" / "laps"
    assert settings.tracks_dir == REPO_ROOT / "tracks"


def test_default_cors_origins_are_the_local_frontend_dev_servers() -> None:
    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://localhost:5173", "http://localhost:3000"]
    assert settings.environment is Environment.DEVELOPMENT


def test_environment_variables_override_the_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://api:secret@db:5432/lapzero")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("CORS_ORIGINS", '["https://lapzero.example"]')
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LAPS_DIR", str(tmp_path / "laps"))

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://api:secret@db:5432/lapzero"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.cors_origins == ["https://lapzero.example"]
    assert settings.environment is Environment.PRODUCTION
    assert settings.laps_dir == tmp_path / "laps"


@pytest.mark.parametrize(
    "value",
    [
        "https://lapzero.example",  # one origin, not JSON
        "https://lapzero.example/",  # trailing slash: a browser's Origin never has one
        '"https://lapzero.example"',  # quoted origin
        '\u00a0["https://lapzero.example"]',  # non-breaking space pasted from a web page
        '\ufeff["https://lapzero.example"]',  # byte order mark
        '\u200b["https://lapzero.example"] ',  # zero-width space and trailing space
        "'[\"https://lapzero.example\"]'",  # JSON list wrapped in shell-style quotes
        "[\u201chttps://lapzero.example\u201d]",  # curly quotes inside the JSON list
    ],
)
def test_cors_origins_tolerate_how_hosting_dashboards_get_them_pasted(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", value)

    assert Settings(_env_file=None).cors_origins == ["https://lapzero.example"]


def test_cors_origins_may_be_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", " https://a.example, https://b.example/ ,")

    assert Settings(_env_file=None).cors_origins == ["https://a.example", "https://b.example"]


def test_cors_origins_in_a_dotenv_file_may_be_comma_separated(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("CORS_ORIGINS=https://a.example,https://b.example\n")

    settings = Settings(_env_file=dotenv)

    assert settings.cors_origins == ["https://a.example", "https://b.example"]


@pytest.mark.parametrize("value", ["", "   ", "[]", '[""]', ",", "''"])
def test_cors_origins_without_an_origin_are_rejected_with_a_hint(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", value)

    with pytest.raises(ValidationError, match="CORS_ORIGINS holds no origin"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('["https://lapzero.example"', "CORS_ORIGINS is not a valid JSON list"),
        ('["https://lapzero.example", 5]', "CORS_ORIGINS must be a JSON list of strings"),
    ],
)
def test_a_broken_json_list_of_cors_origins_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str, message: str
) -> None:
    monkeypatch.setenv("CORS_ORIGINS", value)

    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_champion_comes_from_the_registry_unless_a_snapshot_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings(_env_file=None).champion_snapshot is None

    monkeypatch.setenv("CHAMPION_SNAPSHOT", "backend/champion_snapshot.json")

    assert Settings(_env_file=None).champion_snapshot == Path("backend/champion_snapshot.json")


def test_dotenv_file_is_read_and_its_container_only_keys_are_ignored(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "POSTGRES_PASSWORD=change-me\n"
        "POSTGRES_PORT=5433\n"
        "DATABASE_URL=postgresql+psycopg://lapzero:change-me@localhost:5433/lapzero\n"
    )

    settings = Settings(_env_file=dotenv)

    assert make_url(settings.database_url).port == 5433


def test_unknown_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")

    with pytest.raises(ValidationError, match="environment"):
        Settings(_env_file=None)


def test_empty_model_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "")

    with pytest.raises(ValidationError, match="mlflow_model_name"):
        Settings(_env_file=None)


def test_settings_cannot_be_changed_after_loading() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.database_url = "sqlite://"  # type: ignore[misc]


def test_get_settings_reads_the_environment_once(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "first-policy")
    try:
        first = get_settings()
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "second-policy")

        assert get_settings() is first
        assert first.mlflow_model_name == "first-policy"
    finally:
        get_settings.cache_clear()
