"""Backend settings, read from environment variables and .env (CLAUDE.md rule 5).

Every key is listed with a dummy value in .env.example at the repo root.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lapzero.contract import LAPS_DIR

REPO_ROOT: Final = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    """Where the backend runs."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Backend configuration; each field reads the environment variable of its name in capitals.

    Attributes:
        database_url: SQLAlchemy URL of the PostgreSQL database. The default has no password, so
            set DATABASE_URL (see .env.example).
        mlflow_tracking_uri: MLflow tracking URI holding the model registry; defaults to the
            repo's mlruns/ store, where the training scripts register models.
        mlflow_model_name: Registered model whose champion GET /api/models/champion reports.
        cors_origins: Browser origins allowed to call the API.
        environment: Where the backend runs; logged at startup.
        laps_dir: Directory uploaded lap telemetry is stored in, as <track_id>/<lap_id>.json.
        tracks_dir: Directory holding each track's <track_id>/centreline.csv.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    database_url: str = "postgresql+psycopg://lapzero@localhost:5432/lapzero"
    mlflow_tracking_uri: str = (REPO_ROOT / "mlruns").as_uri()
    mlflow_model_name: str = Field(default="lapzero-policy", min_length=1)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )
    environment: Environment = Environment.DEVELOPMENT
    laps_dir: Path = REPO_ROOT / LAPS_DIR
    tracks_dir: Path = REPO_ROOT / "tracks"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment on first use."""
    return Settings()
