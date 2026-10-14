"""Backend settings, read from environment variables and .env (CLAUDE.md rule 5).

Every key is listed with a dummy value in .env.example at the repo root.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from lapzero.contract import LAPS_DIR

REPO_ROOT: Final = Path(__file__).resolve().parents[2]

_EDGE_JUNK: Final = re.compile(r"^[\s\ufeff\u200b-\u200d\u2060]+|[\s\ufeff\u200b-\u200d\u2060]+$")
"""Whitespace (non-breaking spaces included), the BOM and zero-width characters at either end:
copying a value from a web page can bring them along invisibly."""
_CURLY_QUOTES: Final = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


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
        champion_snapshot: JSON file written by scripts/export_champion.py. When set, the
            champion is served from it instead of the registry, for an API without MLflow.
        cors_origins: Browser origins allowed to call the API. CORS_ORIGINS may be a JSON list,
            one origin, or comma-separated origins (see _parse_cors_origins).
        environment: Where the backend runs; logged at startup.
        laps_dir: Directory uploaded lap telemetry is stored in, as <track_id>/<lap_id>.json.
        tracks_dir: Directory holding each track's <track_id>/centreline.csv.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    database_url: str = "postgresql+psycopg://lapzero@localhost:5432/lapzero"
    mlflow_tracking_uri: str = (REPO_ROOT / "mlruns").as_uri()
    mlflow_model_name: str = Field(default="lapzero-policy", min_length=1)
    champion_snapshot: Path | None = None
    # NoDecode: the raw string reaches _parse_cors_origins instead of failing JSON decoding.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )
    environment: Environment = Environment.DEVELOPMENT
    laps_dir: Path = REPO_ROOT / LAPS_DIR
    tracks_dir: Path = REPO_ROOT / "tracks"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Read CORS_ORIGINS as a JSON list, one origin, or comma-separated origins.

        Whitespace, invisible characters and quotes (curly ones included) around the value and
        around each origin are dropped, and so is a trailing slash, which a browser's Origin
        header never has. "https://a.example/" therefore reads as ["https://a.example"].

        Args:
            value: The raw environment value; anything but a string is left to Pydantic.

        Returns:
            The origins, e.g. ["https://lapzero.vercel.app"].

        Raises:
            ValueError: If the value holds no origin, or starts with "[" but is not a JSON list
                of strings.
        """
        if not isinstance(value, str):
            return value
        text = _unwrap(value.translate(_CURLY_QUOTES))
        if text.startswith("["):
            try:
                # json.loads returns Any; it is checked to be a list of strings below.
                items = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"CORS_ORIGINS is not a valid JSON list: {error}") from error
            if not (isinstance(items, list) and all(isinstance(item, str) for item in items)):
                raise ValueError(f"CORS_ORIGINS must be a JSON list of strings, got {text!r}")
        else:
            items = text.split(",")
        origins = [_unwrap(item).rstrip("/") for item in items]
        origins = [origin for origin in origins if origin]
        if not origins:
            raise ValueError(
                "CORS_ORIGINS holds no origin; set it to the dashboard's origin, "
                "e.g. https://your-app.vercel.app"
            )
        return origins


def _unwrap(text: str) -> str:
    """Strip whitespace and invisible characters, then one pair of matching outer quotes."""
    text = _EDGE_JUNK.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = _EDGE_JUNK.sub("", text[1:-1])
    return text


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment on first use."""
    return Settings()
