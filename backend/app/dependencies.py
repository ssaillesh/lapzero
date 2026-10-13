"""FastAPI dependencies shared by the routers: settings, database sessions and path parameters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Path, Request
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import Settings
from lapzero.io.telemetry import TRACK_ID_PATTERN, ULID_PATTERN


def get_app_settings(request: Request) -> Settings:
    """Return the settings the app was created with (backend.app.main.create_app)."""
    # Starlette's State is untyped; create_app stores a Settings instance here.
    settings: Settings = request.app.state.settings
    return settings


def get_session(request: Request) -> Iterator[Session]:
    """Yield a database session for one request and close it afterwards.

    Services commit their own work; anything left uncommitted is rolled back on close.
    """
    # Starlette's State is untyped; the app's lifespan stores a sessionmaker here.
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[Session, Depends(get_session)]
TrackIdPath = Annotated[str, Path(pattern=TRACK_ID_PATTERN, description="Track ID")]
LapIdPath = Annotated[str, Path(pattern=ULID_PATTERN, description="Lap ID (ULID)")]
