"""FastAPI app: routers, CORS and the database engine's lifecycle.

Run from the repo root with `uvicorn backend.app.main:app`. The database schema is created and
changed only by Alembic (`alembic -c backend/alembic.ini upgrade head`), never at startup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.config import Settings, get_settings
from backend.app.routers import health, laps, leaderboard, models, replays, tracks
from lapzero.contract import CONTRACT_VERSION

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Assemble the API.

    The database engine is created when the app starts and disposed when it stops; requests get
    sessions through backend.app.dependencies.get_session.

    Args:
        settings: Backend settings; defaults to get_settings() (environment and .env).

    Returns:
        The app, with every router under /api.
    """
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(app_settings.database_url, pool_pre_ping=True)
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
        logger.info(
            "LapZero API starting: environment=%s contract_version=%s",
            app_settings.environment,
            CONTRACT_VERSION,
        )
        try:
            yield
        finally:
            engine.dispose()
            logger.info("LapZero API stopped")

    app = FastAPI(title="LapZero", lifespan=lifespan)
    app.state.settings = app_settings

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes every invalid value: NaN and inf then fail to encode
        # (a 500 instead of a 422), and an invalid upload is repeated back in full.
        detail = [
            {"loc": issue["loc"], "msg": issue["msg"], "type": issue["type"]}
            for issue in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    for module in (health, tracks, laps, replays, leaderboard, models):
        app.include_router(module.router)
    return app


app = create_app()
