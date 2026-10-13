"""Unit tests for backend/app/dependencies.py: per-request settings and sessions."""

from fastapi import FastAPI, Request
from sqlalchemy import Engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.config import Settings
from backend.app.dependencies import get_app_settings, get_session
from backend.app.main import create_app


def _request(app: FastAPI) -> Request:
    return Request({"type": "http", "app": app})


def test_requests_get_the_settings_the_app_was_created_with(api_settings: Settings) -> None:
    app = create_app(api_settings)

    assert get_app_settings(_request(app)) is api_settings


def test_request_session_comes_from_the_app_and_is_closed_afterwards(engine: Engine) -> None:
    app = FastAPI()
    app.state.session_factory = sessionmaker(engine)

    sessions = get_session(_request(app))
    session = next(sessions)
    session.execute(text("SELECT 1"))

    assert session.get_bind() is engine
    assert session.in_transaction()
    sessions.close()
    assert not session.in_transaction()
