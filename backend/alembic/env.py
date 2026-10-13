"""Alembic environment: migrates the database at DATABASE_URL (backend/app/config.py).

A URL set on the Alembic config (config.set_main_option("sqlalchemy.url", ...), as the
integration tests do) takes precedence over the settings.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import backend.app.models.domain  # noqa: F401 - registers the tables on Base.metadata
from backend.app.config import get_settings
from backend.app.models.base import Base

config = context.config
if config.config_file_name is not None:
    # Keep loggers configured before Alembic runs (e.g. pytest's) working.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    """Return the URL to migrate: the Alembic config's, else the backend settings'."""
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Write the migration SQL to stdout instead of running it (alembic upgrade --sql)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migrations on a live connection."""
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
