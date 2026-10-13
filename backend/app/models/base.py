"""Declarative base, constraint naming and column types shared by every table."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from sqlalchemy import JSON, DateTime, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

JsonObject = dict[str, object]
"""A JSON object column: JSONB on PostgreSQL, JSON text elsewhere (tests)."""

NAMING_CONVENTION: Final = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Constraint names, so Alembic migrations can refer to them on every database."""


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware datetime stored in UTC.

    Naive datetimes are refused, so a local time is never stored by mistake. Loaded values are
    always UTC-aware, also on SQLite, which drops the offset PostgreSQL keeps.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Convert an aware datetime to UTC before it is stored.

        Raises:
            ValueError: If the datetime has no timezone.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"datetime {value.isoformat()} has no timezone; store UTC")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Return a loaded datetime as UTC-aware."""
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base of every LapZero table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this mapping off the class
        datetime: UTCDateTime(),
        JsonObject: JSON().with_variant(JSONB(), "postgresql"),
    }
