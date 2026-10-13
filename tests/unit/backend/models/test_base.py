"""Unit tests for backend/app/models/base.py: UTC datetimes, JSON columns, constraint names."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from backend.app.models.base import UTCDateTime
from backend.app.models.domain import Lap, Track

_PLUS_TWO = timezone(timedelta(hours=2))


def test_aware_datetime_is_stored_as_utc_and_loaded_utc_aware(session: Session) -> None:
    track = Track(
        track_id="t",
        name="t",
        length_m=1.0,
        created_at=datetime(2026, 10, 20, 16, 32, tzinfo=_PLUS_TWO),
    )
    session.add(track)
    session.flush()
    session.expire(track)

    assert track.created_at == datetime(2026, 10, 20, 14, 32, tzinfo=UTC)
    assert track.created_at.tzinfo is UTC


def test_naive_datetime_is_refused(session: Session) -> None:
    session.add(Track(track_id="t", name="t", length_m=1.0, created_at=datetime(2026, 10, 20)))

    with pytest.raises(StatementError, match="has no timezone"):
        session.flush()


def test_datetime_loaded_with_an_offset_is_converted_to_utc() -> None:
    loaded = UTCDateTime().process_result_value(
        datetime(2026, 10, 20, 16, 32, tzinfo=_PLUS_TWO), postgresql.dialect()
    )

    assert loaded == datetime(2026, 10, 20, 14, 32, tzinfo=UTC)
    assert loaded is not None and loaded.tzinfo is UTC


def test_null_datetime_passes_through() -> None:
    column_type = UTCDateTime()

    assert column_type.process_bind_param(None, sqlite.dialect()) is None
    assert column_type.process_result_value(None, sqlite.dialect()) is None


def test_json_columns_are_jsonb_on_postgresql_and_json_elsewhere() -> None:
    postgres_ddl = str(CreateTable(Track.__table__).compile(dialect=postgresql.dialect()))
    sqlite_ddl = str(CreateTable(Track.__table__).compile(dialect=sqlite.dialect()))

    assert "metadata_json JSONB NOT NULL" in postgres_ddl
    assert "metadata_json JSON NOT NULL" in sqlite_ddl


def test_constraints_and_indexes_follow_the_naming_convention() -> None:
    table = Lap.__table__
    foreign_keys = [c.name for c in table.constraints if isinstance(c, ForeignKeyConstraint)]

    assert table.primary_key.name == "pk_laps"
    assert foreign_keys == ["fk_laps_track_id_tracks"]
    assert {index.name for index in table.indexes} == {"ix_laps_track_id_status_lap_time_s"}
