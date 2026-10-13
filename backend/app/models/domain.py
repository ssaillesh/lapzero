"""Tables of the backend (contract Section 6.1): tracks, laps and lap replays.

Model versions are not stored yet; GET /api/models/champion reads the MLflow registry directly.
Schema changes go through Alembic migrations in backend/alembic/versions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from sqlalchemy import Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, JsonObject
from lapzero.io.replay import driver_label
from lapzero.io.telemetry import DriverType, InputDevice, LapStatus

ULID_LENGTH: Final = 26
ENUM_LENGTH: Final = 16  # VARCHAR length of enum columns, with room for longer values


def _enum(enum: type[StrEnum], name: str) -> Enum:
    """A VARCHAR column type storing `enum` values, with a CHECK constraint named `name`."""
    return Enum(
        enum,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=ENUM_LENGTH,
        values_callable=lambda members: [member.value for member in members],
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Track(Base):
    """A track that at least one uploaded lap was driven on.

    Attributes:
        track_id: Track ID from the telemetry, also the directory name under TRACKS_DIR.
        name: Display name; the track ID until someone names it.
        length_m: Centreline length from the first uploaded lap, metres.
        metadata_json: Free-form display metadata.
        created_at: When the track was first seen (UTC).
    """

    __tablename__ = "tracks"

    track_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    length_m: Mapped[float]
    metadata_json: Mapped[JsonObject] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now)


class Lap(Base):
    """One uploaded lap: the driver and result fields of its telemetry (contract Section 5).

    Attributes:
        telemetry_path: The telemetry JSON file, relative to LAPS_DIR: <track_id>/<lap_id>.json.
        created_at: When the lap was recorded (telemetry created_at, UTC).
        Others: As the telemetry fields of the same name; times in s, speeds in m/s.
    """

    __tablename__ = "laps"
    __table_args__ = (Index(None, "track_id", "status", "lap_time_s"),)

    lap_id: Mapped[str] = mapped_column(String(ULID_LENGTH), primary_key=True)
    episode_id: Mapped[str] = mapped_column(String(ULID_LENGTH))
    lap_index: Mapped[int]
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.track_id"))
    driver_type: Mapped[DriverType] = mapped_column(_enum(DriverType, "driver_type"))
    driver_name: Mapped[str]
    model_version: Mapped[str | None]
    mlflow_run_id: Mapped[str | None]
    human_input_device: Mapped[InputDevice | None] = mapped_column(
        _enum(InputDevice, "human_input_device")
    )
    status: Mapped[LapStatus] = mapped_column(_enum(LapStatus, "status"))
    lap_time_s: Mapped[float | None]
    collision: Mapped[bool]
    off_track_events: Mapped[int]
    max_speed_mps: Mapped[float]
    mean_speed_mps: Mapped[float]
    sample_count: Mapped[int]
    contract_version: Mapped[str]
    telemetry_path: Mapped[str]
    created_at: Mapped[datetime]

    @property
    def driver_label(self) -> str:
        """Label the dashboard shows for the driver, e.g. "AI v7" (lapzero.io.replay)."""
        return driver_label(self.driver_type, self.driver_name, self.model_version)


class LapReplay(Base):
    """The contract Section 6 replay JSON of a lap, generated when the lap is ingested."""

    __tablename__ = "replays"

    lap_id: Mapped[str] = mapped_column(
        String(ULID_LENGTH), ForeignKey("laps.lap_id", ondelete="CASCADE"), primary_key=True
    )
    replay_json: Mapped[JsonObject]
