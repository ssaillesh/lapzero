"""Initial schema: tracks, laps and replays (contract Section 6.1).

Revision ID: 0001
Revises:
Create Date: 2026-10-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _enum(name: str, *values: str) -> sa.Enum:
    """VARCHAR(16) column type with a CHECK constraint, as backend.app.models.domain._enum."""
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True, length=16)


def upgrade() -> None:
    """Create the tracks, laps and replays tables."""
    op.create_table(
        "tracks",
        sa.Column("track_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("length_m", sa.Float(), nullable=False),
        sa.Column("metadata_json", _JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("track_id", name=op.f("pk_tracks")),
    )
    op.create_table(
        "laps",
        sa.Column("lap_id", sa.String(length=26), nullable=False),
        sa.Column("episode_id", sa.String(length=26), nullable=False),
        sa.Column("lap_index", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.String(), nullable=False),
        sa.Column("driver_type", _enum("driver_type", "human", "model"), nullable=False),
        sa.Column("driver_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(), nullable=True),
        sa.Column(
            "human_input_device", _enum("human_input_device", "keyboard", "gamepad"), nullable=True
        ),
        sa.Column(
            "status", _enum("status", "completed", "crashed", "timeout", "stuck"), nullable=False
        ),
        sa.Column("lap_time_s", sa.Float(), nullable=True),
        sa.Column("collision", sa.Boolean(), nullable=False),
        sa.Column("off_track_events", sa.Integer(), nullable=False),
        sa.Column("max_speed_mps", sa.Float(), nullable=False),
        sa.Column("mean_speed_mps", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("telemetry_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["track_id"], ["tracks.track_id"], name=op.f("fk_laps_track_id_tracks")
        ),
        sa.PrimaryKeyConstraint("lap_id", name=op.f("pk_laps")),
    )
    op.create_index(
        op.f("ix_laps_track_id_status_lap_time_s"),
        "laps",
        ["track_id", "status", "lap_time_s"],
        unique=False,
    )
    op.create_table(
        "replays",
        sa.Column("lap_id", sa.String(length=26), nullable=False),
        sa.Column("replay_json", _JSON, nullable=False),
        sa.ForeignKeyConstraint(
            ["lap_id"], ["laps.lap_id"], name=op.f("fk_replays_lap_id_laps"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("lap_id", name=op.f("pk_replays")),
    )


def downgrade() -> None:
    """Drop the tables created by upgrade."""
    op.drop_table("replays")
    op.drop_index(op.f("ix_laps_track_id_status_lap_time_s"), table_name="laps")
    op.drop_table("laps")
    op.drop_table("tracks")
