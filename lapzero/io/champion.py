"""Champion snapshot: the registry's champion version saved as JSON, for an API without MLflow.

scripts/export_champion.py writes it from the local registry (lapzero.registry.export_champion),
and the backend serves it from GET /api/models/champion when CHAMPION_SNAPSHOT names the file.
It holds registry metadata only, never model weights. Shared by lapzero.registry and the backend,
so this module must never import lapzero.sim or lapzero.training.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from lapzero.io.telemetry import SEMVER_PATTERN


class ChampionSnapshot(BaseModel):
    """The model version a registry's champion alias pointed at when the snapshot was taken.

    Attributes:
        model_name: Registered model name, e.g. "lapzero-policy".
        version: Registry version number, e.g. "8".
        run_id: MLflow run that trained the version; None if it was registered without one.
        contract_version: Contract version the run logged; None if unknown.
        tags: Version tags, e.g. the release gate's gate_result and sweep IDs.
        created_at: When the version was registered (UTC).
        exported_at: When the snapshot was taken (UTC).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+$")
    run_id: str | None
    contract_version: Annotated[str, Field(pattern=SEMVER_PATTERN)] | None
    tags: dict[str, str]
    created_at: AwareDatetime
    exported_at: AwareDatetime

    @field_validator("created_at", "exported_at")
    @classmethod
    def _is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"timestamps must be UTC, got offset {value.utcoffset()}")
        return value


def read_champion_snapshot(path: Path) -> ChampionSnapshot:
    """Read and validate a champion snapshot file.

    Args:
        path: JSON file written by write_champion_snapshot.

    Returns:
        The snapshot.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the content is not a valid snapshot.
    """
    return ChampionSnapshot.model_validate_json(path.read_text())


def write_champion_snapshot(path: Path, snapshot: ChampionSnapshot) -> None:
    """Write a champion snapshot as indented JSON, creating parent directories.

    Args:
        path: File to write; replaced if it exists.
        snapshot: The snapshot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2) + "\n")
