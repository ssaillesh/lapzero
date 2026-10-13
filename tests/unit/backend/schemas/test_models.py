"""Unit tests for backend/app/schemas/models.py."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas.models import ChampionResponse
from backend.app.services.models import ChampionVersion


def _champion(created_at: datetime) -> ChampionVersion:
    return ChampionVersion(
        model_name="lapzero-policy",
        version="8",
        run_id="3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
        contract_version="1.1.0",
        tags={"gate_result": "promote"},
        created_at=created_at,
    )


def test_champion_response_is_built_from_the_service_result() -> None:
    response = ChampionResponse.model_validate(_champion(datetime(2026, 10, 1, tzinfo=UTC)))

    assert response.model_dump(mode="json") == {
        "model_name": "lapzero-policy",
        "version": "8",
        "run_id": "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
        "contract_version": "1.1.0",
        "tags": {"gate_result": "promote"},
        "created_at": "2026-10-01T00:00:00Z",
    }


def test_champion_response_refuses_a_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="created_at"):
        ChampionResponse.model_validate(_champion(datetime(2026, 10, 1)))
