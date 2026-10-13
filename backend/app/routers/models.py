"""GET /api/models/champion: the model version the release gate last promoted."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.dependencies import SettingsDep
from backend.app.schemas.models import ChampionResponse
from backend.app.services import models

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/champion")
def get_champion(settings: SettingsDep) -> ChampionResponse:
    """Return the champion version from the MLflow model registry.

    Responds 404 if the model is not registered or has no champion, and 503 if the registry
    cannot be read.
    """
    try:
        champion = models.get_champion(settings.mlflow_tracking_uri, settings.mlflow_model_name)
    except models.ChampionNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    except models.RegistryUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    return ChampionResponse.model_validate(champion)
