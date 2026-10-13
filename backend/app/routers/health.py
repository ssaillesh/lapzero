"""GET /api/health."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.health import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> HealthResponse:
    """Report that the API is up and which contract version it validates uploads against."""
    return HealthResponse()
