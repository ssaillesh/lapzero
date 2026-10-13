"""Base of every API response model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ResponseModel(BaseModel):
    """API response: built from ORM rows or service results; NaN, inf and unknown fields fail."""

    model_config = ConfigDict(
        from_attributes=True, extra="forbid", allow_inf_nan=False, frozen=True
    )
