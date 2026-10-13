"""Unit tests for backend/app/schemas/base.py: the API response base model."""

import math
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from backend.app.schemas.base import ResponseModel


class _Speed(ResponseModel):
    driver: str
    speed_mps: float


@dataclass
class _Row:
    driver: str
    speed_mps: float
    internal: str = "not exposed"


def test_response_is_built_from_attributes_of_any_object() -> None:
    assert _Speed.model_validate(_Row("sam", 6.1)) == _Speed(driver="sam", speed_mps=6.1)


@pytest.mark.parametrize("speed", [math.nan, math.inf, -math.inf])
def test_response_never_carries_nan_or_inf(speed: float) -> None:
    with pytest.raises(ValidationError):
        _Speed(driver="sam", speed_mps=speed)


def test_response_rejects_unknown_fields_and_is_immutable() -> None:
    with pytest.raises(ValidationError):
        _Speed.model_validate({"driver": "sam", "speed_mps": 6.1, "extra": 1})
    speed = _Speed(driver="sam", speed_mps=6.1)
    with pytest.raises(ValidationError):
        speed.speed_mps = 7.0  # type: ignore[misc]
