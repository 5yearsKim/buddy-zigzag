"""Shared immutable schema base and strict integer types."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
