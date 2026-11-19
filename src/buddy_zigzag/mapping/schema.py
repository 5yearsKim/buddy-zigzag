"""Capabilities of compiler backends supported by mapping rules."""

from typing import Annotated

from pydantic import Field

from ..common.schema import PositiveInt, Schema


class BackendCapabilities(Schema):
    name: Annotated[str, Field(min_length=1)]
    operations: Annotated[tuple[str, ...], Field(min_length=1)]
    dtypes: Annotated[tuple[str, ...], Field(min_length=1)]
    dataflow: str
    max_layers: PositiveInt


GEMMINI_WS = BackendCapabilities(
    name="gemmini_ws",
    operations=("linalg.matmul",),
    dtypes=("f32",),
    dataflow="WS",
    max_layers=1,
)
