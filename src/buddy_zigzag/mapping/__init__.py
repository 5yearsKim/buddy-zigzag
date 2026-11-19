"""Workload-aware generation of legal backend mapping constraints."""

from .generator import MappingConstraintGenerator, MappingError
from .schema import GEMMINI_WS, BackendCapabilities

__all__ = ["GEMMINI_WS", "BackendCapabilities", "MappingError", "MappingConstraintGenerator"]
