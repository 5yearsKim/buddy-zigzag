"""Load validated case descriptions and construct reproducible Torch inputs."""

from importlib.resources import files
from pathlib import Path

import yaml

from ..exporter.errors import ExportError
from ..exporter.schema import WorkloadCase


class CaseRepository:
    def load(self, name: str, path: Path | None = None) -> WorkloadCase:
        source = path or files("buddy_zigzag.workloads").joinpath("cases.yaml")
        cases = yaml.safe_load(source.read_text())
        if not isinstance(cases, dict) or name not in cases:
            raise ExportError(f"Unknown case {name!r} in {source}")
        return WorkloadCase.model_validate(cases[name])


class TorchWorkloadFactory:
    def build(self, spec: WorkloadCase):
        try:
            import torch

            from .matmul import BatchMatmul, Matmul, MatmulChain
        except ImportError as exc:
            raise ExportError("Torch is unavailable; run uv sync or export saved MLIR with --mlir") from exc
        models = {"matmul": Matmul, "batch_matmul": BatchMatmul, "matmul_chain": MatmulChain}
        generator = torch.Generator().manual_seed(spec.seed)
        dtype = getattr(torch, spec.dtype)
        inputs = tuple(torch.randn(shape, dtype=dtype, generator=generator) for shape in spec.shapes)
        model = models[spec.kind]().eval()
        with torch.no_grad():
            output = model(*inputs)
        return model, inputs, (output,)
