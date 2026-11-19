"""The only adapter that depends on Buddy's Torch frontend API."""

from .errors import WorkloadError


class TorchFrontend:
    """Adapt Buddy's Torch frontend to textual Linalg MLIR."""

    def import_model(self, model, inputs: tuple) -> str:
        try:
            import torch
            from buddy.compiler.frontend import DynamoCompiler
            from buddy.compiler.ops import linalg
        except ImportError as exc:
            raise WorkloadError("Torch/Buddy frontend is unavailable. Configure .env and run uv sync.") from exc
        compiler = DynamoCompiler(primary_registry=linalg.ops_registry)
        with torch.no_grad():
            graphs = compiler.importer(model.eval(), *inputs)
        if len(graphs) != 1:
            raise WorkloadError(
                f"Expected one complete Buddy graph, received {len(graphs)}; graph breaks are unsupported"
            )
        graph = graphs[0]
        graph.lower_to_top_level_ir()
        # Buddy currently exposes the generated module through this private member.
        # Keep access isolated here so frontend API changes require one adapter edit.
        return graph._imported_module.operation.get_asm(enable_debug_info=True)
