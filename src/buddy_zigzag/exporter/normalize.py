"""Normalize named matmul indexing maps without depending on MLIR bindings."""

from .errors import ExportError
from .schema import TensorSpec


def matmul_dimensions(
    specs: list[TensorSpec], maps: list[tuple[int, ...]], n_dims: int, batched: bool
) -> tuple[tuple[str, ...], tuple[int, ...], list[tuple[str, ...]]]:
    parallel = ("B", "M", "N") if batched else ("M", "N")
    dims = parallel + ("K",)
    if n_dims != len(dims) or len(maps) != 3:
        raise ExportError("Expected three matmul indexing maps with one reduction dimension")
    output_axes = maps[2]
    if len(output_axes) != len(parallel) or len(set(output_axes)) != len(output_axes):
        raise ExportError("Unexpected matmul output indexing map")
    if any(axis < 0 or axis >= n_dims for axes in maps for axis in axes):
        raise ExportError("Indexing map references an invalid loop dimension")
    reductions = set(range(n_dims)) - set(output_axes)
    if len(reductions) != 1:
        raise ExportError("Expected exactly one matmul reduction dimension")
    names = dict(zip(output_axes, parallel))
    names[next(iter(reductions))] = "K"
    accesses = [tuple(names[axis] for axis in axes) for axes in maps]
    batch = {"B"} if batched else set()
    expected = [batch | {"M", "K"}, batch | {"K", "N"}, set(parallel)]
    sizes = {}
    for spec, access, required in zip(specs, accesses, expected):
        if len(spec.shape) != len(access) or len(access) != len(set(access)) or set(access) != required:
            raise ExportError("Unsupported matmul access pattern or operand rank (broadcasting is not supported)")
        for dim, extent in zip(access, spec.shape):
            if extent <= 0:
                raise ExportError("Only positive static dimensions are supported")
            if dim in sizes and sizes[dim] != extent:
                raise ExportError(f"Inconsistent extent for {dim}: {sizes[dim]} versus {extent}")
            sizes[dim] = extent
    return dims, tuple(sizes[dim] for dim in dims), accesses
