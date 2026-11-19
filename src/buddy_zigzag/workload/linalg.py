"""Extract tensor matmul semantics using Buddy's structured MLIR Python API."""

from dataclasses import dataclass

from .errors import WorkloadError
from .identity import LAYER_ID
from .schema import Layer, Operand, TensorSpec, Workload

MATMUL_OPS = {"linalg.matmul", "linalg.batch_matmul"}


@dataclass(frozen=True)
class _Value:
    kind: str
    name: str
    layer_id: int | None = None
    zero: bool = False


class LinalgExtractor:
    """Read the supported Linalg subset through native MLIR bindings."""

    def extract(self, text: str) -> Workload:
        try:
            from buddy_mlir import ir

            # Register the dialect Python wrappers before parsing.
            from buddy_mlir.dialects import arith, func, linalg, tensor  # noqa: F401
        except ImportError as exc:
            raise WorkloadError(
                "Buddy MLIR bindings are unavailable. Configure .env with BUDDY_MLIR_BUILD_DIR or add "
                "buddy-mlir/build/python_packages to PYTHONPATH with the matching Python interpreter."
            ) from exc

        def fail(op, message):
            raise WorkloadError(f"{op.name} at {op.location}: {message}")

        def tensor_spec(value, op):
            if not isinstance(value.type, ir.RankedTensorType):
                fail(op, "Only ranked tensor operands are supported; export before bufferization")
            shape = tuple(value.type.shape)
            if not shape or any(size <= 0 for size in shape):
                fail(op, "Only positive static tensor dimensions are supported")
            element = value.type.element_type
            dtype = str(element)
            if isinstance(element, ir.IntegerType):
                precision = element.width
            elif dtype in {"f16", "bf16", "f32", "f64"}:
                precision = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}[dtype]
            else:
                fail(op, f"Unsupported element type {dtype}")
            return TensorSpec(shape=shape, dtype=dtype, precision=precision)

        def is_zero(attr):
            if isinstance(attr, (ir.FloatAttr, ir.IntegerAttr)):
                return attr.value == 0
            if isinstance(attr, ir.DenseElementsAttr) and attr.is_splat:
                return is_zero(attr.get_splat_value())
            return False

        layers = []
        auxiliary_operations = []
        outputs = []
        values = {}
        with ir.Context():
            try:
                module = ir.Module.parse(text)
            except ir.MLIRError as exc:
                raise WorkloadError(f"Invalid MLIR: {exc}") from exc
            functions = list(module.body.operations)
            if len(functions) != 1 or functions[0].operation.name != "func.func":
                raise WorkloadError("Expected a module containing exactly one function")
            fn = functions[0].operation
            if len(fn.regions[0].blocks) != 1:
                fail(fn, "Expected one function block; control flow and declarations are unsupported")
            block = fn.regions[0].blocks[0]
            for index, arg in enumerate(block.arguments):
                tensor_spec(arg, fn)
                values[arg] = _Value("input", f"arg{index}")
            for index, view in enumerate(block.operations):
                op = view.operation
                name = op.name
                if name == "arith.constant":
                    values[op.results[0]] = _Value("constant", f"constant{index}", zero=is_zero(op.attributes["value"]))
                    role = "constant; zero constants may initialize accumulators"
                elif name == "tensor.empty":
                    tensor_spec(op.results[0], op)
                    values[op.results[0]] = _Value("empty", f"empty{index}")
                    role = "uninitialized destination allocation"
                elif name == "linalg.fill":
                    if len(op.operands) != 2 or len(op.results) != 1:
                        fail(op, "Only tensor fill is supported")
                    source = values.get(op.operands[0])
                    if source is None or source.kind != "constant" or not source.zero:
                        fail(op, "Only constant zero fill is supported")
                    tensor_spec(op.results[0], op)
                    values[op.results[0]] = _Value("constant", f"zero_fill{index}", zero=True)
                    role = "zero initialization; no separate ZigZag compute layer"
                elif name in MATMUL_OPS:
                    if len(op.operands) != 3 or len(op.results) != 1:
                        fail(op, "Expected two tensor inputs, one destination, and one result")
                    init = values.get(op.operands[2])
                    if init is None or not init.zero:
                        fail(op, "Matmul destination must be initialized with constant zero")
                    inputs = [values.get(value) for value in op.operands[:2]]
                    if any(value is None or value.kind == "empty" for value in inputs):
                        fail(op, "Input has an unknown producer or is uninitialized")
                    specs = [tensor_spec(value, op) for value in op.operands]
                    if tensor_spec(op.results[0], op) != specs[2]:
                        fail(op, "Result type differs from destination type")
                    maps = []
                    n_dims = None
                    for attr in op.attributes["indexing_maps"]:
                        affine_map = ir.AffineMapAttr(attr).value
                        if affine_map.n_symbols or any(
                            not isinstance(expr, ir.AffineDimExpr) for expr in affine_map.results
                        ):
                            fail(op, "Only dimension permutation indexing maps are supported")
                        if n_dims is not None and n_dims != affine_map.n_dims:
                            fail(op, "Indexing maps have different loop ranks")
                        n_dims = affine_map.n_dims
                        maps.append(tuple(expr.position for expr in affine_map.results))
                    try:
                        dims, sizes, accesses = matmul_dimensions(specs, maps, n_dims, "batch_matmul" in name)
                    except WorkloadError as exc:
                        fail(op, str(exc))
                    layer_id = (
                        ir.IntegerAttr(op.attributes[LAYER_ID]).value if LAYER_ID in op.attributes else len(layers)
                    )
                    layer_name = f"{name.removeprefix('linalg.')}_{layer_id}"
                    operands = {
                        label: Operand(
                            tensor=specs[i],
                            indices=accesses[i],
                            source_kind=source.kind,
                            source_name=source.name,
                            source_id=source.layer_id,
                        )
                        for i, (label, source) in enumerate(zip(("I", "W"), inputs))
                    }
                    operands["O"] = Operand(
                        tensor=specs[2],
                        indices=accesses[2],
                        source_kind="layer",
                        source_name=f"{layer_name}:O",
                        source_id=layer_id,
                    )
                    layers.append(
                        Layer(
                            id=layer_id,
                            name=layer_name,
                            mlir_op=name,
                            location=str(op.location),
                            loop_dims=dims,
                            loop_sizes=sizes,
                            parallel_dims=dims[:-1],
                            reduction_dims=("K",),
                            operands=operands,
                        )
                    )
                    values[op.results[0]] = _Value("layer", f"{layer_name}:O", layer_id)
                    # Named matmul regions describe the multiply/reduce itself; do not
                    # traverse their scalar arithmetic as additional compute layers.
                    continue
                elif name == "func.return":
                    for value in op.operands:
                        source = values.get(value)
                        if source is None or source.kind == "empty":
                            fail(op, "Return value has an unsupported producer")
                        outputs.append(source.name)
                    continue
                else:
                    fail(op, "Unsupported operation; no computation is silently skipped")
                auxiliary_operations.append({"operation": name, "location": str(op.location), "role": role})
        if not layers:
            raise WorkloadError("No supported matmul operations found")
        return Workload(layers=layers, auxiliary_operations=auxiliary_operations, outputs=outputs)


def matmul_dimensions(
    specs: list[TensorSpec], maps: list[tuple[int, ...]], n_dims: int, batched: bool
) -> tuple[tuple[str, ...], tuple[int, ...], list[tuple[str, ...]]]:
    parallel = ("B", "M", "N") if batched else ("M", "N")
    dims = parallel + ("K",)
    if n_dims != len(dims) or len(maps) != 3:
        raise WorkloadError("Expected three matmul indexing maps with one reduction dimension")
    output_axes = maps[2]
    if len(output_axes) != len(parallel) or len(set(output_axes)) != len(output_axes):
        raise WorkloadError("Unexpected matmul output indexing map")
    if any(axis < 0 or axis >= n_dims for axes in maps for axis in axes):
        raise WorkloadError("Indexing map references an invalid loop dimension")
    reductions = set(range(n_dims)) - set(output_axes)
    if len(reductions) != 1:
        raise WorkloadError("Expected exactly one matmul reduction dimension")
    names = dict(zip(output_axes, parallel))
    names[next(iter(reductions))] = "K"
    accesses = [tuple(names[axis] for axis in axes) for axes in maps]
    batch = {"B"} if batched else set()
    expected = [batch | {"M", "K"}, batch | {"K", "N"}, set(parallel)]
    sizes = {}
    for spec, access, required in zip(specs, accesses, expected):
        if len(spec.shape) != len(access) or len(access) != len(set(access)) or set(access) != required:
            raise WorkloadError("Unsupported matmul access pattern or operand rank (broadcasting is not supported)")
        for dim, extent in zip(access, spec.shape):
            if extent <= 0:
                raise WorkloadError("Only positive static dimensions are supported")
            if dim in sizes and sizes[dim] != extent:
                raise WorkloadError(f"Inconsistent extent for {dim}: {sizes[dim]} versus {extent}")
            sizes[dim] = extent
    return dims, tuple(sizes[dim] for dim in dims), accesses
