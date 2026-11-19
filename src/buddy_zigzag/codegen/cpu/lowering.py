"""Schedule-preserving CPU lowering and native numerical verification."""

import ctypes

from ...common.schema import Schema
from ...schedule.errors import ScheduleError
from ...settings import Settings
from ...workload.schema import Layer
from ..utils import run_tool

BUFFERIZE_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
    "buffer-results-to-out-params{modify-public-functions=true},"
    "buffer-deallocation-pipeline,canonicalize)"
)
LLVM_PIPELINE = (
    "builtin.module(convert-linalg-to-loops,expand-strided-metadata,lower-affine,convert-scf-to-cf,"
    "func.func(llvm-request-c-wrappers),convert-arith-to-llvm,finalize-memref-to-llvm,"
    "convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
)


class LoweredCode(Schema):
    entry_point: str
    bufferized_mlir: str
    llvm_mlir: str
    llvm_ir: str


class CpuLowerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def lower(self, scheduled: str, layer: Layer) -> LoweredCode:
        from buddy_mlir import ir

        if self.settings.buddy_mlir_build_dir is None or self.settings.llvm_mlir_build_dir is None:
            raise ScheduleError("BUDDY_MLIR_BUILD_DIR and LLVM_MLIR_BUILD_DIR must be configured")
        tool = self.settings.buddy_mlir_build_dir / "bin/buddy-opt"
        bufferized = run_tool(tool, [f"--pass-pipeline={BUFFERIZE_PIPELINE}"], scheduled)
        # Validate the native ABI before producing code that Python may invoke.
        with ir.Context():
            module = ir.Module.parse(bufferized)
            functions = [view.operation for view in module.body.operations if view.operation.name == "func.func"]
            if len(functions) != 1:
                raise ScheduleError("CPU verification requires exactly one function")
            fn = functions[0]
            function_type = ir.FunctionType(ir.TypeAttr(fn.attributes["function_type"]).value)
            if len(function_type.inputs) != 3 or function_type.results:
                raise ScheduleError("Expected two input memrefs and one output memref, with no returned values")
            specs = {layer.operands[name].source_name: layer.operands[name].tensor for name in ("I", "W")}
            expected = [specs["arg0"], specs["arg1"], layer.operands["O"].tensor]
            for actual, spec in zip(function_type.inputs, expected):
                if (
                    not isinstance(actual, ir.MemRefType)
                    or tuple(actual.shape) != spec.shape
                    or str(actual.element_type) != "f32"
                ):
                    raise ScheduleError("Bufferized entry point does not match the expected FP32 memref ABI")
            entry_point = ir.StringAttr(fn.attributes["sym_name"]).value
        llvm_mlir = run_tool(tool, [f"--pass-pipeline={LLVM_PIPELINE}"], bufferized)
        llvm_ir = run_tool(self.settings.llvm_mlir_build_dir / "bin/mlir-translate", ["--mlir-to-llvmir"], llvm_mlir)
        return LoweredCode(entry_point=entry_point, bufferized_mlir=bufferized, llvm_mlir=llvm_mlir, llvm_ir=llvm_ir)

    def verify(self, code: LoweredCode, layer: Layer, *, trials: int = 3) -> dict:
        import numpy as np
        import torch
        from buddy_mlir import ir
        from buddy_mlir.execution_engine import ExecutionEngine
        from buddy_mlir.runtime import get_ranked_memref_descriptor

        if not 1 <= trials <= 100:
            raise ScheduleError("Verification trials must be between 1 and 100")
        library = self.settings.llvm_mlir_build_dir / "lib/libmlir_c_runner_utils.so"
        if not library.is_file():
            raise ScheduleError(f"MLIR runner library is unavailable: {library}")
        specs = {layer.operands[name].source_name: layer.operands[name].tensor for name in ("I", "W")}
        max_error = 0.0
        with ir.Context():
            module = ir.Module.parse(code.llvm_mlir)
            engine = ExecutionEngine(module, opt_level=2, shared_libs=[str(library)])
            for seed in range(trials):
                rng = np.random.default_rng(seed)
                arrays = [rng.standard_normal(specs[f"arg{i}"].shape).astype(np.float32) for i in range(2)]
                output = np.full(layer.operands["O"].tensor.shape, np.nan, dtype=np.float32)
                values = {f"arg{i}": torch.from_numpy(value) for i, value in enumerate(arrays)}
                reference = values[layer.operands["I"].source_name] @ values[layer.operands["W"].source_name]
                descriptors = [get_ranked_memref_descriptor(value) for value in [*arrays, output]]
                pointers = [ctypes.pointer(ctypes.pointer(descriptor)) for descriptor in descriptors]
                engine.invoke(code.entry_point, *pointers)
                try:
                    torch.testing.assert_close(torch.from_numpy(output), reference, rtol=1e-4, atol=1e-5)
                except AssertionError as exc:
                    raise ScheduleError(f"Native generated code differs from Torch: {exc}") from exc
                max_error = max(max_error, float(np.max(np.abs(output - reference.numpy()))))
        return {
            "status": "passed",
            "method": "native LLVM JIT versus Torch matmul",
            "trials": trials,
            "max_abs_error": max_error,
        }
