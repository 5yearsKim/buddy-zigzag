"""Test-only functional execution of emitted WS commands, without accelerator timing."""

import ctypes

import numpy as np


class WsCommandEmulator:
    def __init__(self, dim):
        self.dim = dim
        self.state = {}
        self.accumulator = None
        self.calls = self.loads = self.stores = 0
        self.errors = []

    def matrix(self, pointer, rows, columns, stride):
        flat = np.ctypeslib.as_array((ctypes.c_float * ((rows - 1) * stride + columns)).from_address(pointer))
        return np.lib.stride_tricks.as_strided(flat, shape=(rows, columns), strides=(stride * 4, 4))

    def command(self, name, first, second):
        try:
            if name in ("config", "flush"):
                return
            if name.startswith("loop_ws_config_"):
                self.state[name.removeprefix("loop_ws_config_")] = (first, second)
                return
            assert name == "loop_ws"
            assert first & 2 and not (second & 3), "expected full-C, untransposed FP32 WS"
            padding, bounds = self.state["bounds"]
            rows, cols, depth = [
                ((bounds >> shift) & 65535) * self.dim - ((padding >> shift) & 65535) for shift in (0, 16, 32)
            ]
            a, b = self.state["addrs_ab"]
            d, c = self.state["addrs_dc"]
            sa, sb = self.state["strides_ab"]
            sd, sc = self.state["strides_dc"]
            if d:
                assert self.accumulator is None, "D must load only on the first K block"
                self.accumulator = self.matrix(d, rows, cols, sd).copy()
                self.loads += 1
            else:
                assert first & 1 and self.accumulator is not None, "later K blocks must retain partial sums"
            self.accumulator += self.matrix(a, rows, depth, sa) @ self.matrix(b, depth, cols, sb)
            self.calls += 1
            if c:
                self.matrix(c, rows, cols, sc)[:] = self.accumulator
                self.accumulator = None
                self.stores += 1
        except Exception as exc:
            self.errors.append(exc)

    def execute(self, text, entry, arrays, output, runner_library):
        from buddy_mlir import ir
        from buddy_mlir.dialects import llvm
        from buddy_mlir.execution_engine import ExecutionEngine
        from buddy_mlir.runtime import get_ranked_memref_descriptor

        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(text)
            commands = []

            def collect(op):
                if op.name.startswith("gemmini.intr."):
                    commands.append(op)
                return ir.WalkResult.ADVANCE

            module.operation.walk(collect)
            names = {op.name.removeprefix("gemmini.intr.") for op in commands}
            callbacks = {}
            for name in names:
                symbol = f"test_ws_{name}"
                declaration = ir.Module.parse(f"module {{ llvm.func @{symbol}(i64, i64) }}")
                module.body.append(declaration.body.operations[0].operation.clone())
                callbacks[symbol] = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64)(
                    lambda first, second, name=name: self.command(name, first, second)
                )
            for op in commands:
                with ir.InsertionPoint(op):
                    llvm.CallOp(
                        None, list(op.operands), [], [], callee=f"test_ws_{op.name.removeprefix('gemmini.intr.')}"
                    )
                op.erase()
            assert module.operation.verify()
            engine = ExecutionEngine(module, opt_level=2, shared_libs=[str(runner_library)])
            for name, callback in callbacks.items():
                engine.raw_register_runtime(name, ctypes.cast(callback, ctypes.c_void_p))
            descriptors = [get_ranked_memref_descriptor(value) for value in [*arrays, output]]
            engine.invoke(entry, *[ctypes.pointer(ctypes.pointer(value)) for value in descriptors])
        assert not self.errors, str(self.errors)
        assert self.accumulator is None
