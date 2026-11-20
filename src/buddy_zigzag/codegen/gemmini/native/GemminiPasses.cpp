// Project-owned scheduled FP32 Gemmini WS lowering.
// Command encoding follows Buddy-MLIR's Gemmini lowering (Apache License 2.0).
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace {
struct Schedule {
  int64_t m, n, k, dim, bankRows, accRows;
};

FailureOr<Schedule> readSchedule(Operation *op) {
  auto attributes = op->getAttrOfType<DictionaryAttr>("gemmini.schedule");
  if (!attributes || attributes.size() != 7) {
    op->emitOpError("requires a seven-field gemmini.schedule dictionary");
    return failure();
  }
  auto integer = [&](StringRef name) -> int64_t {
    auto value = dyn_cast_or_null<IntegerAttr>(attributes.get(name));
    return value && value.getType().isSignlessInteger(64) ? value.getInt() : -1;
  };
  Schedule s{integer("tile_m"),    integer("tile_n"),    integer("tile_k"),
             integer("array_dim"), integer("bank_rows"), integer("acc_rows")};
  if (s.dim <= 0 || s.dim > 65535 || s.bankRows <= 0 || s.accRows <= 0 ||
      integer("dataflow") != 1) {
    op->emitOpError("requires positive Gemmini dimensions and WS dataflow");
    return failure();
  }
  for (int64_t tile : {s.m, s.n, s.k}) {
    if (tile <= 0 || tile % s.dim || tile / s.dim > 65535) {
      op->emitOpError(
          "tiles must be positive array multiples with 16-bit block counts");
      return failure();
    }
  }
  uint64_t m = s.m / s.dim, n = s.n / s.dim, k = s.k / s.dim;
  if ((m * k + k * n) * s.dim > uint64_t(s.bankRows) * 2 ||
      m * n * s.dim > uint64_t(s.accRows) / 2) {
    op->emitOpError("tiles exceed double-buffered Gemmini memory capacity");
    return failure();
  }
  return s;
}

bool isMatrix(Type type) {
  auto matrix = dyn_cast<MemRefType>(type);
  return matrix && matrix.getRank() == 2 && matrix.hasStaticShape() &&
         matrix.getElementType().isF32();
}

// Use existing registered Gemmini operations by name. No Buddy source headers
// or private compiler APIs are needed; buddy-opt owns the dialect definitions.
Operation *createOperation(OpBuilder &builder, Location loc, StringRef name,
                           ValueRange operands,
                           ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addAttributes(attributes);
  return builder.create(state);
}

class ConvertMatmulPass
    : public PassWrapper<ConvertMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMatmulPass)
  StringRef getArgument() const final {
    return "buddy-zigzag-convert-linalg-to-gemmini";
  }
  StringRef getDescription() const final {
    return "Convert scheduled FP32 Linalg matmul using destination "
           "accumulation";
  }
  void runOnOperation() override {
    if (!getContext().getOrLoadDialect("gemmini")) {
      getOperation().emitError(
          "the host tool must register Buddy's Gemmini dialect");
      return signalPassFailure();
    }
    SmallVector<linalg::MatmulOp> operations;
    getOperation().walk([&](linalg::MatmulOp op) { operations.push_back(op); });
    IRRewriter builder(&getContext());
    for (auto op : operations) {
      if (failed(readSchedule(op)) || op->getNumOperands() != 3 ||
          !llvm::all_of(op->getOperandTypes(), isMatrix)) {
        op.emitOpError(
            "scheduled matmul requires three static FP32 rank-2 buffers");
        return signalPassFailure();
      }
      builder.setInsertionPoint(op);
      SmallVector<NamedAttribute> attrs;
      auto add = [&](StringRef name, Attribute value) {
        attrs.emplace_back(builder.getStringAttr(name), value);
      };
      add("gemmini.schedule", op->getAttr("gemmini.schedule"));
      for (StringRef name :
           {"aScaleFactor", "bScaleFactor", "dScaleFactor", "accScale"})
        add(name, builder.getF32FloatAttr(1));
      add("bertScale", builder.getF32FloatAttr(0));
      for (StringRef name :
           {"repeatingBias", "aTranspose", "bTranspose", "lowD"})
        add(name, builder.getBoolAttr(false));
      add("fullC", builder.getBoolAttr(true));
      add("act", builder.getI64IntegerAttr(0));
      add("weightA", builder.getI64IntegerAttr(0));
      add("dataflow", builder.getI64IntegerAttr(1));
      // Linalg semantics are C += A*B. D and C must refer to that same
      // initialized destination, including when it is a dynamically offset
      // view.
      createOperation(builder, op.getLoc(), "gemmini.tile_matmul",
                      {op->getOperand(0), op->getOperand(1), op->getOperand(2),
                       op->getOperand(2)},
                      attrs);
      builder.eraseOp(op);
    }
  }
};

class LowerTilesPass
    : public PassWrapper<LowerTilesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTilesPass)
  LowerTilesPass() = default;
  LowerTilesPass(const LowerTilesPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final {
    return "buddy-zigzag-lower-gemmini-tiles";
  }
  StringRef getDescription() const final {
    return "Lower scheduled FP32 output tiles to existing Gemmini WS "
           "intrinsics";
  }
  Option<int64_t> dim{*this, "dim", llvm::cl::init(4)};
  Option<int64_t> bankRows{*this, "bank_rows", llvm::cl::init(4096)};
  Option<int64_t> accRows{*this, "acc_rows", llvm::cl::init(4096)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect>();
  }

  LogicalResult lower(Operation *op, IRRewriter &builder) {
    auto schedule = readSchedule(op);
    if (failed(schedule))
      return failure();
    auto s = *schedule;
    if (s.dim != dim || s.bankRows != bankRows || s.accRows != accRows)
      return op->emitOpError(
          "explicit WS schedule disagrees with lowering hardware options");
    if (op->getNumOperands() != 4 ||
        !llvm::all_of(op->getOperandTypes(), isMatrix))
      return op->emitOpError("requires static FP32 rank-2 buffers");
    auto boolean = [&](StringRef name, bool expected) {
      auto value = op->getAttrOfType<BoolAttr>(name);
      return value && value.getValue() == expected;
    };
    auto integer = [&](StringRef name, int64_t expected) {
      auto value = op->getAttrOfType<IntegerAttr>(name);
      return value && value.getInt() == expected;
    };
    if (!boolean("fullC", true) || !boolean("lowD", false) ||
        !boolean("aTranspose", false) || !boolean("bTranspose", false) ||
        !boolean("repeatingBias", false) || !integer("act", 0) ||
        !integer("dataflow", 1) || !integer("weightA", 0))
      return op->emitOpError(
          "requires untransposed full-C FP32 WS without activation");
    for (StringRef name :
         {"aScaleFactor", "bScaleFactor", "dScaleFactor", "accScale"}) {
      auto value = op->getAttrOfType<FloatAttr>(name);
      if (!value || value.getValueAsDouble() != 1.0)
        return op->emitOpError("requires identity FP32 scaling");
    }
    SmallVector<int64_t> strides;
    for (Type type : op->getOperandTypes()) {
      SmallVector<int64_t> values;
      int64_t offset;
      if (failed(cast<MemRefType>(type).getStridesAndOffset(values, offset)) ||
          values[1] != 1 || ShapedType::isDynamic(values[0]) || values[0] <= 0)
        return op->emitOpError(
            "requires static positive row strides and contiguous columns");
      strides.push_back(values[0]);
    }
    auto a = cast<MemRefType>(op->getOperand(0).getType());
    auto b = cast<MemRefType>(op->getOperand(1).getType());
    auto c = cast<MemRefType>(op->getOperand(2).getType());
    auto d = cast<MemRefType>(op->getOperand(3).getType());
    int64_t rows = a.getDimSize(0), depth = a.getDimSize(1),
            cols = b.getDimSize(1);
    if (rows <= 0 || cols <= 0 || depth <= 0 || rows > s.m || cols > s.n ||
        b.getDimSize(0) != depth ||
        c.getShape() != ArrayRef<int64_t>({rows, cols}) ||
        c.getShape() != d.getShape())
      return op->emitOpError(
          "expected one output tile with compatible nonempty matrix shapes");
    if (uint64_t(strides[2]) * 4 > UINT32_MAX)
      return op->emitOpError(
          "output byte stride exceeds the Gemmini store field");

    builder.setInsertionPoint(op);
    Location loc = op->getLoc();
    auto constant = [&](uint64_t value) -> Value {
      return arith::ConstantOp::create(builder, loc,
                                       builder.getI64IntegerAttr(value));
    };
    auto intrinsic = [&](StringRef suffix, Value first, Value second) {
      createOperation(builder, loc, ("gemmini.intr." + suffix).str(),
                      {first, second});
    };
    auto bytePointer = [&](Value buffer) -> Value {
      Value pointer = memref::ExtractAlignedPointerAsIndexOp::create(
          builder, loc, TypeRange(builder.getIndexType()), buffer);
      auto metadata =
          memref::ExtractStridedMetadataOp::create(builder, loc, buffer);
      Value base = arith::IndexCastOp::create(builder, loc,
                                              builder.getI64Type(), pointer);
      Value offset = arith::IndexCastOp::create(
          builder, loc, builder.getI64Type(), metadata.getOffset());
      Value bytes = arith::MulIOp::create(builder, loc, offset, constant(4));
      return arith::AddIOp::create(builder, loc, base, bytes);
    };
    SmallVector<Value> pointers;
    for (Value value : op->getOperands())
      pointers.push_back(bytePointer(value));
    auto address = [&](Value base, uint64_t elements) -> Value {
      return arith::AddIOp::create(builder, loc, base, constant(elements * 4));
    };

    // Identity FP32 scale bits, WS dataflow, and the standard unit strides.
    // Configure once per output tile, outside the remaining static K expansion.
    constexpr uint64_t scale = uint64_t(0x3f800000) << 32;
    intrinsic("config", constant(scale | (uint64_t(1) << 16) | 4),
              constant(uint64_t(1) << 48));
    intrinsic("config", constant(2),
              constant(scale | uint64_t(strides[2] * 4)));
    for (unsigned id = 0; id != 3; ++id) {
      uint64_t packed =
          scale | (uint64_t(s.dim) << 16) | (uint64_t(1) << 8) | (id << 3) | 1;
      intrinsic("config", constant(packed),
                constant(strides[id == 2 ? 3 : id] * 4));
    }

    // Only K expands in the compiler. The Python pass generates runtime M/N
    // loops, with at most four statically shaped bodies for full/tail tiles.
    uint64_t i = (rows + s.dim - 1) / s.dim, j = (cols + s.dim - 1) / s.dim;
    for (int64_t start = 0; start < depth; start += s.k) {
      uint64_t extent = std::min(s.k, depth - start);
      uint64_t k = (extent + s.dim - 1) / s.dim;
      uint64_t padding = ((k * s.dim - extent) << 32) |
                         ((j * s.dim - cols) << 16) | (i * s.dim - rows);
      intrinsic("loop_ws_config_bounds", constant(padding),
                constant((k << 32) | (j << 16) | i));
      intrinsic("loop_ws_config_addrs_ab", address(pointers[0], start),
                address(pointers[1], uint64_t(start) * strides[1]));
      intrinsic("loop_ws_config_addrs_dc",
                start == 0 ? pointers[3] : constant(0),
                start + s.k >= depth ? pointers[2] : constant(0));
      intrinsic("loop_ws_config_strides_ab", constant(strides[0]),
                constant(strides[1]));
      intrinsic("loop_ws_config_strides_dc", constant(strides[3]),
                constant(strides[2]));
      intrinsic("loop_ws", constant(3), constant(0)); // full-C and accumulation
    }
    intrinsic("flush", constant(0), constant(0));
    LLVM::FenceOp::create(builder, loc, LLVM::AtomicOrdering::seq_cst);
    builder.eraseOp(op);
    return success();
  }

  void runOnOperation() override {
    SmallVector<Operation *> operations;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "gemmini.tile_matmul")
        operations.push_back(op);
    });
    IRRewriter builder(&getContext());
    for (Operation *op : operations)
      if (failed(lower(op, builder)))
        return signalPassFailure();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "BuddyZigzagGemmini", LLVM_VERSION_STRING,
          []() {
            PassRegistration<ConvertMatmulPass>();
            PassRegistration<LowerTilesPass>();
          }};
}
