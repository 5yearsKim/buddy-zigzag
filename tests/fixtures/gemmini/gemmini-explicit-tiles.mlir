// Exercised by tests/test_gemmini_plugin.py with the project pass plugin.
// Two K blocks are requested explicitly although the heuristic could fit all
// K in one block. A partial K block has three padding elements. D loads only
// for the first K block and C stores only for the last K block.
func.func @explicit_tiles(%a: memref<3x5xf32>, %b: memref<5x4xf32>, %c: memref<3x4xf32>) {
  gemmini.tile_matmul %a %b %c %c {
    fullC = true, gemmini.schedule = {
      acc_rows = 4096 : i64, array_dim = 4 : i64, bank_rows = 4096 : i64,
      dataflow = 1 : i64, tile_m = 4 : i64, tile_n = 4 : i64, tile_k = 4 : i64
    }
  } : memref<3x5xf32> memref<5x4xf32> memref<3x4xf32> memref<3x4xf32>
  return
}
