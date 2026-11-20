// Exercised by tests/test_gemmini_plugin.py with the project pass plugin.
// Scheduled Linalg uses the existing C destination as the D bias. There must
// be no fresh bias allocation or fill that would discard partial sums.
func.func @scheduled(%a: memref<3x5xf32>, %b: memref<5x7xf32>, %c: memref<3x7xf32>) {
  linalg.matmul {gemmini.schedule = {
    acc_rows = 4096 : i64, array_dim = 4 : i64, bank_rows = 4096 : i64,
    dataflow = 1 : i64, tile_m = 4 : i64, tile_n = 4 : i64, tile_k = 4 : i64
  }} ins(%a, %b : memref<3x5xf32>, memref<5x7xf32>) outs(%c : memref<3x7xf32>)
  return
}
