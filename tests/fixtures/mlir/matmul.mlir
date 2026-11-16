module {
  func.func @matmul(%a: tensor<3x5xf32>, %b: tensor<5x7xf32>) -> tensor<3x7xf32> {
    %zero = arith.constant dense<0.0> : tensor<3x7xf32>
    %result = linalg.matmul ins(%a, %b : tensor<3x5xf32>, tensor<5x7xf32>)
      outs(%zero : tensor<3x7xf32>) -> tensor<3x7xf32>
    return %result : tensor<3x7xf32>
  }
}
