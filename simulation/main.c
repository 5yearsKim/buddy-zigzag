/* Reusable bare-metal harness. Workload-specific values come from the
 * supplied workload_config.h; link with the lowered kernel and runtime. */
#include "memref_runtime.h"
#include "workload_config.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static float lhs[WORKLOAD_M * WORKLOAD_K] __attribute__((aligned(64)));
static float rhs[WORKLOAD_K * WORKLOAD_N] __attribute__((aligned(64)));
static float output[WORKLOAD_M * WORKLOAD_N] __attribute__((aligned(64)));

extern void WORKLOAD_ENTRY(MemRef2D *, MemRef2D *, MemRef2D *);

static uint64_t cycles(void) {
#ifdef __riscv
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value) : : "memory");
  return value;
#else
  return 0;
#endif
}

static void fence(void) {
#ifdef __riscv
  __asm__ volatile("fence" : : : "memory");
#endif
}

/* Print uint64 without relying on nano printf's optional long-long support. */
static void print_u64(uint64_t value) {
  char digits[20];
  int count = 0;
  do {
    digits[count++] = (char)('0' + value % 10);
    value /= 10;
  } while (value);
  while (count)
    putchar(digits[--count]);
}

int main(void) {
  for (int64_t i = 0; i < (int64_t)WORKLOAD_M * WORKLOAD_K; ++i)
    lhs[i] = (float)((i * 7 + 3) % 23 - 11) / 8.0f;
  for (int64_t i = 0; i < (int64_t)WORKLOAD_K * WORKLOAD_N; ++i)
    rhs[i] = (float)((i * 5 + 1) % 19 - 9) / 8.0f;
  for (int64_t i = 0; i < (int64_t)WORKLOAD_M * WORKLOAD_N; ++i)
    output[i] = NAN;

  MemRef2D a = {lhs, lhs, 0, {WORKLOAD_M, WORKLOAD_K}, {WORKLOAD_K, 1}};
  MemRef2D b = {rhs, rhs, 0, {WORKLOAD_K, WORKLOAD_N}, {WORKLOAD_N, 1}};
  MemRef2D c = {output, output, 0, {WORKLOAD_M, WORKLOAD_N}, {WORKLOAD_N, 1}};
  printf("Buddy-ZigZag: executing %d x %d x %d matmul\n", WORKLOAD_M, WORKLOAD_N, WORKLOAD_K);
  fence();
  uint64_t start = cycles();
  WORKLOAD_ENTRY(WORKLOAD_ARG0_IS_LHS ? &a : &b, WORKLOAD_ARG0_IS_LHS ? &b : &a, &c);
  fence();
  uint64_t elapsed = cycles() - start;

  uint64_t mismatches = 0;
  for (int64_t row = 0; row < WORKLOAD_M; ++row) {
    for (int64_t col = 0; col < WORKLOAD_N; ++col) {
      double expected = 0.0;
      for (int64_t k = 0; k < WORKLOAD_K; ++k)
        expected += (double)lhs[row * WORKLOAD_K + k] * rhs[k * WORKLOAD_N + col];
      double actual = output[row * WORKLOAD_N + col];
      if (!isfinite(actual) ||
          fabs(actual - expected) > WORKLOAD_ATOL + WORKLOAD_RTOL * fabs(expected))
        ++mismatches;
    }
  }
  printf("BZ_SIM_RESULT {\"status\":\"%s\",\"cycles\":",
         mismatches ? "failed" : "passed");
  print_u64(elapsed);
  printf(",\"mismatches\":");
  print_u64(mismatches);
  printf("}\n");
  return mismatches ? 1 : 0;
}
