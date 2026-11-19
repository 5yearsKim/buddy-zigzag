#ifndef BUDDY_ZIGZAG_MEMREF_RUNTIME_H
#define BUDDY_ZIGZAG_MEMREF_RUNTIME_H

#include <stdint.h>

/* MLIR's C ABI: allocated pointer, aligned pointer, element offset, sizes,
 * then element strides. This runtime targets 64-bit hosts and RISC-V RV64. */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D;

typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRef;

void memrefCopy(int64_t element_size, UnrankedMemRef *source,
                UnrankedMemRef *destination);

#endif
