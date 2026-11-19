#include "memref_runtime.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  void *allocated;
  char *aligned;
  int64_t offset;
  int64_t dimensions[]; /* rank sizes followed by rank strides */
} RankedMemRef;

void memrefCopy(int64_t element_size, UnrankedMemRef *source,
                UnrankedMemRef *destination) {
  if (!source || !destination || !source->descriptor ||
      !destination->descriptor || element_size <= 0 || source->rank < 0 ||
      source->rank > 64 || source->rank != destination->rank)
    abort();

  int64_t rank = source->rank;
  RankedMemRef *src = source->descriptor;
  RankedMemRef *dst = destination->descriptor;
  int empty = 0;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (src->dimensions[axis] < 0 ||
        src->dimensions[axis] != dst->dimensions[axis])
      abort();
    empty |= src->dimensions[axis] == 0;
  }
  if (empty)
    return;

  char *src_base = src->aligned + src->offset * element_size;
  char *dst_base = dst->aligned + dst->offset * element_size;
  if (rank == 0) {
    memcpy(dst_base, src_base, (size_t)element_size);
    return;
  }

  int64_t indices[64] = {0};
  int64_t src_index = 0, dst_index = 0;
  for (;;) {
    memcpy(dst_base + dst_index * element_size,
           src_base + src_index * element_size, (size_t)element_size);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      ++indices[axis];
      src_index += src->dimensions[rank + axis];
      dst_index += dst->dimensions[rank + axis];
      if (indices[axis] < src->dimensions[axis])
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      src_index -= src->dimensions[axis] * src->dimensions[rank + axis];
      dst_index -= dst->dimensions[axis] * dst->dimensions[rank + axis];
    }
  }
}
