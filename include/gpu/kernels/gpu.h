#ifndef GPU_OPS_H
#define GPU_OPS_H
#include "utils/status.h"
#include <cstdint>

namespace gpu {
int add (float *lhs, float *rhs, float *ret, int length, int64_t priority = 0);

template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col, int64_t priority = 0);
}   // namespace gpu
#endif
