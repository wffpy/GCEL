#ifndef GPU_OPS_H
#define GPU_OPS_H
#include "utils/status.h"
#include <cstdint>
using namespace utils;

namespace gpu {
int add (float *lhs, float *rhs, float *ret, int length, int64_t priority = 0);

template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col, int64_t priority = 0);

template <typename T>
GCELResult gemm(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k);

}   // namespace gpu
#endif
