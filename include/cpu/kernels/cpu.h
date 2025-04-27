#ifndef CPU_OPS_H
#define CPU_OPS_H
#include <iostream>
#include "utils/status.h"
#include <vector>

namespace cpu {
int add(float *lhs, float *rhs, float* ret, int length);

template <typename T>
utils::GCELResult gemm(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k, bool lhs_trans=false, bool rhs_trans=false);

template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col, int64_t row_block_size=64, int64_t col_block_size=64);
}   // namespace Op

#endif