#ifndef CPU_OPS_H
#define CPU_OPS_H
#include <iostream>
#include "common/tensor/tensor.h"
#include <vector>
using namespace common;
namespace cpu {
// convert tensor type
Tensor to(const Tensor &lhs, DataType type);

// element-wise add
Tensor add(const Tensor &lhs, const Tensor &rhs);

// transpose
Tensor transpose(const Tensor &lhs);

// matrix multiplication, lhs{m, k}, rhs{k, n}, output{m, n}
Tensor matmul(const Tensor &lhs, const Tensor &rhs);
}   // namespace cpu

#endif  // CPU_OPS_H