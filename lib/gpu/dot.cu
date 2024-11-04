#include <cuda_runtime.h>
#include "gpu/gpu.h"

namespace gpu_kernel {
// CUDA kernel function to do dot-product of two vectors
template <typename T>
__global__ void dot(const T *lhs, T *b, T *c, int64_t m, int64_t n, int64_t k, bool lhs_trans, bool rhs_trans) {
}
}   // namespace gpu_kernel

namespace gpu {
}   // namespace gpu