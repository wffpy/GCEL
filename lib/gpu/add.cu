#include <cuda_runtime.h>
#include "gpu/gpu.h"
namespace gpu_kernel{
// CUDA kernel function to add elements of two arrays
__global__ void add(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
}   // namespace gpu_kernel


namespace gpu {
int add(float *lhs, float *rhs, float *ret, int length) {
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    gpu_kernel::add<<<numBlocks, blockSize>>>(lhs, rhs, ret, length);
    return 0;
}

}   // namespace gpu