#include <cuda_runtime.h>

#include "gpu/gpu.h"
namespace gpu_kernel {
// CUDA kernel function to transpose a matrix
template <typename T>
__global__ void transpose_impl_(const T* input, T* ret, int64_t row,
                                int64_t col, int64_t tile_dim,
                                int64_t block_size) {
    int x = blockIdx.x * tile_dim + threadIdx.x;
    int y = blockIdx.y * tile_dim + threadIdx.y;
    int width = gridDim.x * tile_dim;

    for (int index = 0; index < tile_dim; index += block_size) {
        ret[x * width + y + index] = input[(y + index) * width + x];
    }
}
}  // namespace gpu_kernel

namespace gpu {
template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col) {
    int tile_dim = 32;
    int block_size = 8;
    dim3 dimGrid(row / tile_dim, col / tile_dim, 1);
    dim3 dimBlock(tile_dim, block_size, 1);

    gpu_kernel::transpose_impl_<<<dimGrid, dimBlock>>>(input, ret, row, col,
                                                       tile_dim, block_size);

    return utils::GCELResult::SUCCESS;
}

#define INSTANTIATE_TRANSPOSE(T) \
    template utils::GCELResult transpose<T>(const T*, T*, int64_t, int64_t);

INSTANTIATE_TRANSPOSE(char);
INSTANTIATE_TRANSPOSE(int32_t);
INSTANTIATE_TRANSPOSE(int64_t);
INSTANTIATE_TRANSPOSE(float);
INSTANTIATE_TRANSPOSE(double);
}  // namespace gpu