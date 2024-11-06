#include <cuda_runtime.h>

#include "gpu/gpu.h"
namespace gpu_kernel {
// CUDA kernel function to transpose a matrix
template <typename T>
__global__ void transpose_impl_(const T* input, T* ret, int64_t row,
                                int64_t col, int64_t row_tile_dim,
                                int64_t col_tile_dim,
                                int64_t block_size) {
    int x = blockIdx.x * col_tile_dim + threadIdx.x;
    int y = blockIdx.y * row_tile_dim + threadIdx.y;
    int width = gridDim.x * col_tile_dim;

    for (int index = 0; index < row_tile_dim; index += block_size) {
        ret[x * row + y + index] = input[(y + index) * width + x];
    }
}
}  // namespace gpu_kernel

namespace gpu {
template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col) {
    int row_tile_dim = 32;
    int col_tile_dim = 32;
    int block_size = 8;

    if (col < col_tile_dim) {
        col_tile_dim = col;
    }
    if (row < row_tile_dim) {
        row_tile_dim = row;
    }

    if (row < block_size) {
        block_size = row;
    }

    dim3 dimGrid(col / col_tile_dim, row / row_tile_dim, 1);
    dim3 dimBlock(col_tile_dim, block_size, 1);

    gpu_kernel::transpose_impl_<<<dimGrid, dimBlock>>>(input, ret, row, col,
                                                       row_tile_dim, col_tile_dim, block_size);

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