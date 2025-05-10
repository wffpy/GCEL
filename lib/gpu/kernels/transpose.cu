#include <cuda_runtime.h>

#include "gpu/kernels/gpu_kernels.h"
#include "utils/cal_helper/cal_helper.h"
namespace gpu_kernel {
// CUDA kernel function to transpose a matrix
template <typename T>
__global__ void transpose_impl_0_(const T* input, T* ret, int64_t row,
                                  int64_t col, int64_t row_tile_dim,
                                  int64_t col_tile_dim, int64_t block_size) {
    int x = blockIdx.x * col_tile_dim + threadIdx.x;
    int y = blockIdx.y * row_tile_dim + threadIdx.y;
    int width = col;

    if (x < col) {
        for (int index = 0; index < row_tile_dim; index += block_size) {
            if (y + index < row) {
                ret[x * row + y + index] = input[(y + index) * width + x];
            }
        }
    }
}

template <typename T>
__global__ void transpose_impl_1_(const T* input, T* ret, int64_t row,
                                  int64_t col, int64_t row_tile_dim,
                                  int64_t col_tile_dim, int64_t block_size) {}

}  // namespace gpu_kernel

namespace gpu_kernels {

#define CALL_TRANSPOSE_IMPL(n)                                \
    gpu_kernel::transpose_impl_##n##_<<<dimGrid, dimBlock>>>( \
        input, ret, row, col, row_tile_dim, col_tile_dim, block_size);

template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col,
                            int64_t priority) {
    int row_tile_dim = 32;
    int col_tile_dim = 32;
    int block_size = 8;

    int round_up_col = cal_helper::roundUp(col, col_tile_dim);
    int round_up_row = cal_helper::roundUp(row, row_tile_dim);

    dim3 dimGrid(round_up_col / col_tile_dim, round_up_row / row_tile_dim, 1);
    dim3 dimBlock(col_tile_dim, block_size, 1);

    if (priority == 0) {
        CALL_TRANSPOSE_IMPL(0);
    } else if (priority == 1) {
        CALL_TRANSPOSE_IMPL(1);
    }

    return utils::GCELResult::SUCCESS;
}

#define INSTANTIATE_TRANSPOSE(T) \
    template utils::GCELResult transpose<T>(const T*, T*, int64_t, int64_t, int64_t);

INSTANTIATE_TRANSPOSE(char);
INSTANTIATE_TRANSPOSE(int32_t);
INSTANTIATE_TRANSPOSE(int64_t);
INSTANTIATE_TRANSPOSE(float);
INSTANTIATE_TRANSPOSE(double);
}  // namespace gpu_kernels