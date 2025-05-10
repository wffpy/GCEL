#include <cuda_runtime.h>
#include "gpu/kernels/gpu_kernels.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utils/cal_helper/cal_helper.h"
using namespace nvcuda;

namespace gpu_kernel {
// CUDA kernel function to do dot-product of two vectors
template <typename T>
__global__ void gemm_cudacore(const T *lhs, const T *rhs, T *ret, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        T sum = 0;
        for (int l = 0; l < k; ++l) {
            sum += lhs[row * k + l] * rhs[l * n + col];
        }
        ret[row * n + col] = sum;
    }
}

// m, n, k 必须是16的倍数
template <typename T>
__global__ void gemm_tensorcore(const T *A, const T *B, T *C, int m, int n, int k) {
    // 以16x16x16为tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 16;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 16;

    if (warpM * 16 < m && warpN * 16 < n) {
        // fragment声明
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int tileK = 0; tileK < k; tileK += 16) {
            // 加载A和B的tile
            const T *tileA = A + warpM * 16 * k + tileK;
            const T *tileB = B + tileK * n + warpN * 16;

            wmma::load_matrix_sync(a_frag, tileA, k);
            wmma::load_matrix_sync(b_frag, tileB, n);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // 写回C
        T *tileC = C + warpM * 16 * n + warpN * 16;
        wmma::store_matrix_sync(tileC, c_frag, n, wmma::mem_row_major);
    }
}

}   // namespace gpu_kernel

namespace gpu_kernels {
template <typename T>
GCELResult gemm(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k) {
    int row_tile_dim = 32; 
    int col_tile_dim = 32;
    int round_up_col = cal_helper::roundUp(m, col_tile_dim);
    int round_up_row = cal_helper::roundUp(n, row_tile_dim);
    dim3 dimBlock(row_tile_dim, col_tile_dim, 1);
    dim3 dimGrid(round_up_row / row_tile_dim, round_up_col / col_tile_dim, 1);
    gpu_kernel::gemm_cudacore<<<dimGrid, dimBlock>>>(lhs, rhs, ret, m, n, k);
    return GCELResult::SUCCESS;
}

template <typename T, typename std::enable_if<std::is_same<T, half>::value>::type>
GCELResult gemm(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k) {
    if (IS_TYPE(T, half) || IS_TYPE(T, nv_bfloat16)) {
        dim3 block(32, 8); // 8 warps per block
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        gpu_kernel::gemm_tensorcore<<<grid, block>>>(lhs, rhs, ret, m, n, k);
    }
    return GCELResult::SUCCESS;
}


#define INSTANTIATE_GEMM(T) \
    template GCELResult gemm<T>(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k);

INSTANTIATE_GEMM(half);
INSTANTIATE_GEMM(float);
}   // namespace gpu_kernels