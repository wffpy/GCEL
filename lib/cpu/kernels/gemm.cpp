#include <immintrin.h>  // AVX指令集

#include <cstring>

#include "cpu/kernels/cpu.h"
#include "utils/log/Log.h"

namespace cpu {

// // [v0, v1, v2, v3, v4, v5, v6, v7]
// float reduce_with_add(__m256 v) {
//     // [v0, v1, v2, v3]
//     __m128 vlow = _mm256_castps256_ps128(v);
//     // [v4, v5, v6, v7]
//     __m128 vhigh = _mm256_extractf128_ps(v, 1);

//     // [v0+v4, v1+v5, v2+v6, v3+v7]
//     __m128 sums = _mm_add_ps(vlow, vhigh);

//     // [v1+v5, v1+v5, v3+v7, v3+v7]
//     __m128 shuf = _mm_movehdup_ps(sums);

//     // [v0+v4+v1+v5, v1+v5+v1+v5, v2+v6+v3+v7, v3+v7+v3+v7]
//     sums = _mm_add_ps(sums, shuf);

//     // [v2+v6+v3+v7, v3+v7+v3+v7, v1+v5, v1+v5]
//     shuf = _mm_movehl_ps(shuf, sums);

//     // [v0+v4+v1+v5+v2+v6+v3+v7, v1+v5+v1+v5, v2+v6+v3+v7, v3+v7+v3+v7]
//     // just add the first elment of two 128-bit vectors
//     sums = _mm_add_ss(sums, shuf);

//     // extract the first element of 128-bit vector
//     return _mm_cvtss_f32(sums);
// }

// template <typename T>
// utils::GCELResult gemm_impl(const T* lhs, const T* rhs, T* ret, int64_t M,
//                            int64_t N, int64_t K, bool lhs_trans = false,
//                            bool rhs_trans = false) {
//     float* aligned_vec1 =
//         static_cast<float*>(_mm_malloc(M * K * sizeof(float), 32));
//     float* aligned_vec2 =
//         static_cast<float*>(_mm_malloc(N * K * sizeof(float), 32));
//     float* aligned_result =
//         static_cast<float*>(_mm_malloc(M * N * sizeof(float), 32));
//     // std::copy(lhs, lhs + K, aligned_vec1);
//     // std::copy(rhs, rhs + K, aligned_vec2);
//     int64_t lhs_length = M * K;
//     if (lhs_trans) {
//         T lhs_trsns[M * K];
//         auto ret = transpose(lhs, lhs_trsns, K, M);
//         lhs = lhs_trsns;
//     }
//     std::copy(lhs, lhs + lhs_length, aligned_vec1);

//     int64_t rhs_length = N * K;
//     if (rhs_trans) {
//         T rhs_trsns[N * K];
//         auto ret = transpose(rhs, rhs_trsns, K, N);
//         rhs = rhs_trsns;
//     }
//     std::copy(rhs, rhs + rhs_length, aligned_vec2);

//     // init the result matrix
//     memset(ret, 0, M * N);

//     // gemm product
//     int64_t aligned_K = K & (~0x7);
//     int64_t res_K = K & 0x7;
//     for (int64_t m = 0; m < M; m += 8) {
//         for (int64_t k = 0; k < aligned_K; k += 8) {
//             __m256 vec1_part = _mm256_load_ps(&aligned_vec1[m * K + k]);
//             for (int64_t n = 0; n < N; n += 8) {
//                 __m256 vec2_part = _mm256_load_ps(&aligned_vec2[n * K + k]);
//                 __m256 result_part = _mm_mul_ss(vec1_part, vec2_part);
//                 ret[m * N + n] += reduce_with_add(result_part);
//             }
//         }
//         for (int64_t n = 0; n < N; n++) {
//             for (int64_t k = aligned_K; k < K; k++) {
//                 ret[m * N + n] += lhs[m * K + k] * rhs[n * K + k];
//             }
//         }
//     }
//     return utils::GCELResult::SUCCESS;
// }

// default case: lhs[m, k], rhs[k, n]
template<typename T>
utils::GCELResult gemm_impl(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n, int64_t k, bool lhs_trans, bool rhs_trans) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            T sum = 0;
            for (int64_t l = 0; l < k; l++) {
                sum += lhs[i * k + l] * rhs[l * n + j];
            }
            ret[i * n + j] = sum;
        }
    }
    return utils::GCELResult::SUCCESS;
}

template <typename T>
utils::GCELResult gemm(const T* lhs, const T* rhs, T* ret, int64_t m, int64_t n,
                      int64_t k, bool lhs_trans,
                      bool rhs_trans) {
    gemm_impl(lhs, rhs, ret, m, n, k, lhs_trans, rhs_trans);
    return utils::GCELResult::SUCCESS;
}

#define INSTANTIATE_GEMM(T)                                             \
    template utils::GCELResult gemm(const T* lhs, const T* rhs, T* ret, \
                                   int64_t m, int64_t n, int64_t k,    \
                                   bool lhs_trans, bool rhs_trans);

INSTANTIATE_GEMM(uint8_t);
INSTANTIATE_GEMM(int32_t);
INSTANTIATE_GEMM(int64_t);
INSTANTIATE_GEMM(float);
}  // namespace cpu
