#include <immintrin.h>  // AVX指令集

#include <cstring>

#include "cpu/cpu.h"
#include "utils/log/Log.h"

namespace cpu {
int add(float* lhs, float* rhs, float* ret, int length) {
    float* aligned_vec1 =
        static_cast<float*>(_mm_malloc(length * sizeof(float), 32));
    float* aligned_vec2 =
        static_cast<float*>(_mm_malloc(length * sizeof(float), 32));
    float* aligned_result =
        static_cast<float*>(_mm_malloc(length * sizeof(float), 32));
    std::copy(lhs, lhs + length, aligned_vec1);
    std::copy(rhs, rhs + length, aligned_vec2);

    int64_t length1 = length & (~0x7);
    int64_t length2 = length & 0x7;
    DLOG() << "512 bits aligned bytes: " << length1;
    DLOG() << "Non-512 bits aligned bytes: " << length2;

    for (int64_t i = 0; i < length1; i += 8) {
        __m256 vec1_part = _mm256_load_ps(&aligned_vec1[i]);
        __m256 vec2_part = _mm256_load_ps(&aligned_vec2[i]);
        __m256 result_part = _mm256_add_ps(vec1_part, vec2_part);
        _mm256_store_ps(&aligned_result[i], result_part);
    }
    std::copy(&aligned_result[0], &aligned_result[length], ret);
    for (int i = 0; i < length2; ++i) {
        ret[length1 + i] = lhs[length1 + i] + rhs[length1 + i];
    }
    return 0;
}

}  // namespace cpu
