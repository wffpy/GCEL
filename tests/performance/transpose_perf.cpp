#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "cpu/kernels/cpu_kernels.h"
#include "gpu/kernels/gpu_kernels.h"
#include "perf_util.h"
#include "utils/random.h"
#include "common/tensor/tensor.h"
#include "utils/test_helper.h"
using namespace utils;
using namespace common;

#define DEFINE_TRANSPOSE_TEST(r, c)                                          \
    TEST(TransposePerfTest, r##_x_##c) {                                         \
        int64_t row = r;                                                     \
        int64_t col = c;                                                     \
        std::vector<int64_t> shape{row, col};                                \
        Tensor input = gen_order_tensor<float>(shape);                \
        auto gpu_input = input.to(DeviceType::GPU);                          \
        Tensor gpu_result(shape, DataType::FLOAT32, DeviceType::GPU); \
        PROFILE_SCOPE_BEGIN(100);                                            \
        gpu_kernels::transpose(gpu_input.data_ptr<float>(),                           \
                        gpu_result.data_ptr<float>(),                        \
                       row, col);                                            \
        PROFILE_SCOPE_END();                                                 \
    }

DEFINE_TRANSPOSE_TEST(64, 64);
