// test_main.cpp
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "cpu/cpu.h"
#include "gpu/gpu.h"
#include "utils/random.h"
#include "utils/tensor/tensor.h"
#include "utils/test_helper.h"
using namespace utils;

#define DEFINE_TRANSPOSE_TEST(r, c)                                           \
    TEST(TransposeTest, r##_x_##c) {                                          \
        int64_t row = r;                                                      \
        int64_t col = c;                                                      \
                                                                              \
        std::vector<int64_t> shape{row, col};                                 \
        utils::Tensor input = gen_order_tensor<float>(shape);                 \
                                                                              \
        std::vector<int64_t> result_shape{row, col};                          \
        utils::Tensor cpu_result(result_shape, input.data_type());            \
                                                                              \
        cpu::transpose(input.data_ptr<float>(), cpu_result.data_ptr<float>(), \
                       row, col);                                             \
                                                                              \
        auto gpu_input = input.to(DeviceType::GPU);                           \
                                                                              \
        utils::Tensor gpu_result(shape, DataType::FLOAT32, DeviceType::GPU);  \
        gpu::transpose(gpu_input.data_ptr<float>(),                           \
                       gpu_result.data_ptr<float>(), row, col);               \
        auto gpu_result_cpu = gpu_result.to(DeviceType::CPU);                 \
        ASSERT_TRUE(compare_tensor(cpu_result, gpu_result_cpu));              \
    }

DEFINE_TRANSPOSE_TEST(8, 4)
DEFINE_TRANSPOSE_TEST(4, 8)
DEFINE_TRANSPOSE_TEST(16, 32)
DEFINE_TRANSPOSE_TEST(32, 16)
DEFINE_TRANSPOSE_TEST(56, 56)
DEFINE_TRANSPOSE_TEST(64, 64)
