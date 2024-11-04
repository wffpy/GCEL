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

TEST(TransposeTest, _64_x_64) {
    int64_t row = 64;
    int64_t col = 64;

    std::vector<int64_t> shape{row, col};
    utils::Tensor input = gen_order_tensor<float>(shape);

    std::vector<int64_t> result_shape{row, col};
    utils::Tensor cpu_result(result_shape, input.data_type());

    cpu::transpose(input.data_ptr<float>(), cpu_result.data_ptr<float>(), row,
                   col);
    // print_tensor(cpu_result, "cpu result");

    auto gpu_input = input.to(DeviceType::GPU);

    utils::Tensor gpu_result(shape, DataType::FLOAT32, DeviceType::GPU);
    gpu::transpose(gpu_input.data_ptr<float>(), gpu_result.data_ptr<float>(),
                   row, col);

    auto gpu_result_cpu = gpu_result.to(DeviceType::CPU);
    // print_tensor(gpu_result_cpu);

    ASSERT_TRUE(compare_tensor(cpu_result, gpu_result_cpu));
}