// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/kernels/cpu_kernels.h"
#include "gpu/kernels/gpu_kernels.h"
#include "common/tensor/tensor.h"
#include "utils/random.h"
using namespace utils;
using namespace common;

TEST(GemmTest, _2560_x_8192_x_8192_x_128) {
    std::vector<int64_t> lhs_shape{48, 32};
    std::vector<int64_t> rhs_shape{32, 16};
    std::vector<int64_t> ret_shape{48, 16};
    Tensor lhs = gen_rand_tensor(lhs_shape);
    Tensor rhs = gen_rand_tensor(rhs_shape);
    Tensor cpu_result(ret_shape);
    Tensor gpu_result(ret_shape, DataType::FLOAT32, DeviceType::GPU);

    int64_t m = lhs_shape[0];
    int64_t k = lhs_shape[1];
    int64_t n = rhs_shape[1];

    auto lhs_gpu = lhs.to(DeviceType::GPU);
    auto rhs_gpu = rhs.to(DeviceType::GPU);
    gpu_kernels::gemm(lhs_gpu.data_ptr<float>(), rhs_gpu.data_ptr<float>(), gpu_result.data_ptr<float>(), m, n, k);
    auto ret = gpu_result.to(DeviceType::CPU);

    cpu_kernels::gemm(lhs.data_ptr<float>(), rhs.data_ptr<float>(), cpu_result.data_ptr<float>(), m, n, k);

    for (int64_t index = 0; index < cpu_result.elements_num(); index++) {
        float cpu_result_val = cpu_result.data_ptr<float>()[index];
        float gpu_result_val = ret.data_ptr<float>()[index];
        float error = std::abs(cpu_result_val - gpu_result_val);
        if (error > 1e-5) {
            std::cout << "error index: " << index << std::endl;
            std::cout << "cpu_result: " << cpu_result_val << std::endl;
            std::cout << "gpu_result: " << gpu_result_val << std::endl;
            ASSERT_TRUE(false);
        }
    }
}
