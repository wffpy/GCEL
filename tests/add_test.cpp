// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/cpu.h"
#include "gpu/gpu.h"
#include "utils/tensor.h"
#include "utils/random.h"
using namespace utils;

TEST(AddTest, _2560_x_8192) {
    std::vector<int64_t> shape{2560, 8192};
    utils::Tensor<float> lhs = gen_rand_tensor<float>(shape);
    utils::Tensor<float> rhs = gen_rand_tensor<float>(shape);
    utils::Tensor<float> cpu_result(shape);
    utils::Tensor<float> gpu_result(shape);

    cpu::add(lhs.data_ptr(), rhs.data_ptr(), cpu_result.data_ptr(), cpu_result.elements_num());
    gpu::add(lhs.data_ptr(), rhs.data_ptr(), gpu_result.data_ptr(), gpu_result.elements_num());

    for (int64_t index = 0; index < lhs.elements_num(); index++) {
        if (cpu_result.data_ptr()[index] != gpu_result.data_ptr()[index]) {
            ASSERT_TRUE(false);
        }
    }
}