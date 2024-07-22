// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/cpu.h"
#include "gpu/gpu.h"
#include "utils/tensor/tensor.h"
#include "utils/random.h"
using namespace utils;

TEST(AddTest, _2560_x_8192) {
    std::vector<int64_t> shape{2560, 8192};
    utils::Tensor lhs = gen_rand_tensor(shape);
    utils::Tensor rhs = gen_rand_tensor(shape);
    utils::Tensor cpu_result(shape);
    utils::Tensor gpu_result(shape);

    cpu::add(lhs.data_ptr<float>(), rhs.data_ptr<float>(), cpu_result.data_ptr<float>(), cpu_result.elements_num());
    gpu::add(lhs.data_ptr<float>(), rhs.data_ptr<float>(), gpu_result.data_ptr<float>(), gpu_result.elements_num());

    for (int64_t index = 0; index < lhs.elements_num(); index++) {
        if (cpu_result.data_ptr<float>()[index] != gpu_result.data_ptr<float>()[index]) {
            ASSERT_TRUE(false);
        }
    }
}