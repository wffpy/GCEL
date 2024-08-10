// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/cpu.h"
#include "gpu/gpu.h"
#include "utils/tensor/tensor.h"
#include "utils/random.h"
#include "utils/test_helper.h"
using namespace utils;

TEST(TransposeTest, _2560_x_8192) {
    std::vector<int64_t> shape{2, 16};
    utils::Tensor input = gen_order_tensor<int32_t>(shape);
    
    std::vector<int64_t> result_shape{16, 2};
    utils::Tensor cpu_result(result_shape, input.data_type());
    print_tensor(input);

    cpu::transpose(input.data_ptr<int32_t>(), cpu_result.data_ptr<int32_t>(), 2, 16);
    print_tensor(cpu_result);
}