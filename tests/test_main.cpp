// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/cpu.h"
#include "gpu/gpu.h"

TEST(SampleTest, AssertionTrue) {
    std::vector<float> lhs = {1,2,3};
    std::vector<float> rhs = {1,2,3};
    std::vector<float> cpu_result = {0,0,0};
    std::vector<float> gpu_result = {0,0,0};
    cpu::add(lhs.data(), rhs.data(), cpu_result.data(), cpu_result.size());
    // std::cout << "cpu result: " << cpu_result[0] << ", " << cpu_result[1] << ", " << cpu_result[2] << std::endl;
    gpu::add(lhs.data(), rhs.data(), gpu_result.data(), gpu_result.size());
    // std::cout << "gpu result: " << gpu_result[0] << ", " << gpu_result[1] << ", " << gpu_result[2] << std::endl;
    for (int64_t index = 0; index < lhs.size(); index++) {
        if (cpu_result[index] != gpu_result[index]) {
            ASSERT_TRUE(false);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

