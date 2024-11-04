#ifndef UTILS_TEST_HELPER_H
#define UTILS_TEST_HELPER_H

#include "utils/tensor/tensor.h"

namespace utils {
void print_tensor(const Tensor& t, std::string t_name = "");

bool compare_tensor(const Tensor& t1, const Tensor& t2,
                    float error_threshold = 1e-3);
}  // namespace utils

#endif