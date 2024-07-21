#ifndef UTILS_H
#define UTILS_H
#include "tensor.h"
#include "status.h"
#include <random>
namespace utils {

template <typename DT>
GCELResult gen_rand(DT* dst,int64_t size, DT start = 0, DT end = 1) {
    // Create a random device and seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // Define the range for the random floats
    std::uniform_real_distribution<> dis(start, end);

    for (int64_t i = 0; i < size; i++) {
        dst[i] = dis(gen);
    }
    return GCELResult::SUCCESS;
}

template <typename DT>
Tensor<DT> gen_rand_tensor(std::vector<int64_t>shape) {
    Tensor<DT> t(shape);
    auto ptr = t.data_ptr();
    int64_t elems = t.elements_num();
    gen_rand(ptr, elems);
    return t;
}

}   // utils
#endif