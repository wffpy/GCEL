#ifndef UTILS_H
#define UTILS_H
#include "common/tensor/tensor.h"
#include "status.h"
#include "common/tensor/basic.h"
#include <random>
using namespace common;
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

template<typename DT>
GCELResult gen_order_seq(DT* dst, int64_t size, DT start, DT step) {
    for (int64_t i = 0; i < size; i++) {
        dst[i] = start + step * i;
    }
    return GCELResult::SUCCESS;
}


Tensor gen_rand_tensor(std::vector<int64_t> shape, DataType dtype=DataType::FLOAT32, DeviceType device_type=DeviceType::CPU);

template <class T>
Tensor gen_order_tensor(std::vector<int64_t> shape, DeviceType device_type=DeviceType::CPU, const T start = 0, const T step = 1) {
    DataType data_type = CppTypeMap<T>::value;
    Tensor t(shape, data_type, device_type);
    auto ptr = t.data_ptr<T>();
    gen_order_seq<T>(ptr, t.elements_num(), start, step);
    return t;
}

}   // utils
#endif