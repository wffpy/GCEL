#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <initializer_list>
#include <cstdint>
#include <memory>
#include <numeric>

#include "utils/tensor/basic.h"
#include "utils/tensor/TensorImpl.h"

namespace utils {

int64_t get_dtype_bytes(DataType dtype);

// class DataPtr {
// public:
//     DataPtr() = delete;
//     DataPtr(int64_t bytes);
//     ~DataPtr();
//     uint8_t* data() { return ptr; }
// private:
//     uint8_t* ptr;
//     int64_t bytes_;
// };

class Tensor {
public:
    // Tensor (std::initializer_list<int64_t> dims, DataType dtype=DataType::FLOAT32, DeviceType device_type=DeviceType::CPU);
    Tensor (const std::vector<int64_t>& dims, DataType dtype=DataType::FLOAT32, DeviceType device_type=DeviceType::CPU);
    const std::vector<int64_t>& shape();
    int64_t elements_num();
    template <typename DT>
    DT* data_ptr();
    DataType data_type() { return impl_->data_type(); }
    DeviceType device_type() { return impl_->device_type(); }
    Tensor to(DeviceType dev_type);
    // void copy_from(const Tensor& other);
private:
    std::shared_ptr<TensorImpl> impl_;
};


template <typename DT>
DT* Tensor::data_ptr() {
    return (DT*)(impl_->data());
}

}   // namespace utils
#endif