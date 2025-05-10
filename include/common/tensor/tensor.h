#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <initializer_list>
#include <cstdint>
#include <memory>
#include <numeric>

#include "common/tensor/basic.h"
#include "common/tensor/TensorImpl.h"

namespace common {

int64_t get_dtype_bytes(DataType dtype);

class Tensor {
public:
    // Tensor (std::initializer_list<int64_t> dims, DataType dtype=DataType::FLOAT32, DeviceType device_type=DeviceType::CPU);
    Tensor (const std::vector<int64_t>& dims, DataType dtype=DataType::FLOAT32, DeviceType device_type=DeviceType::CPU);
    Tensor (const Tensor& other);
    const std::vector<int64_t>& shape() const;
    int64_t elements_num() const;
    template <typename DT>
    DT* data_ptr() const;
    DataType data_type() const { return impl_->data_type(); }
    DeviceType device_type() const { return impl_->device_type(); }
    Tensor to(DeviceType dev_type);
private:
    std::shared_ptr<TensorImpl> impl_;
};


template <typename DT>
DT* Tensor::data_ptr() const {
    return (DT*)(impl_->data());
}

}   // namespace common
#endif