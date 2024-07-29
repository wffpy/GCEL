#include "utils/tensor/tensor.h"
#include "utils/tensor/CpuTensorImpl.h"
#include "utils/tensor/GpuTensorImpl.h"
#include <iostream>

namespace utils {

std::shared_ptr<TensorImpl> get_tensor_impl(std::vector<int64_t> shape, DataType dtype, DeviceType device) {
    switch(device) {
        case DeviceType::CPU:
            return std::make_shared<CpuTensorImpl>(shape, dtype);
        case DeviceType::GPU:
            return std::make_shared<GpuTensorImpl>(shape, dtype);
        default:
            // throw std::runtime_error("Unsupported device type");
            return nullptr;
    }
    return nullptr;
}


Tensor::Tensor(const std::vector<int64_t>& dims, DataType dtype, DeviceType device_type) {
    impl_ = get_tensor_impl(dims, dtype, device_type);
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl_;
}

const std::vector<int64_t>& Tensor::shape() {
    return impl_->shape();
}

int64_t Tensor::elements_num() {
    auto shape = impl_->shape();
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

Tensor Tensor::to(DeviceType device) {
    auto cur_dev_type = impl_->device_type();
    if (cur_dev_type == device) {
        // std::cout << "Tensor is already on the device" << std::endl;
        return *this;
    }
    Tensor dst_tensor(impl_->shape(), impl_->data_type(), device);
    dst_tensor.impl_->copy_from(this->impl_);
    return  dst_tensor;
}


}   // namespace utils 