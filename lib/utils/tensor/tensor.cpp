#include "utils/tensor/tensor.h"
#include "utils/tensor/CpuTensorImpl.h"
#include "utils/tensor/GpuTensorImpl.h"
#include "utils/log/Log.h"
#include "utils/tensor/basic.h"
#include <iostream>

namespace utils {

std::shared_ptr<TensorImpl> get_tensor_impl(std::vector<int64_t> shape, DataType dtype, DeviceType device) {
    switch(device) {
        case DeviceType::CPU:
            return std::make_shared<CpuTensorImpl>(shape, dtype);
        case DeviceType::GPU:
            return std::make_shared<GpuTensorImpl>(shape, dtype);
        default:
            ELOG() << "Unsupported device type: " << static_cast<int>(device);
    }
    return nullptr;
}


Tensor::Tensor(const std::vector<int64_t>& dims, DataType dtype, DeviceType device_type) {
    impl_ = get_tensor_impl(dims, dtype, device_type);
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl_;
}

const std::vector<int64_t>& Tensor::shape() const {
    return impl_->shape();
}

int64_t Tensor::elements_num() const {
    auto shape = impl_->shape();
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

Tensor Tensor::to(DeviceType device) {
    auto cur_dev_type = impl_->device_type();
    if (cur_dev_type == device) {
        LOG() << "Tensor is already on the device";
        return *this;
    }
    DLOG() << "copy tensor from " << get_device_str(cur_dev_type) << " to " << get_device_str(device);
    Tensor dst_tensor(impl_->shape(), impl_->data_type(), device);
    dst_tensor.impl_->copy_from(this->impl_);
    return  dst_tensor;
}


}   // namespace utils 