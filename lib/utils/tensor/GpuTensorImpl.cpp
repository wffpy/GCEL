#include "utils/tensor/GpuTensorImpl.h"
#include "cuda_runtime.h"
#include <numeric>

namespace utils {
GpuDataStorage::GpuDataStorage(const int64_t bytes, DeviceType device_type) : DataStorage(bytes, device_type) {
    if (bytes > 0) {
        cudaMalloc((void **)&data_, bytes);
    } else {
        data_ = nullptr;
    }
}

GpuDataStorage::~GpuDataStorage() {
    if (data_ != nullptr) {
        cudaFree(data_);
    }
    data_ = nullptr;
}

uint8_t* GpuDataStorage::data() {
    return data_;
}

GpuTensorImpl::GpuTensorImpl(const std::vector<int64_t>& shape, DataType dtype) : TensorImpl(shape, dtype, DeviceType::CPU) {
    int64_t dtype_bytes = get_dtype_bytes(dtype);
    int64_t bytes = std::accumulate(shape.begin(), shape.end(), dtype_bytes, std::multiplies<int64_t>());
    std::shared_ptr<GpuDataStorage> data_storage = std::make_shared<GpuDataStorage>(bytes, DeviceType::CPU);
    set_storage(std::move(data_storage));
}

void GpuTensorImpl::copy_from(std::shared_ptr<TensorImpl> src) {
    if (src->device_type() == DeviceType::CPU) {
        int64_t bytes = src->capacity();
        cudaMemcpy(data(), src->data(), bytes, cudaMemcpyDeviceToHost);
    } else if (src->device_type() == DeviceType::GPU) {
        int64_t bytes = src->capacity();
        cudaMemcpy(data(), src->data(), bytes, cudaMemcpyDeviceToDevice);
    }
}

}   // namespace utils