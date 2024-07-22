#include "utils/tensor/CpuTensorImpl.h"
#include <cuda_runtime.h>
#include <numeric>
#include <cstring>

namespace utils {
CpuDataStorage::CpuDataStorage(const int64_t bytes, DeviceType device_type) : DataStorage(bytes, device_type) {
    if (bytes > 0) {
        data_ = new uint8_t[bytes];
    } else {
        data_ = nullptr;
    }
}

CpuDataStorage::~CpuDataStorage() {
    if (data_ != nullptr) {
        delete[] data_;
    }
    data_ = nullptr;
}

uint8_t* CpuDataStorage::data() {
    return data_;
}

CpuTensorImpl::CpuTensorImpl(const std::vector<int64_t>& shape, DataType dtype) : TensorImpl(shape, dtype, DeviceType::CPU) {
    int64_t dtype_bytes = get_dtype_bytes(dtype);
    int64_t bytes = std::accumulate(shape.begin(), shape.end(), dtype_bytes, std::multiplies<int64_t>());
    std::shared_ptr<CpuDataStorage> data_storage = std::make_shared<CpuDataStorage>(bytes, DeviceType::CPU);
    set_storage(std::move(data_storage));
}

void CpuTensorImpl::copy_from(std::shared_ptr<TensorImpl> src) {
    if (src == nullptr) {
        return;
    }
    if (src->device_type() == DeviceType::CPU) {
        auto src_impl = std::dynamic_pointer_cast<CpuTensorImpl>(src);
        if (src_impl != nullptr) {
            int64_t bytes = src->capacity();
            std::memcpy(data(), src_impl->data(), bytes);
        }
    } else if (src->device_type() == DeviceType::GPU) {
        int64_t bytes = src->capacity();
        // TODO: implement copy from GPU to CPU
        cudaMemcpy(data(), src->data(), bytes, cudaMemcpyDeviceToHost);
    }

}

}   // namespace utils