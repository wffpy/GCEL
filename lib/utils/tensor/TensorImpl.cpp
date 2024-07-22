#include "utils/tensor/TensorImpl.h"

namespace utils {
DataStorage::DataStorage(const int64_t bytes, DeviceType device_type) : bytes_(bytes), device_type_(device_type) {
}
DataStorage::~DataStorage() {
}

DeviceType DataStorage::device_type() const {
    return device_type_;
}

int64_t DataStorage::capacity() const {
    return bytes_;
}

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, DataType dtype, DeviceType device_type) :
    shape_(shape), dtype_(dtype), device_type_(device_type) {}

const std::vector<int64_t>& TensorImpl::shape() const {
    return shape_;
}

DeviceType TensorImpl::device_type() const {
    return device_type_;
}

DataType TensorImpl::data_type() const {
    return dtype_;
}

uint8_t* TensorImpl::data() const {
    return storage_->data();
}

int64_t TensorImpl::capacity() const {
    return storage_->capacity();
}

void TensorImpl::set_storage(std::shared_ptr<DataStorage> storage) {
    storage_ = storage;
}
}   // namespace utils