#ifndef TENSOR_GPUTENSORIMPL_H
#define TENSOR_GPUTENSORIMPL_H
#include "common/tensor/TensorImpl.h"

namespace common {
class GpuDataStorage : public DataStorage {
public:
    GpuDataStorage(const int64_t bytes, DeviceType device_type);
    ~GpuDataStorage();
    uint8_t* data() override;
private:
    uint8_t* data_;
};

class GpuTensorImpl : public TensorImpl {
public:
    GpuTensorImpl(const std::vector<int64_t>& shape, DataType dtype);
    ~GpuTensorImpl() {};
    virtual void copy_from(std::shared_ptr<TensorImpl> src) override;
};

}   // namespace common

#endif