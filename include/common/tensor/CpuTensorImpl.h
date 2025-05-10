#ifndef TENSOR_CPUTENSORIMPL_H
#define TENSOR_CPUTENSORIMPL_H
#include "common/tensor/TensorImpl.h"

namespace common {
class CpuDataStorage : public DataStorage {
public:
    CpuDataStorage(const int64_t bytes, DeviceType device_type);
    ~CpuDataStorage();
    uint8_t* data() override;
private:
    uint8_t* data_;
};

class CpuTensorImpl : public TensorImpl {
public:
    CpuTensorImpl(const std::vector<int64_t>& shape, DataType dtype);
    ~CpuTensorImpl() {};
    virtual void copy_from(std::shared_ptr<TensorImpl> src) override;
};

}   // namespace common
#endif