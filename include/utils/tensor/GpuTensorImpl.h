#ifndef UTILS_TENSOR_GPUTENSOREMPL_H
#define UTILS_TENSOR_GPUTENSOREMPL_H
#include "utils/tensor/TensorImpl.h"

namespace utils {
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

}   // namespace utils

#endif