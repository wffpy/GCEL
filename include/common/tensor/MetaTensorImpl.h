#ifndef TENSOR_METATENSORIMPL_H
#define TENSOR_METATENSORIMPL_H
#include "common/tensor/TensorImpl.h"

namespace common {
// class GpuDataStorage : public DataStorage {
// public:
//     GpuDataStorage(const int64_t bytes, DeviceType device_type);
//     ~GpuDataStorage();
//     uint8_t* data() override;
// private:
//     uint8_t* data_;
// };

class MetaTensorImpl : public TensorImpl {
public:
    MetaTensorImpl(const std::vector<int64_t>& shape, DataType dtype);
    ~MetaTensorImpl() {};
    virtual void copy_from(std::shared_ptr<TensorImpl> src) override;
};

}   // namespace common

#endif