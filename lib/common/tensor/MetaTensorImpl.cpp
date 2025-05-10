#include "common/tensor/MetaTensorImpl.h"
#include <numeric>
#include <iostream> 
#include "utils/log/Log.h"

namespace common {

MetaTensorImpl::MetaTensorImpl(const std::vector<int64_t>& shape, DataType dtype) : TensorImpl(shape, dtype, DeviceType::GPU) {
    set_storage(nullptr);
}

void MetaTensorImpl::copy_from(std::shared_ptr<TensorImpl> src) {
    ELOG() << "MetaTensorImpl::copy_from: not implemented yet";
}

}   // namespace common