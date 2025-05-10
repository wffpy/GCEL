#include "meta/meta_ops.h"

namespace meta {
Tensor add(const Tensor& a, const Tensor& b) {
    auto shape = a.shape();
    auto dtype = a.data_type();
    Tensor result(shape, dtype, DeviceType::META);
    return result;
}
}   // namespace meta