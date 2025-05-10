#ifndef META_OPS_H
#define META_OPS_H
#include "common/tensor/tensor.h"
using namespace common;

namespace meta {
Tensor add(const Tensor &lhs, const Tensor &rhs);

} // namespace meta
#endif