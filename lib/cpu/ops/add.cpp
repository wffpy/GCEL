#include "cpu/ops/cpu_ops.h"
#include "cpu/kernels/cpu_kernels.h"
#include "meta/meta_ops.h"

namespace cpu {
Tensor add(const Tensor& lhs, const Tensor& rhs) {
   Tensor meta_ret = meta::add(lhs, rhs);
   Tensor cpu_ret(meta_ret.shape(), meta_ret.data_type());
   int64_t elem_num = meta_ret.elements_num();

    if (lhs.data_type() == DataType::FLOAT32) {
        cpu_kernels::add<float>(lhs.data_ptr<float>(), rhs.data_ptr<float>(), cpu_ret.data_ptr<float>(), elem_num);
    } else {
        throw std::runtime_error("Unsupported data type for add operation.");
    }
   return cpu_ret;
}

}   // namespace cpu
