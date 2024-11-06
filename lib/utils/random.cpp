#include "utils/random.h"

namespace utils {
Tensor gen_rand_tensor(std::vector<int64_t> shape, DataType dtype,
                       DeviceType device_type) {
    Tensor t(shape, dtype, device_type);
    auto ptr = t.data_ptr<TypeMap<DataType::FLOAT32>::type>();
    int64_t elems = t.elements_num();
    gen_rand(ptr, elems);
    return t;
}

}  // namespace utils