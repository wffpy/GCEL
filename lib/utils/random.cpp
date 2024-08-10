#include "utils/random.h"

namespace utils {

// void gen_rand_float(float* dst,int64_t size, float start, float end) {
//     // Create a random device and seed the random number generator
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     // Define the range for the random floats
//     std::uniform_real_distribution<> dis(start, end);

//     for (int64_t i = 0; i < size; i++) {
//         dst[i] = dis(gen);
//     }
// }

Tensor gen_rand_tensor(std::vector<int64_t> shape, DataType dtype, DeviceType device_type) {
    Tensor t(shape, dtype, device_type);
    auto ptr = t.data_ptr<TypeMap<DataType::FLOAT32>::type>();
    int64_t elems = t.elements_num();
    gen_rand(ptr, elems);
    return t;
}

}   // namespace utils