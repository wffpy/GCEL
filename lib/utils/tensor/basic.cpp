#include "utils/tensor/basic.h"

namespace utils {
int64_t get_dtype_bytes(DataType dtype) {
    switch(dtype) {
        case DataType::BOOL:
        case DataType::UINT8:
        case DataType::INT8:
            return 1;
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return 2;
        case DataType::FLOAT32:
            return 4;
        default:
            // throw std::runtime_error("Unsupported data type");
            return 4;
    }
    return 4;
}

}   // namespace utils