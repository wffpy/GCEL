#include "utils/tensor/basic.h"
#include <string>

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

std::string get_dtype_str(DataType dtype) {
    switch(dtype) {
        case DataType::BOOL:
            return "bool";
        case DataType::UINT8:
            return "uint8";
        case DataType::INT8:
            return "int8";
        case DataType::FLOAT16:
            return "float16";
        case DataType::BFLOAT16:
            return "bfloat16";
        case DataType::FLOAT32:
            return "float32";
        default:
            // throw std::runtime
            return "unknown";
    }
    return "unknown";
}

std::string get_device_str(DeviceType device) {
    switch(device) {
        case DeviceType::CPU:
            return "cpu";
        case DeviceType::GPU:
            return "gpu";
        default:
            // throw std::runtime_error("Unsupported device type");
            return "unknown";
    }
    return "unknown";
}
}   // namespace utils