#ifndef UTILS_TENSORE_BASIC_H
#define UTILS_TENSORE_BASIC_H
#include <cstdint>
#include <string>

namespace utils {
enum class DataType {BOOL, INT8, UINT8, FLOAT16, BFLOAT16, INT32, UINT32, FLOAT32};
enum class DeviceType {CPU, GPU};

int64_t get_dtype_bytes(DataType dtype);

template <DataType T>
struct TypeMap;

#define DECLARE_TYPEMAP(data_type, cpp_type) \
template <> \
struct TypeMap<data_type> { \
    using type = cpp_type; \
};

DECLARE_TYPEMAP(DataType::BOOL, bool);
DECLARE_TYPEMAP(DataType::INT8, int8_t);
DECLARE_TYPEMAP(DataType::UINT8, uint8_t);
// DECLARE_TYPEMAP(DataType::FLOAT16, float);
// DECLARE_TYPEMAP(DataType::BFLOAT16, float);
DECLARE_TYPEMAP(DataType::INT32, int32_t);
DECLARE_TYPEMAP(DataType::UINT32, uint32_t);
DECLARE_TYPEMAP(DataType::FLOAT32, float);

std::string get_dtype_str(DataType dtype);

std::string get_device_str(DeviceType device);

}   // namespace utils
#endif