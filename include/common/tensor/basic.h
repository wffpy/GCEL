#ifndef COMMON_TENSORE_BASIC_H
#define COMMON_TENSORE_BASIC_H
#include <cstdint>
#include <string>
#include "common/tensor/types.h"

namespace common {
enum class DataType {BOOL, INT8, UINT8, FLOAT16, BFLOAT16, INT32, UINT32, FLOAT32};
enum class DeviceType {CPU, GPU, META};

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
DECLARE_TYPEMAP(DataType::FLOAT16, float16);
DECLARE_TYPEMAP(DataType::BFLOAT16, bfloat16);
DECLARE_TYPEMAP(DataType::INT32, int32_t);
DECLARE_TYPEMAP(DataType::UINT32, uint32_t);
DECLARE_TYPEMAP(DataType::FLOAT32, float);


template <class T>
struct CppTypeMap;

#define DECLARE_CPPTYPEMAP(cpp_type, data_type) \
template<>  \
struct CppTypeMap<cpp_type> { \
    static const DataType value = data_type; \
};

DECLARE_CPPTYPEMAP(bool, DataType::BOOL);
DECLARE_CPPTYPEMAP(int8_t, DataType::INT8);
DECLARE_CPPTYPEMAP(uint8_t, DataType::UINT8);
DECLARE_CPPTYPEMAP(float16, DataType::FLOAT16);
DECLARE_CPPTYPEMAP(bfloat16, DataType::BFLOAT16);
DECLARE_CPPTYPEMAP(int32_t, DataType::INT32);
DECLARE_CPPTYPEMAP(uint32_t, DataType::UINT32);
DECLARE_CPPTYPEMAP(float, DataType::FLOAT32);

std::string get_dtype_str(DataType dtype);

std::string get_device_str(DeviceType device);

}   // namespace common
#endif