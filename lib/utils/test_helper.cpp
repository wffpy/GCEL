#include <iostream>
#include "utils/test_helper.h"
#include "utils/tensor/basic.h"
#include "utils/log/Log.h"

namespace utils {

template <class T>
void print_tensor_(const Tensor& t) {
    int64_t ele_num = t.elements_num();
    auto data_ptr = t.data_ptr<T>();
    for (int64_t index = 0; index <ele_num; ++index) {
        std::cout << data_ptr[index] << " ";
    }
    std::cout << std::endl;
}

void print_tensor(const Tensor& t) {
    if (t.data_type() == DataType::BOOL) {
        print_tensor_<bool>(t);
    } else if (t.data_type() == DataType::UINT8) {
        print_tensor_<uint8_t>(t);
    } else if (t.data_type() == DataType::INT32) {
        print_tensor_<int32_t>(t);
    } else if (t.data_type() == DataType::FLOAT32) {
        print_tensor_<float>(t);
    } else {
        ELOG() << "Unsupported data type: " << get_dtype_str(t.data_type());
    }
}

}   // namepsace utils
