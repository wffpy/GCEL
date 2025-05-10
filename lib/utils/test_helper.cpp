#include "utils/test_helper.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils/log/Log.h"
#include "common/tensor/basic.h"

namespace utils {

template <class T>
std::string to_string(std::vector<T> vec) {
    std::string str = "[";
    for (auto& i : vec) {
        str += std::to_string(i);
        if (&i != &vec.back()) {
            str += ", ";
        }
    }
    return str + "]";
}

template <class T>
void print_tensor_(const Tensor& t) {
    int64_t ele_num = t.elements_num();
    auto data_ptr = t.data_ptr<T>();
    for (int64_t index = 0; index < ele_num; ++index) {
        std::cout << data_ptr[index] << " ";
    }
    std::cout << std::endl;
}

void print_tensor(const Tensor& t, std::string t_n) {
    if (!t_n.empty()) {
        LOG() << "Tensor name: " << t_n;
    }
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

template <typename T>
bool check_tensor_data(const Tensor& t1, const Tensor& t2,
                       float error_threshold) {
    std::vector<int64_t> indices;
    std::vector<std::pair<T, T>> mismatches;
    for (int64_t i = 0; i < t1.elements_num(); ++i) {
        auto error = abs(t1.data_ptr<T>()[i] - t2.data_ptr<T>()[i]);
        if (error > error_threshold) {
            indices.push_back(i);
            mismatches.push_back(
                std::make_pair(t1.data_ptr<T>()[i], t2.data_ptr<T>()[i]));
        }
    }
    if (indices.size() > 0) {
        LOG() << "Mismatch data found in tensors, Error indices: ";
        for (int64_t index = 0; index < indices.size(); ++index) {
            LOG() << "indice: " << indices[index]
                  << ", lhs: " << mismatches[index].first
                  << ", rhs: " << mismatches[index].second;
        }
    }
    return indices.size() == 0;
}

bool compare_tensor(const Tensor& t1, const Tensor& t2, float error_threshold) {
    // check device type
    if (t1.device_type() != t2.device_type()) {
        ELOG() << "Device type not match, lhs: "
               << get_device_str(t1.device_type())
               << ", rhs: " << get_device_str(t2.device_type());
    }

    // check data type
    if (t1.data_type() != t2.data_type()) {
        ELOG() << "Data type not match, lhs: " << get_dtype_str(t1.data_type())
               << ", rhs: " << get_dtype_str(t2.data_type());
    }

    // check shape rank
    auto shape1 = t1.shape();
    auto shape2 = t2.shape();
    CHECK(shape1.size() == shape2.size(), "Shape rank not match");

    // check shape
    if (shape1 != shape2) {
        ELOG() << "Shape not match, lhs: " << to_string(shape1)
               << ", rhs: " << to_string(shape2);
    }

    bool ret = true;
    DataType dtype = t1.data_type();
    // check data
    if (dtype == DataType::FLOAT32) {
        using cpp_type = TypeMap<DataType::FLOAT32>::type;
        ret = check_tensor_data<cpp_type>(t1, t2, error_threshold);
    } else if (dtype == DataType::BOOL) {
        using cpp_type = TypeMap<DataType::BOOL>::type;
        ret = check_tensor_data<cpp_type>(t1, t2, error_threshold);
    } else if (dtype == DataType::INT32) {
        using cpp_type = TypeMap<DataType::INT32>::type;
        ret = check_tensor_data<cpp_type>(t1, t2, error_threshold);
    } else {
        ELOG() << "Unsupported data type: " << get_dtype_str(t1.data_type());
    }
    return ret;
}

}  // namespace utils
