#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <initializer_list>
#include <cstdint>
#include <memory>
#include <numeric>
namespace utils {

enum class DataType {BOOL, INT8, UINT8, FLOAT16, BFLOAT16, INT32, UINT32, FLOAT32};

class DataPtr {
public:
    DataPtr() = delete;
    DataPtr(int64_t bytes);
    ~DataPtr();
    uint8_t* data() { return ptr; }
private:
    uint8_t* ptr;
};

template <typename DT>
class Tensor {
public:
    Tensor (std::initializer_list<int64_t> dims);
    Tensor (std::vector<int64_t>& dims);
    const std::vector<int64_t>& shape();
    int64_t elements_num();
    DT* data_ptr();
private:
    std::vector<int64_t> shape_;
    std::shared_ptr<DataPtr> data_ptr_;
};

template <typename DT>
Tensor<DT>::Tensor(std::initializer_list<int64_t> dims) : shape_(dims) {
    int64_t dt_bytes = sizeof(DT);
    int64_t bytes = std::accumulate(shape_.begin(), shape_.end(), dt_bytes, std::multiplies<int64_t>());
    data_ptr_ = std::make_shared<DataPtr>(bytes);
}

template <typename DT>
Tensor<DT>::Tensor(std::vector<int64_t>& dims) : shape_(dims) {
    int64_t dt_bytes = sizeof(DT);
    int64_t bytes = std::accumulate(shape_.begin(), shape_.end(), dt_bytes, std::multiplies<int64_t>());
    data_ptr_ = std::make_shared<DataPtr>(bytes);
}

template <typename DT>
const std::vector<int64_t>& Tensor<DT>::shape() {
    return shape_;
}

template <typename DT>
DT* Tensor<DT>::data_ptr() {
    return (DT*)(data_ptr_->data());
}

template <typename DT>
int64_t Tensor<DT>::elements_num() {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
}

}   // namespace utils
#endif