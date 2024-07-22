#ifndef UTILS_TENSOR_TENSORIMPL_H
#define UTILS_TENSOR_TENSORIMPL_H
#include <memory>
#include <vector>

#include "utils/tensor/basic.h"
namespace utils {
class DataStorage {
  public:
    DataStorage(const int64_t bytes, DeviceType device_type);
    ~DataStorage();
    virtual uint8_t* data() = 0;
    DeviceType device_type() const;
    int64_t capacity() const;
  private:
    int64_t bytes_;
    DeviceType device_type_;
    
};

class TensorImpl {
  public:
    TensorImpl(const std::vector<int64_t>& shape, DataType dtype, DeviceType device_type);
    ~TensorImpl() {}
    const std::vector<int64_t>& shape() const;
    DeviceType device_type() const;
    DataType data_type() const;
    uint8_t* data() const;
    void set_storage(std::shared_ptr<DataStorage> storage);
    virtual void copy_from(std::shared_ptr<TensorImpl> src) = 0;
    int64_t capacity() const;
  private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    DeviceType device_type_;
    std::shared_ptr<DataStorage> storage_;
};

}   // namespace utils
#endif