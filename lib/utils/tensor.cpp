#include "utils/tensor.h"

namespace utils {

DataPtr::DataPtr(int64_t bytes) {
    ptr = new uint8_t[bytes];
}

DataPtr::~DataPtr() {
    delete[] ptr;
}

}   // namespace utils 