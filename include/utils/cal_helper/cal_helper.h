#ifndef UTIL_CAL_HELPER_H
#define UTIL_CAL_HELPER_H
#include <cstdint>
#include <type_traits>

#define IS_TYPE(T, Type) (std::is_same<T, Type>::value)


namespace cal_helper {
int64_t roundUp(int64_t num_to_round, int64_t multiple = 32);
}   // namespace cal_helper
#endif  // UTIL_CAL_HELPER_H