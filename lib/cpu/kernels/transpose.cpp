#include <algorithm>

#include "cpu/kernels/cpu.h"
#include "utils/log/Log.h"

namespace cpu {
template <typename T>
utils::GCELResult transpose(const T* input, T* ret, int64_t row, int64_t col,
                            int64_t row_block_size, int64_t col_block_size) {
    for (int64_t ii = 0; ii < row; ii += row_block_size) {
        int64_t block_row_end = std::min(row, ii + row_block_size);
        for (int64_t jj = 0; jj < col; jj += col_block_size) {
            int64_t block_col_end = std::min(col, jj + col_block_size);
            for (int64_t b_i = ii; b_i < block_row_end; b_i++) {
                for (int64_t b_j = jj; b_j < block_col_end; b_j++) {
                    ret[b_j * row + b_i] = input[b_i * col + b_j];
                }
            }
        }
    }
    return utils::GCELResult::SUCCESS;
}

#define INSTANTIATE_TRANSPOSE(T)                                            \
    template utils::GCELResult transpose<T>(const T*, T*, int64_t, int64_t, \
                                            int64_t, int64_t);

INSTANTIATE_TRANSPOSE(char);
INSTANTIATE_TRANSPOSE(int32_t);
INSTANTIATE_TRANSPOSE(int64_t);
INSTANTIATE_TRANSPOSE(float);
INSTANTIATE_TRANSPOSE(double);
}  // namespace cpu
