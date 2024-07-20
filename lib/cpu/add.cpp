#include "cpu/cpu.h"

namespace cpu {
int add(float *lhs, float* rhs, float* ret, int length) {
    for (int i = 0; i < length; ++i) {
        ret[i] = lhs[i] + rhs[i];
    }
    return 0;
}

}  // namespace cpu
