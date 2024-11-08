#include "utils/cal_helper/cal_helper.h"
#include "utils/log/Log.h"

namespace cal_helper {  
int64_t roundUp(int64_t num_to_round, int64_t multiple) {
    if (multiple == 0) {
        ELOG() << "Multiple cannot be 0";
    }

    return (num_to_round + multiple - 1) / multiple * multiple;
}


}   // namespace cal_helper