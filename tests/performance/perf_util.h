#include "common/profiler/profiler.h"

#define PROFILE_SCOPE_BEGIN(num) \
    { \
        profiler::Profiler profiler(profiler::ProfilerType::GPU); \
        profiler.config(); \
        profiler.start(); \
        for (int64_t profile_index = 0; profile_index < num; ++profile_index) { 


#define PROFILE_SCOPE_END() \
        } \
        profiler.stop(); \
    }

