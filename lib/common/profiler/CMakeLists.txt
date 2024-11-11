add_library(PROFILER STATIC profiler.cpp)

target_link_libraries(
    PROFILER
    CPU_PROFILER
    GPU_PROFILER
)