#include "cpu/profiler/CpuProfiler.h"
#include "utils/log/Log.h"

#include <chrono>

namespace cpu_profiler {
CpuProfilerImpl::CpuProfilerImpl() {
    ELOG() << "CpuProfilerImpl is not implemented yet";
}

CpuProfilerImpl::~CpuProfilerImpl() {
    ELOG() << "CpuProfilerImpl is not implemented yet";
}

void CpuProfilerImpl::config() {
    DLOG() << "call " << __FUNCTION__;
}

void CpuProfilerImpl::start() {
    DLOG() << "call " << __FUNCTION__;
    start_time_ = std::chrono::high_resolution_clock::now();
}

void CpuProfilerImpl::stop() {
    DLOG() << "call " << __FUNCTION__;
    stop_time_ = std::chrono::high_resolution_clock::now();
}

int64_t CpuProfilerImpl::get_duration() {
    DLOG() << "call " << __FUNCTION__;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_)
        .count();
}

}   // namespace cpu_profiler