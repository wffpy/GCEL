#include "gpu/profiler/GpuProfiler.h"
#include "utils/log/Log.h"

namespace gpu_profiler {
GpuProfilerImpl::GpuProfilerImpl() {
    ELOG() << "GpuProfilerImpl is not implemented";
}

GpuProfilerImpl::~GpuProfilerImpl() {
    ELOG() << "GpuProfilerImpl is not implemented";
}

void GpuProfilerImpl::config() {
    ELOG() << "GpuProfilerImpl is not implemented";
}

void GpuProfilerImpl::start() {
    ELOG() << "GpuProfilerImpl is not implemented";
}

void GpuProfilerImpl::stop() {
    ELOG() << "GpuProfilerImpl is not implemented";
}

int64_t GpuProfilerImpl::get_duration() {
    ELOG() << "GpuProfilerImpl is not implemented";
    return 0;
}
}   // namespace gpu_profiler