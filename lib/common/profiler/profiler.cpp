#include "common/profiler/profiler.h"
#include "cpu/profiler/CpuProfiler.h"
#include "gpu/profiler/GpuProfiler.h"
#include "utils/log/Log.h"

namespace profiler {
Profiler::Profiler(ProfilerType type) : type_(type) {
    if (type_ == ProfilerType::CPU) {
        impl_ = std::make_shared<cpu_profiler::CpuProfilerImpl>();
    } else if (type_ == ProfilerType::GPU) {
        impl_ = std::make_shared<gpu_profiler::GpuProfilerImpl>();
    } else {
        ELOG() << "Invalid profiler type";
    }
}

Profiler::~Profiler() {
    ELOG() << "Please print the profiler results here.";
}

void Profiler::config() { impl_->config(); }

void Profiler::start() { impl_->start(); }

void Profiler::stop() { impl_->stop(); }

int64_t Profiler::get_duration() { return impl_->get_duration(); }

}   // namespace profiler