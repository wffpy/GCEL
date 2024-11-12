#ifndef PORFILER_CPUPROFILER_H
#define PROFILER_CPUPROFILER_H
#include "common/profiler/profiler.h"
#include <chrono>

namespace cpu_profiler {

class CpuProfilerImpl : public profiler::ProfilerImpl {
public:
    CpuProfilerImpl();
    ~CpuProfilerImpl();
    virtual void  config() override;
    virtual void start() override;
    virtual void stop() override;
    virtual int64_t get_duration() override;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
};

}   // namespace cpu_profiler

#endif // PORFILER_CPUPROFILER_H