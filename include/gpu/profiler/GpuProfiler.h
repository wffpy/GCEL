#ifndef PORFILER_GPUPROFILER_H
#define PROFILER_GPUPROFILER_H
#include "common/profiler/profiler.h"

namespace gpu_profiler {

class GpuProfilerImpl : public profiler::ProfilerImpl {
public:
    GpuProfilerImpl();
    ~GpuProfilerImpl();
    virtual void  config() override;
    virtual void start() override;
    virtual void stop() override;
    virtual int64_t get_duration() override;

};

}   // namespace gpu_profiler

#endif // PORFILER_GPUPROFILER_H