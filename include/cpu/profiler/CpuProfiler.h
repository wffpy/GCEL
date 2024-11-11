#ifndef PORFILER_CPUPROFILER_H
#define PROFILER_CPUPROFILER_H
#include "common/profiler/profiler.h"

namespace cpu_profiler {

class CpuProfilerImpl : public profiler::ProfilerImpl {
public:
    CpuProfilerImpl();
    ~CpuProfilerImpl();
    virtual void  config() override;
    virtual void start() override;
    virtual void stop() override;

};

}   // namespace cpu_profiler

#endif // PORFILER_CPUPROFILER_H