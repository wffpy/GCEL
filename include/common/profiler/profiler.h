#ifndef PORFILER_H
#define PORFILER_H
#include <memory>

namespace profiler {

enum class ProfilerType {
    CPU = 0, GPU = 1
};

class ProfilerImpl;

class Profiler {
public:
    Profiler(ProfilerType type);
    virtual ~Profiler();
    void config();
    void start();
    void stop();
    int64_t get_duration();
private:
    ProfilerType type_;
    std::shared_ptr<ProfilerImpl> impl_;
};

class ProfilerImpl {
public:
    virtual void config() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual int64_t get_duration() = 0;
};

} // namespace profiler

#endif
