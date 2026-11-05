#include "common/platform.h"

#include <atomic>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <sched.h>
#include <cstdio>
#endif

#include "common/log.h"
#include "runtime/runtime.h"

namespace dsr::platform {
namespace {
std::atomic<CreateUserFileFn> g_create{nullptr};
}
void PreferFastCores(const char* who) {
#if defined(__ANDROID__)
    thread_local bool done = false;
    if (done) return;
    done = true;
    if (!dsr::GetRuntime().config.fast_cores) return;
    static cpu_set_t set;
    static int count = -1;
    static std::once_flag once;
    static std::string list;
    std::call_once(once, [] {
        long khz[32] = {}, lowest = 0, highest = 0;
        for (int cpu = 0; cpu < 32; cpu++) {
            char path[96];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%ld", &khz[cpu]) != 1) khz[cpu] = 0;
                fclose(f);
            }
            if (khz[cpu] > 0 && (!lowest || khz[cpu] < lowest)) lowest = khz[cpu];
            if (khz[cpu] > highest) highest = khz[cpu];
        }
        CPU_ZERO(&set);
        count = 0;
        if (!lowest || lowest == highest) return;  // one kind of core: nothing to avoid
        for (int cpu = 0; cpu < 32; cpu++)
            if (khz[cpu] > lowest) {
                CPU_SET(cpu, &set);
                list += " " + std::to_string(cpu);
                count++;
            }
    });
    if (count <= 0) return;
    const int r = sched_setaffinity(0, sizeof(set), &set);
    LOGI("platform", "%s kept on CPUs%s: %s", who, list.c_str(), r == 0 ? "ok" : "refused");
#else
    (void)who;
#endif
}
void SetCreateUserFile(CreateUserFileFn fn) { g_create.store(fn); }
int CreateUserFile(const char* name, const char* mime_type) {
    CreateUserFileFn fn = g_create.load();
    return fn ? fn(name, mime_type) : -1;
}
namespace {
std::atomic<RequestTextFn> g_request_text{nullptr};
std::mutex g_text_mutex;
bool g_text_ready = false, g_text_ok = false;
std::u16string g_text;
}  // namespace
void SetRequestText(RequestTextFn fn) { g_request_text.store(fn); }
bool RequestText(const char* title, const std::u16string& initial, int max_length) {
    {
        std::lock_guard<std::mutex> lk(g_text_mutex);
        g_text_ready = false;
    }
    RequestTextFn fn = g_request_text.load();
    return fn && fn(title, initial, max_length);
}
void DeliverText(bool ok, const std::u16string& text) {
    std::lock_guard<std::mutex> lk(g_text_mutex);
    g_text_ready = true;
    g_text_ok = ok;
    g_text = text;
}
bool TakeText(bool* ok, std::u16string* text) {
    std::lock_guard<std::mutex> lk(g_text_mutex);
    if (!g_text_ready) return false;
    g_text_ready = false;
    *ok = g_text_ok;
    *text = g_text;
    return true;
}
}  // namespace dsr::platform

namespace dsr::platform {
bool NativeCodeOptimized() {
#ifdef __OPTIMIZE__
    return true;
#else
    return false;
#endif
}
}  // namespace dsr::platform
