#include "common/platform.h"

#include <atomic>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#endif

#include "common/log.h"
#include "runtime/runtime.h"

namespace dsr::platform {
namespace {
std::atomic<CreateUserFileFn> g_create{nullptr};
}
// The cores to keep busy threads on (all but the slowest kind), computed once. count 0: one kind of core.
namespace {
#if defined(__ANDROID__)
struct FastCores {
    cpu_set_t set;
    int count = 0;
    std::string list;
};
const FastCores& GetFastCores() {
    static FastCores fc;
    static std::once_flag once;
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
        CPU_ZERO(&fc.set);
        if (!lowest || lowest == highest) return;  // one kind of core: nothing to avoid
        for (int cpu = 0; cpu < 32; cpu++)
            if (khz[cpu] > lowest) {
                CPU_SET(cpu, &fc.set);
                fc.list += " " + std::to_string(cpu);
                fc.count++;
            }
    });
    return fc;
}
// Android moves an app's threads between cpusets as it changes state (start-up, dialogs, focus), and on older
// kernels (4.x: Snapdragon 845) that resets their CPU affinity: the busy threads ended up on the efficiency
// cores for the whole session. A watchdog thread checks the registered threads every second and restores it.
struct Watched {
    pid_t tid;
    std::string who;
    int restores = 0;
};
std::mutex g_watch_mutex;
std::vector<Watched> g_watched;
// The cpuset a thread is in and that cpuset's CPUs, e.g. "/top-app (0-7)" (diagnostics).
std::string CpusetOf(pid_t tid) {
    char path[96], name[128] = {}, cpus[64] = {};
    snprintf(path, sizeof(path), "/proc/self/task/%d/cpuset", int(tid));
    if (FILE* f = fopen(path, "r")) {
        if (!fgets(name, sizeof(name), f)) name[0] = 0;
        fclose(f);
    }
    name[strcspn(name, "\n")] = 0;
    snprintf(path, sizeof(path), "/dev/cpuset%s/cpus", name);
    if (FILE* f = fopen(path, "r")) {
        if (!fgets(cpus, sizeof(cpus), f)) cpus[0] = 0;
        fclose(f);
    }
    cpus[strcspn(cpus, "\n")] = 0;
    return std::string(name[0] ? name : "?") + " (" + (cpus[0] ? cpus : "?") + ")";
}
// Online CPUs and their current clocks, e.g. "0:1766 1:1766 2:- 3:- 4:2803 ..." (MHz; "-" = offline). Phones
// take cores offline when hot (thermal) or idle: the affinity resets on the 6T came with that.
std::string CpuState() {
    std::string out;
    for (int cpu = 0; cpu < 16; cpu++) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", cpu);
        int online = 1;
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%d", &online) != 1) online = 1;
            fclose(f);
        } else if (cpu > 0) {
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
            if (FILE* d = fopen(path, "r")) fclose(d);
            else break;  // no such CPU
        }
        long khz = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
        if (online)
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%ld", &khz) != 1) khz = 0;
                fclose(f);
            }
        char item[32];
        if (!online) snprintf(item, sizeof(item), "%d:off ", cpu);
        else if (khz) snprintf(item, sizeof(item), "%d:%ld ", cpu, khz / 1000);
        else snprintf(item, sizeof(item), "%d:on ", cpu);
        out += item;
    }
    return out;
}
// First line of a small text file ("" if unreadable).
std::string ReadLine(const char* path) {
    char buf[256] = {};
    if (FILE* f = fopen(path, "r")) {
        if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
        fclose(f);
    }
    buf[strcspn(buf, "\n")] = 0;
    return buf;
}
// Why a thread may not use the performance cores: its cpuset, the CPUs the kernel allows it (Cpus_allowed_list)
// and the CPUs isolated by the Qualcomm core control (where the kernel shows them).
std::string WhyRefused(pid_t tid) {
    char path[96];
    snprintf(path, sizeof(path), "/proc/self/task/%d/status", int(tid));
    std::string allowed = "?";
    if (FILE* f = fopen(path, "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (!strncmp(line, "Cpus_allowed_list:", 18)) {
                line[strcspn(line, "\n")] = 0;
                const char* v = line + 18;
                while (*v == ' ' || *v == '\t') v++;
                allowed = v;
                break;
            }
        fclose(f);
    }
    std::string isolated = ReadLine("/sys/devices/system/cpu/isolated");
    std::string ctl;
    for (int cpu : {4, 6, 7}) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/core_ctl/active_cpus", cpu);
        std::string a = ReadLine(path);
        if (!a.empty()) {
            ctl = " | core control: " + a + " active of the cluster at CPU " + std::to_string(cpu);
            break;
        }
    }
    return "cpuset " + CpusetOf(tid) + ", allowed CPUs " + allowed + ", isolated " +
           (isolated.empty() ? std::string("?") : isolated) + ctl;
}
void WatchAffinity() {
    pthread_setname_np(pthread_self(), "dsr-affinity");
    const FastCores& fc = GetFastCores();
    // Checked 10 times a second: on the OnePlus 6T the system reset the affinity about once a second, so with
    // a one-second check the threads still spent much of their time on the efficiency cores.
    int restored = 0, refused = 0;
    std::string why;
    bool why_logged = false;
    auto summary_at = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::lock_guard<std::mutex> lk(g_watch_mutex);
        pid_t refused_tid = 0;
        for (size_t i = 0; i < g_watched.size();) {
            Watched& w = g_watched[i];
            cpu_set_t cur;
            CPU_ZERO(&cur);
            if (sched_getaffinity(w.tid, sizeof(cur), &cur) != 0) {  // thread gone
                g_watched.erase(g_watched.begin() + long(i));
                continue;
            }
            if (!CPU_EQUAL(&cur, &fc.set)) {
                const int r = sched_setaffinity(w.tid, sizeof(fc.set), &fc.set);
                (r == 0 ? restored : refused)++;
                if (r != 0 && !refused_tid) refused_tid = w.tid;
                if (++w.restores <= 2)
                    LOGI("platform", "%s (tid %d): CPU affinity reset by the system (%d CPUs allowed, cpuset %s), "
                         "restored to CPUs%s: %s", w.who.c_str(), int(w.tid), CPU_COUNT(&cur),
                         CpusetOf(w.tid).c_str(), fc.list.c_str(), r == 0 ? "ok" : "refused");
            }
            i++;
        }
        if (refused_tid && !why_logged) why = WhyRefused(refused_tid), why_logged = true;
        if (std::chrono::steady_clock::now() >= summary_at) {
            summary_at += std::chrono::seconds(10);
            if (restored || refused)
                LOGI("platform", "CPU affinity of the busy threads reset by the system and restored %d times in the "
                     "last 10 s (%d refused: none of CPUs%s online/allowed); CPUs now: %s", restored, refused,
                     fc.list.c_str(), CpuState().c_str());
            if (refused && !why.empty()) LOGI("platform", "refused: %s", why.c_str());
            restored = refused = 0;
            why.clear();
            why_logged = false;
        }
    }
}
#endif
}  // namespace

void PreferFastCores(const char* who) {
#if defined(__ANDROID__)
    thread_local bool done = false;
    if (done) return;
    done = true;
    if (!dsr::GetRuntime().config.fast_cores) return;
    const FastCores& fc = GetFastCores();
    if (fc.count <= 0) return;
    const int r = sched_setaffinity(0, sizeof(fc.set), &fc.set);
    LOGI("platform", "%s kept on CPUs%s: %s", who, fc.list.c_str(), r == 0 ? "ok" : "refused");
    {
        std::lock_guard<std::mutex> lk(g_watch_mutex);
        g_watched.push_back({pid_t(gettid()), who});
    }
    static std::once_flag watch_once;
    std::call_once(watch_once, [] { std::thread(WatchAffinity).detach(); });
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
