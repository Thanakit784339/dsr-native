#include "common/profiler.h"

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "runtime/runtime.h"

#if defined(__aarch64__) && defined(__linux__)
#include <ucontext.h>
#define DSR_PROFILER_SUPPORTED 1
#endif

namespace dsr::profiler {
namespace {
constexpr const char* kTag = "profile";
constexpr int kMaxThreads = 64;
constexpr int kDepth = 16;
constexpr size_t kMaxSamples = 1 << 16;

struct ThreadSlot {
    std::atomic<int> tid{0};
    char name[32] = {};
    uintptr_t stack_lo = 0, stack_hi = 0;
    timer_t timer{};
    bool timer_ok = false;
};
struct Sample {
    int16_t slot;
    uint8_t depth;
    uintptr_t pc[kDepth];  // pc, then return addresses (frame-pointer walk)
};

ThreadSlot g_slots[kMaxThreads];
std::atomic<int> g_nslots{0};
Sample* g_samples = nullptr;  // ring written by the signal handler
std::atomic<size_t> g_write{0};
std::atomic<bool> g_enabled{false};
std::mutex g_mutex;

int Gettid() { return int(syscall(SYS_gettid)); }

int SlotOf(int tid) {
    const int n = g_nslots.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++)
        if (g_slots[i].tid.load(std::memory_order_relaxed) == tid) return i;
    return -1;
}

#ifdef DSR_PROFILER_SUPPORTED
// Async-signal-safe: no locks, no allocation, no TLS (guest code may be running on this thread).
void OnSignal(int, siginfo_t*, void* uc_) {
    const int slot = SlotOf(Gettid());
    if (slot < 0 || !g_samples) return;
    const auto* uc = static_cast<const ucontext_t*>(uc_);
    const size_t at = g_write.fetch_add(1, std::memory_order_relaxed) & (kMaxSamples - 1);
    Sample& s = g_samples[at];
    s.slot = int16_t(slot);
    s.pc[0] = uintptr_t(uc->uc_mcontext.pc);
    int d = 1;
    s.pc[d++] = uintptr_t(uc->uc_mcontext.regs[30]);  // lr: the caller, when pc is in a leaf
    uintptr_t fp = uintptr_t(uc->uc_mcontext.regs[29]);
    const uintptr_t lo = g_slots[slot].stack_lo, hi = g_slots[slot].stack_hi;
    while (d < kDepth && fp >= lo && fp + 16 <= hi && (fp & 7) == 0) {
        const uintptr_t next = reinterpret_cast<const uintptr_t*>(fp)[0];
        const uintptr_t ret = reinterpret_cast<const uintptr_t*>(fp)[1];
        if (!ret) break;
        s.pc[d++] = ret;
        if (next <= fp) break;
        fp = next;
    }
    s.depth = uint8_t(d);
}
#endif

// ---- symbolization (reporting thread) ----
struct Where {
    int kind;  // 0 game (main), 1 our library, 2 other library, 3 unknown
    uintptr_t off;
    std::string label;
};
std::unordered_map<uintptr_t, Where> g_where_cache;
uintptr_t g_self_base = 0;

Where Locate(uintptr_t a) {
    auto it = g_where_cache.find(a);
    if (it != g_where_cache.end()) return it->second;
    Where w{3, a, ""};
    const loader::Module& m = GetRuntime().main;
    char buf[160];
    if (m.base() && m.Contains(a)) {
        w = {0, a - m.base(), ""};
        snprintf(buf, sizeof(buf), "main+0x%lx", (unsigned long)(a - m.base()));
        w.label = buf;
    } else {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(a), &info) && info.dli_fname) {
            const uintptr_t base = uintptr_t(info.dli_fbase);
            const char* file = strrchr(info.dli_fname, '/');
            file = file ? file + 1 : info.dli_fname;
            if (base == g_self_base) {
                w = {1, a - base, ""};
                snprintf(buf, sizeof(buf), "dsr+0x%lx", (unsigned long)(a - base));
            } else {
                w = {2, a - base, ""};
                snprintf(buf, sizeof(buf), "%s+0x%lx", file, (unsigned long)(a - base));
            }
            w.label = buf;
            if (info.dli_sname) {
                w.label += " ";
                w.label += std::string(info.dli_sname).substr(0, 60);
            }
        } else {
            snprintf(buf, sizeof(buf), "0x%lx", (unsigned long)a);
            w.label = buf;
        }
    }
    if (g_where_cache.size() > 200000) g_where_cache.clear();
    g_where_cache[a] = w;
    return w;
}

void Report() {
    std::vector<Sample> samples;
    {
        const size_t end = g_write.load();
        static size_t read = 0;
        const size_t begin = end - read > kMaxSamples ? end - kMaxSamples : read;
        for (size_t i = begin; i < end; i++) samples.push_back(g_samples[i & (kMaxSamples - 1)]);
        read = end;
    }
    if (samples.empty()) return;
    const int n = g_nslots.load();
    for (int slot = 0; slot < n; slot++) {
        std::map<std::string, int> self, ours, game, libs;
        int total = 0;
        for (const Sample& s : samples) {
            if (s.slot != slot) continue;
            total++;
            // Self: the pc, labelled down to the function where our symbols allow it (rounded to 64 bytes so
            // nearby instructions add up; exact offsets are not needed to find a function).
            Where w = Locate(s.pc[0]);
            {  // by library: the part of the label before '+'
                const size_t plus = w.label.find('+');
                libs[w.kind == 0 ? "game" : w.kind == 1 ? "ours" : plus != std::string::npos ? w.label.substr(0, plus) : "?"]++;
            }
            self[w.kind == 1 || w.kind == 0 ? Locate(s.pc[0] & ~uintptr_t(63)).label : w.label]++;
            bool got_ours = false, got_game = false;
            for (int d = 0; d < s.depth && (!got_ours || !got_game); d++) {
                const uintptr_t a = d == 0 ? s.pc[0] : s.pc[d] - 4;  // return address -> the call
                const Where x = Locate(a);
                if (x.kind == 1 && !got_ours) {
                    ours[Locate(a & ~uintptr_t(63)).label]++;
                    got_ours = true;
                }
                if (x.kind == 0 && !got_game) {
                    game[Locate(a & ~uintptr_t(63)).label]++;
                    got_game = true;
                }
            }
        }
        if (total < 50) continue;
        // Current name (game threads are named after they start) and exact CPU time since the last report.
        const int tid = g_slots[slot].tid.load();
        char path[64], name[32] = {};
        snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
        if (FILE* f = fopen(path, "r")) {
            if (fgets(name, sizeof(name), f)) name[strcspn(name, "\n")] = 0;
            fclose(f);
        }
        if (name[0]) snprintf(g_slots[slot].name, sizeof(g_slots[slot].name), "%s", name);
        double cpu_s = -1;
        snprintf(path, sizeof(path), "/proc/self/task/%d/schedstat", tid);
        if (FILE* f = fopen(path, "r")) {
            unsigned long long ns = 0;
            if (fscanf(f, "%llu", &ns) == 1) {
                static unsigned long long last_ns[kMaxThreads] = {};
                if (last_ns[slot]) cpu_s = (ns - last_ns[slot]) / 1e9;
                last_ns[slot] = ns;
            }
            fclose(f);
        }
        auto top = [&](const char* what, const std::map<std::string, int>& m) {
            std::vector<std::pair<int, std::string>> v;
            for (auto& [k, c] : m) v.push_back({c, k});
            std::sort(v.rbegin(), v.rend());
            std::string line;
            for (size_t i = 0; i < v.size() && i < 14; i++) {
                char b[200];
                snprintf(b, sizeof(b), "%s%.1f%% %s", i ? " | " : "", 100.0 * v[i].first / total, v[i].second.c_str());
                line += b;
            }
            LOGI(kTag, "  %s: %s", what, line.c_str());
        };
        // The CPU it last ran on (field 39 of /proc/.../stat), to see which cores the busy threads get.
        int cpu = -1;
        snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
        if (FILE* f = fopen(path, "r")) {
            char buf[1024];
            const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = 0;
            fclose(f);
            const char* p = strrchr(buf, ')');  // the name may contain spaces
            for (int field = 2; p && *p && field < 39; p++)
                if (*p == ' ') field++;
            if (p) cpu = atoi(p);
        }
        LOGI(kTag, "thread '%s' (tid %d, last on CPU %d): %d samples, %.1f s of CPU in the last 10 s", g_slots[slot].name,
             tid, cpu, total, cpu_s);
        top("by library", libs);
        top("self", self);
        top("in our code", ours);
        top("in game code", game);
    }
}

void StartReporter() {
    static bool started = false;
    if (started) return;
    started = true;
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            Report();
        }
    }).detach();
}

void StartTimer(ThreadSlot& t) {
#ifdef DSR_PROFILER_SUPPORTED
    sigevent sev{};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGPROF;
#ifdef sigev_notify_thread_id
    sev.sigev_notify_thread_id = t.tid.load();
#else
    sev._sigev_un._tid = t.tid.load();  // glibc spells the field this way
#endif
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &t.timer) != 0) return;
    itimerspec its{};
    its.it_interval.tv_nsec = 2'000'000;
    its.it_value.tv_nsec = 2'000'000;
    t.timer_ok = timer_settime(t.timer, 0, &its, nullptr) == 0;
#else
    (void)t;
#endif
}
}  // namespace

bool Enabled() { return g_enabled.load(); }

void Enable() {
#ifdef DSR_PROFILER_SUPPORTED
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_enabled.exchange(true)) return;
    g_samples = new Sample[kMaxSamples]();
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&Enable), &info)) g_self_base = uintptr_t(info.dli_fbase);
    struct sigaction sa {};
    sa.sa_sigaction = OnSignal;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, nullptr);
    StartReporter();
    LOGI(kTag, "sampling profiler on (500 Hz of CPU time per registered thread); our library at %p",
         reinterpret_cast<void*>(g_self_base));
#endif
}

void RegisterCurrentThread(const char* name) {
#ifdef DSR_PROFILER_SUPPORTED
    if (!g_enabled.load()) return;
    std::lock_guard<std::mutex> lk(g_mutex);
    const int tid = Gettid();
    if (SlotOf(tid) >= 0) return;
    const int i = g_nslots.load();
    if (i >= kMaxThreads) return;
    ThreadSlot& t = g_slots[i];
    snprintf(t.name, sizeof(t.name), "%s", name ? name : "?");
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
        void* lo = nullptr;
        size_t sz = 0;
        pthread_attr_getstack(&attr, &lo, &sz);
        t.stack_lo = uintptr_t(lo);
        t.stack_hi = uintptr_t(lo) + sz;
        pthread_attr_destroy(&attr);
    }
    t.tid.store(tid);
    g_nslots.store(i + 1, std::memory_order_release);
    StartTimer(t);
#else
    (void)name;
#endif
}

}  // namespace dsr::profiler
