// Hooks on specific functions inside the game's main module (this build only, matched by build id).
// They replace engine functions whose output is useful to us but goes nowhere in a retail build.
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <unistd.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <csignal>
#include <vector>
#include <string>
#include <mutex>
#include <map>

#include "common/log.h"
#include "hle/os/os.h"
#include <atomic>

#include "runtime/runtime.h"

namespace dsr {
namespace {
float* g_game_step = nullptr;  // the game's per-frame time step (scratch page), when patched

// Replacement for the game's frame limiter, main+0x70bd90 (this, fps): called once per game update on the
// render thread. Same contract as the original: skip when the engine's global flag says so; sleep until a
// frame period has passed since the last call; remember the tick at this+0x310. Then the game's time step is
// set to the period that actually passed, and the frame-skip state (this+0x31d/0x31e/0x320) is kept clear.
void FrameLimiter(uint8_t* self, int /*fps*/) {
    const Runtime& rt = GetRuntime();
    if (*reinterpret_cast<const int32_t*>(rt.main.base() + 0x251fb4c) != 0) return;
    using clock = std::chrono::steady_clock;
    static clock::time_point last;
    const int fps = FrameRateNow();
    const auto period = std::chrono::nanoseconds(1'000'000'000 / fps);
    auto now = clock::now();
    if (last.time_since_epoch().count() != 0 && now < last + period) {
        std::this_thread::sleep_until(last + period);
        now = clock::now();
    }
    const double seconds = last.time_since_epoch().count() ? std::chrono::duration<double>(now - last).count() : 1.0 / fps;
    last = now;
    // Up to 1/20 s per step: slower devices (20-30 fps) keep the game at normal speed instead of slow motion.
    if (g_game_step) *g_game_step = float(std::clamp(seconds, 1.0 / 60.0, 1.0 / 20.0));
    *reinterpret_cast<int64_t*>(self + 0x310) = hle::os::NowTicks();
    self[0x31d] = 0;
    self[0x31e] = 0;
    *reinterpret_cast<double*>(self + 0x320) = 0;
}

// Dark Souls Remastered (Switch) 1.0.3, main build id 5a797a3a5da6c4f3468cb7c84e4dde63...
constexpr const char* kBuildId = "5a797a3a5da6c4f3468cb7c84e4dde63";

void Trim(char* s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

uintptr_t GuestOffset(const void* p) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const loader::Module& m = GetRuntime().main;
    return m.Contains(a) ? a - m.base() : 0;
}

// main+0x7fea18: engine debug print, (int level, const char* fmt, ...)
void EngineLog(int level, const char* fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt ? fmt : "", ap);
    va_end(ap);
    Trim(msg);
    if (msg[0]) LOGI("engine", "[%d] %s", level, msg);
}

// main+0x7feb30: same with a va_list
void EngineLogV(int level, const char* fmt, va_list ap) {
    char msg[2048];
    vsnprintf(msg, sizeof(msg), fmt ? fmt : "", ap);
    Trim(msg);
    if (msg[0]) LOGI("engine", "[%d] %s", level, msg);
}

// main+0x7892d8: DL_PANIC(const char* file, int line, const char* fmt, ...)
void EnginePanic(const char* file, int line, const char* fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt ? fmt : "", ap);
    va_end(ap);
    Trim(msg);
    const char* base = file ? strrchr(file, '/') : nullptr;
    FATAL("engine", "DL_PANIC: %s  (%s:%d, called from main+0x%lx)", msg, base ? base + 1 : (file ? file : "?"), line,
          (unsigned long)GuestOffset(__builtin_return_address(0)));
}

// main+0x81ad14: the pad device's SetAxis(this, axis, value), virtual. Each frame the game reads the axes
// (main+0x81a698: x / 32767, right stick at this+0x468/+0x46c), then the input mapping (main+0x834c1c and
// +0x834dc4) applies dead zone and scale, clamps to its maximum (1) and writes them back through here. The touch
// camera feeds the right stick past full deflection (dsr::input::AddLook) so fast swipes turn faster: for the
// right stick, a value still past 1 at this point can only come from that, and is kept instead of the clamped
// one. Controllers (|value| <= 1) behave as before.
void PadSetAxis(uint8_t* self, int axis, float value) {
    const unsigned i = unsigned(axis - 3);
    if (i > 5) return;
    float* slot = reinterpret_cast<float*>(self + 0x468) + i;
    if (i <= 1 && std::fabs(*slot) > 1.001f) return;  // right stick overdriven by a swipe: keep it
    *slot = value;
}

// ---- camera probe (diagnostics) ------------------------------------------------------------------------------
// main+0x8312d0: the input mapping's action writer (device, action, value): keeps the largest positive and
// negative contribution of the frame (device+0xd0 / +0xf8) and stores their sum in the action values
// (device+0x148). Reimplemented here to learn which action the right stick feeds: only a swipe's overdriven
// stick gives |value| > 1. That action's value is then watched: a background thread write-protects its page
// at random moments (20 ms apart), and the first access after that faults; the fault handler records who read
// it (pc, lr and the frame-pointer chain), unprotects the page and lets the access go on. The camera code that
// turns (and caps the turn speed) is among the readers. Logged every 10 s for 30 s, once per run.
namespace probe {
std::atomic<uint8_t*> g_dev{nullptr};
std::atomic<int> g_action{-1};
std::atomic<uintptr_t> g_page{0}, g_target{0};
std::atomic<bool> g_armed{false};
struct Sample {
    uintptr_t chain[8];
};
constexpr int kSamples = 4096;
Sample g_samples[kSamples];
std::atomic<int> g_nsamples{0};
struct sigaction g_old_segv;
size_t g_pagesize = 4096;

void OnFault(int sig, siginfo_t* si, void* uc_v) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(si->si_addr);
    const uintptr_t page = g_page.load(std::memory_order_acquire);
    if (page && addr >= page && addr < page + g_pagesize) {
        mprotect(reinterpret_cast<void*>(page), g_pagesize, PROT_READ | PROT_WRITE);
        g_armed.store(false, std::memory_order_release);
        const uintptr_t target = g_target.load(std::memory_order_relaxed);
        if (addr >= target && addr < target + 4) {
            const int i = g_nsamples.fetch_add(1, std::memory_order_relaxed);
            if (i < kSamples) {
#if defined(__aarch64__)
                auto* uc = static_cast<ucontext_t*>(uc_v);
                const loader::Module& m = GetRuntime().main;
                auto rel = [&](uintptr_t a) { return m.Contains(a) ? a - m.base() : 0; };
                Sample& smp = g_samples[i];
                memset(&smp, 0, sizeof(smp));
                smp.chain[0] = rel(uc->uc_mcontext.pc);
                smp.chain[1] = rel(uc->uc_mcontext.regs[30]);
                const uintptr_t* fp = reinterpret_cast<const uintptr_t*>(uc->uc_mcontext.regs[29]);
                for (int d = 2; d < 8 && fp && (reinterpret_cast<uintptr_t>(fp) & 7) == 0; d++) {
                    const uintptr_t lr = fp[1];
                    if (!m.Contains(lr)) break;
                    smp.chain[d] = rel(lr);
                    const uintptr_t* next = reinterpret_cast<const uintptr_t*>(fp[0]);
                    if (next <= fp) break;
                    fp = next;
                }
#endif
            }
        }
        return;
    }
    // Not ours: the previous handler (crash report).
    if (g_old_segv.sa_flags & SA_SIGINFO)
        g_old_segv.sa_sigaction(sig, si, uc_v);
    else if (g_old_segv.sa_handler != SIG_DFL && g_old_segv.sa_handler != SIG_IGN)
        g_old_segv.sa_handler(sig);
    else {
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

void Run() {
    pthread_setname_np(pthread_self(), "dsr-camprobe");
    g_pagesize = size_t(sysconf(_SC_PAGESIZE));
    struct sigaction sa {};
    sa.sa_sigaction = OnFault;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_old_segv);
    uint8_t* dev = g_dev.load();
    const int action = g_action.load();
    const uintptr_t target = reinterpret_cast<uintptr_t>(*reinterpret_cast<float**>(dev + 0x148) + action);
    g_target = target;
    LOGI("hooks", "camera probe: right stick feeds action %d (device %p, value at %p): watching its readers", action,
         dev, reinterpret_cast<void*>(target));
    uint32_t rnd = 12345;
    const auto start = std::chrono::steady_clock::now();
    auto report = start + std::chrono::seconds(10);
    for (;;) {
        rnd = rnd * 1664525u + 1013904223u;
        std::this_thread::sleep_for(std::chrono::microseconds(15000 + (rnd >> 8) % 10000));
        if (!g_armed.load()) {
            const uintptr_t page = target & ~(uintptr_t(g_pagesize) - 1);
            g_page.store(page, std::memory_order_release);
            g_armed = true;
            mprotect(reinterpret_cast<void*>(page), g_pagesize, PROT_NONE);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= report) {
            report += std::chrono::seconds(10);
            const int n = std::min(g_nsamples.exchange(0), kSamples);
            std::map<std::vector<uintptr_t>, int> counts;
            for (int i = 0; i < n; i++)
                counts[std::vector<uintptr_t>(g_samples[i].chain, g_samples[i].chain + 8)]++;
            for (const auto& [c, k] : counts) {
                std::string line;
                for (uintptr_t a : c) {
                    if (!a) break;
                    char buf[24];
                    snprintf(buf, sizeof(buf), " %lx", (unsigned long)a);
                    line += buf;
                }
                LOGI("hooks", "camera probe: action %d read %d times by:%s", action, k, line.c_str());
            }
            if (now - start > std::chrono::seconds(31)) {
                // Done: stop watching (the page is left writable).
                while (g_armed.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                g_page = 0;
                LOGI("hooks", "camera probe: done");
                return;
            }
        }
    }
}
}  // namespace probe

void ActionWrite(uint8_t* dev, int action, float v) {
    float* pos = *reinterpret_cast<float**>(dev + 0xd0);
    float* neg = *reinterpret_cast<float**>(dev + 0xf8);
    float p = pos[action], n = neg[action];
    if (v >= 0.0f) {
        if (p < v) pos[action] = p = v;
    } else if (n > v) {
        neg[action] = n = v;
    }
    (*reinterpret_cast<float**>(dev + 0x148))[action] = p + n;
    if (std::fabs(v) > 1.001f && probe::g_action.load(std::memory_order_relaxed) < 0) {
        int expected = -1;
        if (probe::g_action.compare_exchange_strong(expected, action)) {
            probe::g_dev = dev;
            std::thread(probe::Run).detach();
        }
    }
}

struct Hook {
    uint64_t offset;
    void* fn;
    const char* name;
};
const Hook kHooks[] = {
    {0x7fea18, reinterpret_cast<void*>(&EngineLog), "engine log"},
    {0x7feb30, reinterpret_cast<void*>(&EngineLogV), "engine vlog"},
    {0x7892d8, reinterpret_cast<void*>(&EnginePanic), "DL_PANIC"},
    {0x81ad14, reinterpret_cast<void*>(&PadSetAxis), "pad axis (touch camera speed)"},
    {0x8312d0, reinterpret_cast<void*>(&ActionWrite), "input action write (camera probe)"},
};

}  // namespace

float GameFrameTime() { return g_game_step ? *g_game_step : 1.0f / 30.0f; }
namespace {
std::atomic<int> g_movies{0};
}
void SetMoviePlaying(bool playing) {
    const int n = playing ? ++g_movies : --g_movies;
    LOGI("hooks", "movie %s: frame rate %d fps", playing ? "started" : "ended", n > 0 ? 30 : std::clamp(GetRuntime().config.target_fps, 30, 60));
}
int FrameRateNow() {
    return g_movies.load(std::memory_order_relaxed) > 0 ? 30 : std::clamp(GetRuntime().config.target_fps, 30, 60);
}

void InstallGameHooks() {
    Runtime& rt = GetRuntime();
    if (rt.main_nso.build_id_hex.rfind(kBuildId, 0) != 0) {
        LOGW("hooks", "unknown game build %.32s: engine hooks disabled", rt.main_nso.build_id_hex.c_str());
        return;
    }
    // Frame rate (fps = 60). The same two places a well-known Switch 60 fps mod changes, done differently:
    // 1. The game's per-frame update (main+0x66c99c) steps every system by a constant 1/30 s loaded from
    //    read-only data (main+0x66c9c8 adrp x8, page; +0x66c9cc ldr s8, [x8, #0x970]). It now loads a float in
    //    the scratch page instead.
    // 2. The render thread's frame limiter (main+0x70bd90, called once per game update with a hard-coded 30) is
    //    replaced by FrameLimiter: it paces to 60 and sets that float to the measured frame period (1/60..1/20 s),
    //    so the game keeps normal speed at 60 fps and when heavy areas run slower. It also keeps the game's
    //    frame-skip mode off (with it, slow frames ran two updates per shown frame).
    if (rt.config.target_fps > 30) {
        std::string err;
        float* step = reinterpret_cast<float*>(rt.main.ScratchPage());
        *step = 1.0f / 30.0f;
        const uint32_t adrp = rt.main.EncodeAdrp(0x66c9c8, 8, reinterpret_cast<uintptr_t>(step));
        bool ok = rt.main.PatchInstruction(0x66c9c8, 0xB00080A8, adrp, &err) &&
                  rt.main.PatchInstruction(0x66c9cc, 0xBD497108, 0xBD400108, &err);  // ldr s8, [x8]
        if (ok) ok = rt.main.HookFunction(0x70bd90, reinterpret_cast<void*>(&FrameLimiter), &err);
        if (ok) {
            g_game_step = step;
            LOGI("hooks", "60 fps: own frame limiter, game time step follows the frame time");
        } else {
            LOGW("hooks", "60 fps patches failed (%s): staying at 30 fps", err.c_str());
            rt.main.PatchInstruction(0x66c9cc, 0xBD400108, 0xBD497108, &err);  // undo, if the first ones applied
            rt.main.PatchInstruction(0x66c9c8, rt.main.EncodeAdrp(0x66c9c8, 8, reinterpret_cast<uintptr_t>(step)),
                                     0xB00080A8, &err);
            rt.config.target_fps = 30;
        }
    }
    for (const Hook& h : kHooks) {
        std::string err;
        if (rt.main.HookFunction(h.offset, h.fn, &err))
            LOGI("hooks", "hooked %s at main+0x%lx", h.name, (unsigned long)h.offset);
        else
            LOGW("hooks", "could not hook %s: %s", h.name, err.c_str());
    }
}

}  // namespace dsr
