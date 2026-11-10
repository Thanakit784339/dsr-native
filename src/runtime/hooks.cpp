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

// main+0x81a5dc: the pad device's GetAxis(this, axis), virtual. Diagnostics for the touch camera: the call
// chains (frame-pointer walk, 8 levels) that read the right stick (axes 3, 4), logged every 10 s with counts,
// to find where the camera turns.
std::mutex g_axis_mutex;
std::map<std::vector<uintptr_t>, uint64_t> g_axis_chains;
std::chrono::steady_clock::time_point g_axis_report;
float PadGetAxis(uint8_t* self, int axis) {
    const unsigned i = unsigned(axis - 3);
    if (i > 5) return 0.0f;
    const float v = reinterpret_cast<const float*>(self + 0x468)[i];
    if (i == 0) {
        const loader::Module& m = GetRuntime().main;
        std::vector<uintptr_t> chain;
        chain.push_back(reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
        // Our frame record: [0] = the guest caller's frame pointer, then each record {fp, lr}.
        const uintptr_t* fp = static_cast<const uintptr_t*>(__builtin_frame_address(0));
        if (fp) fp = reinterpret_cast<const uintptr_t*>(fp[0]);
        for (int d = 0; d < 8 && fp && (reinterpret_cast<uintptr_t>(fp) & 7) == 0; d++) {
            const uintptr_t lr = fp[1];
            if (!m.Contains(lr)) break;
            chain.push_back(lr);
            const uintptr_t* next = reinterpret_cast<const uintptr_t*>(fp[0]);
            if (next <= fp) break;
            fp = next;
        }
        for (uintptr_t& a : chain) a = m.Contains(a) ? a - m.base() : 0;
        std::lock_guard<std::mutex> lk(g_axis_mutex);
        g_axis_chains[chain]++;
        const auto now = std::chrono::steady_clock::now();
        if (now - g_axis_report > std::chrono::seconds(10)) {
            g_axis_report = now;
            for (const auto& [c, n] : g_axis_chains) {
                std::string line;
                for (uintptr_t a : c) {
                    char buf[24];
                    snprintf(buf, sizeof(buf), " %lx", (unsigned long)a);
                    line += buf;
                }
                LOGI("hooks", "right stick x read %llu times by:%s", (unsigned long long)n, line.c_str());
            }
            g_axis_chains.clear();
        }
    }
    return v;
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
    {0x81a5dc, reinterpret_cast<void*>(&PadGetAxis), "pad axis read (diagnostics)"},
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
