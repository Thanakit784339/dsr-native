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
#include <vector>
#include <string>
#include <mutex>
#include <map>

#include "common/log.h"
#include "hle/input.h"
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

// ---- touch camera: fast swipes past the game's caps --------------------------------------------------------
// The right stick reaches the camera as input "actions" (6 and 7 in this build): main+0x8312d0 stores each
// action's value for the frame, then main+0x834978 applies the action's dead zone and scale and clamps it to
// its maximum (1). A swipe faster than a full stick push gives values past 1 (dsr::input::AddLook); for those
// actions, while the finger is moving, the clamp is skipped, so a fast flick turns the camera faster.
std::atomic<uint64_t> g_look_actions{0};  // bit per action that has received an overdriven value

// main+0x8312d0 (device, action, value): the action writer, reimplemented; it also learns the camera actions.
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
    if (std::fabs(v) > 1.001f && action >= 0 && action < 64) {
        const uint64_t bit = 1ull << action;
        if (!(g_look_actions.fetch_or(bit) & bit)) LOGI("hooks", "touch camera: right stick feeds action %d", action);
    }
}

// main+0x821218 (list, i, &action, &pair, &a, &threshold, &dead_zone, &scale, [stack] &max): reads entry i
// (0x1c bytes) of an action configuration list, reimplemented. For a camera action while a swipe is going on,
// max reads as 0, which main+0x834978 takes as "no clamp".
void ActionConfig(uint8_t* list, int i, int32_t* action, int32_t* pair, int32_t* a, int32_t* threshold,
                  int32_t* dead_zone, int32_t* scale, int32_t* max) {
    const auto* e = reinterpret_cast<const int32_t*>(*reinterpret_cast<uint8_t**>(list + 8) + int64_t(i) * 0x1c);
    *action = e[0];
    *pair = e[1];
    *a = e[2];
    *threshold = e[3];
    *scale = e[4];
    *max = e[5];
    *dead_zone = e[6];
    const int id = e[0];
    if (id >= 0 && id < 64 && (g_look_actions.load(std::memory_order_relaxed) >> id & 1) && dsr::input::LookActive())
        *max = 0;
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
    {0x8312d0, reinterpret_cast<void*>(&ActionWrite), "input action write (touch camera)"},
    {0x821218, reinterpret_cast<void*>(&ActionConfig), "input action config (touch camera)"},
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
