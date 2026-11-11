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
#include "hle/os/heap.h"
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

// ---- aspect ratio: camera aspect (experimental) --------------------------------------------------------------
// With a wider or narrower render resolution the 3D view still came out stretched: the camera that renders the
// scene keeps a 16:9 aspect ratio. Cameras keep {fov, aspect, near, far} at +0x50..+0x5c (main+0xfa7f4 reads
// fov and aspect for the view frustum). This scans the guest heap and the game's data for such blocks with a
// 16:9 aspect, logs them (with the owner's vtable, to find the code that sets it) and writes the wanted aspect
// into them, again every second (rescanning every 2 s until cameras are found, then every 10 s).
float g_aspect = 0;
void CameraAspectThread() {
    pthread_setname_np(pthread_self(), "dsr-aspect");
    const loader::Module& m = GetRuntime().main;
    std::vector<float*> found;
    int scans = 0;
    auto plausible = [](const float* a) {  // a = &aspect: fov before it, near and far after
        const float fov = a[-1], nearz = a[1], farz = a[2];
        return fov > 0.2f && fov < 2.5f && nearz > 0.001f && nearz < 50.0f && farz > nearz && farz < 1e6f;
    };
    for (;;) {
        const auto scan_start = std::chrono::steady_clock::now();
        // Scan: 16:9 exactly (0x3FE38E39) at a 4-byte boundary.
        uintptr_t hb = 0;
        size_t hs = 0;
        hle::os::GuestHeapRange(&hb, &hs);
        std::vector<std::pair<uintptr_t, size_t>> ranges;
        if (hb && hs) ranges.push_back({hb, hs});
        if (m.size() > 0x1ae2000)
            ranges.push_back({m.base() + 0x1ae2000, std::min<size_t>(m.size() - 0x1ae2000, 0x7d65f8 + 0x3339a08)});  // .data + .bss
        int new_found = 0;
        for (const auto& [base, size] : ranges) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(base + 4);
            const uint32_t* end = reinterpret_cast<const uint32_t*>(base + size - 12);
            for (; p < end; p++) {
                if (*p != 0x3FE38E39u) continue;
                float* a = reinterpret_cast<float*>(const_cast<uint32_t*>(p));
                if (!plausible(a)) continue;
                if (std::find(found.begin(), found.end(), a) != found.end()) continue;
                found.push_back(a);
                new_found++;
                const uintptr_t obj = reinterpret_cast<uintptr_t>(a) - 0x54;
                uintptr_t vt = 0;
                if ((obj & 7) == 0) memcpy(&vt, reinterpret_cast<const void*>(obj), 8);
                if (found.size() <= 40)
                    LOGI("hooks", "camera aspect: 16:9 at %p (fov %.3f near %.3f far %.1f), object vtable %s0x%lx", a,
                         a[-1], a[1], a[2], m.Contains(vt) ? "main+" : "", (unsigned long)(m.Contains(vt) ? vt - m.base() : vt));
            }
        }
        scans++;
        const double scan_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scan_start).count();
        if (new_found || scans <= 3 || scans % 60 == 0)
            LOGI("hooks", "camera aspect: scan %d (%.0f ms): %d new 16:9 cameras, %zu in all; setting them to %.3f", scans,
                 scan_ms, new_found, found.size(), g_aspect);
        const int rewrites = found.empty() ? 2 : 10;  // 2 s, then 10 s between scans
        for (int k = 0; k < rewrites; k++) {
            for (float* a : found)
                if (*a > 1.7777f && *a < 1.7779f && plausible(a)) *a = g_aspect;
            std::this_thread::sleep_for(std::chrono::seconds(1));
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
bool FrameRateAdjustable() { return g_game_step != nullptr; }
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
    // Installed at every frame rate (30 included), so the cap can be changed while the game runs (in-game
    // settings menu): FrameLimiter reads the configured rate every frame.
    {
        std::string err;
        float* step = reinterpret_cast<float*>(rt.main.ScratchPage());
        *step = 1.0f / 30.0f;
        const uint32_t adrp = rt.main.EncodeAdrp(0x66c9c8, 8, reinterpret_cast<uintptr_t>(step));
        bool ok = rt.main.PatchInstruction(0x66c9c8, 0xB00080A8, adrp, &err) &&
                  rt.main.PatchInstruction(0x66c9cc, 0xBD497108, 0xBD400108, &err);  // ldr s8, [x8]
        if (ok) ok = rt.main.HookFunction(0x70bd90, reinterpret_cast<void*>(&FrameLimiter), &err);
        if (ok) {
            g_game_step = step;
            LOGI("hooks", "frame rate: own frame limiter at %d fps (changeable in game), game time step follows the frame time",
                 std::clamp(rt.config.target_fps, 30, 60));
        } else {
            LOGW("hooks", "60 fps patches failed (%s): staying at 30 fps", err.c_str());
            rt.main.PatchInstruction(0x66c9cc, 0xBD400108, 0xBD497108, &err);  // undo, if the first ones applied
            rt.main.PatchInstruction(0x66c9c8, rt.main.EncodeAdrp(0x66c9c8, 8, reinterpret_cast<uintptr_t>(step)),
                                     0xB00080A8, &err);
            rt.config.target_fps = 30;
        }
    }
    // Aspect ratio. Two places set the game's resolution, both with built-in defaults:
    // - the render resolution, read from the game's settings with defaults 1280 / 720 (main+0x6a16b0:
    //   "Normal.Resolution.Width" main+0x6a170c mov w2, #1280; "...Height" main+0x6a1758 mov w2, #720), and
    //   again by the width getter main+0x6a18cc (main+0x6a18f8 mov w2, #1280) whose callers (main+0x101d80,
    //   +0x66e99c, +0x66e9cc) size the post-processing buffers and their copies of the scene. It
    //   also becomes the screen size floats (main+0x1af4a7c / +0x1af4a84) the cameras take their aspect ratio
    //   from (e.g. main+0xbca88), so the 3D view widens (or narrows) instead of stretching;
    // - the handheld display mode (main+0x716158: main+0x716170 mov w1, #1280; +0x716178 mov w2, #720).
    // Frame buffers are allocated at 1920x1080 (main+0x70a20c), the limit. The height always stays 720: the
    // game's half- and quarter-size buffers (640x360, 320x180) keep their 720-line sizes whatever the render
    // resolution, and a 4:3 test at 1280x960 died silently on loading a save. Only the width changes: wider
    // than 16:9 shows more at the sides, narrower (4:3: 960x720) less.
    if (rt.config.aspect_ratio != "16:9" && !rt.config.docked) {
        float ar = 0;
        if (rt.config.aspect_ratio == "auto") {
            ar = rt.config.screen_aspect;
        } else {
            float w = 0, h = 0;
            if (sscanf(rt.config.aspect_ratio.c_str(), "%f:%f", &w, &h) == 2 && h > 0) ar = w / h;
        }
        if (ar > 0) {
            const int height = 720;
            const int width = std::clamp(int(std::lround(720.0f * ar / 8.0f)) * 8, 640, 1920);
            std::string err;
            auto mov = [](int rd, int v) { return 0x52800000u | (uint32_t(v) << 5) | uint32_t(rd); };
            bool ok = true;
            if (width != 1280)
                ok = ok && rt.main.PatchInstruction(0x6a170c, 0x5280A002, mov(2, width), &err) &&
                     rt.main.PatchInstruction(0x6a18f8, 0x5280A002, mov(2, width), &err) &&
                     rt.main.PatchInstruction(0x716170, 0x5280A001, mov(1, width), &err);
            if (ok) {
                LOGI("hooks", "aspect ratio %s (%.3f): the game renders at %dx%d (was 1280x720)",
                     rt.config.aspect_ratio.c_str(), ar, width, height);
                g_aspect = float(width) / float(height);
                rt.config.render_width = width;
                std::thread(CameraAspectThread).detach();
            }
            else
                LOGW("hooks", "aspect ratio patch failed: %s", err.c_str());
        } else {
            LOGW("hooks", "aspect_ratio %s: not understood (or screen size unknown), staying at 16:9",
                 rt.config.aspect_ratio.c_str());
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
