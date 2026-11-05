// Hooks on specific functions inside the game's main module (this build only, matched by build id).
// They replace engine functions whose output is useful to us but goes nowhere in a retail build.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "common/log.h"
#include "runtime/runtime.h"

namespace dsr {
namespace {
float* g_game_step = nullptr;  // the game's per-frame time step (scratch page), when patched

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

void SetGameFrameTime(double seconds) {
    if (!g_game_step) return;
    *g_game_step = float(std::clamp(seconds, 1.0 / 60.0, 1.0 / 30.0));
}
float GameFrameTime() { return g_game_step ? *g_game_step : 1.0f / 30.0f; }

void InstallGameHooks() {
    Runtime& rt = GetRuntime();
    if (rt.main_nso.build_id_hex.rfind(kBuildId, 0) != 0) {
        LOGW("hooks", "unknown game build %.32s: engine hooks disabled", rt.main_nso.build_id_hex.c_str());
        return;
    }
    // Frame rate. Two patches (the same places a well-known Switch 60 fps mod changes):
    // 1. The render thread's frame limiter (main+0x70bd90, "sleep until 1000/fps ms have passed") is called
    //    with a hard-coded 30 (main+0x70bcb0, orr w1, wzr, #30): it gets 60.
    // 2. The game's per-frame update (main+0x66c99c) steps every system by a constant 1/30 s, loaded from
    //    read-only data (main+0x66c9c8 adrp x8, page; +0x66c9cc ldr s8, [x8, #0x970]). It now loads a variable
    //    in the scratch page instead, which SetGameFrameTime keeps at the measured frame time (1/60..1/30 s):
    //    normal speed at 60 fps, and still at 35 fps in heavy areas.
    if (rt.config.target_fps == 60) {
        std::string err;
        bool ok = rt.main.PatchInstruction(0x70bcb0, 0x321F0FE1, 0x52800781, &err);  // -> movz w1, #60
        if (!ok) LOGW("hooks", "could not set the frame limiter to 60 fps: %s", err.c_str());
        float* step = reinterpret_cast<float*>(rt.main.ScratchPage());
        *step = 1.0f / 30.0f;
        if (ok) {
            const uint32_t adrp = rt.main.EncodeAdrp(0x66c9c8, 8, reinterpret_cast<uintptr_t>(step));
            ok = rt.main.PatchInstruction(0x66c9c8, 0xB00080A8, adrp, &err) &&
                 rt.main.PatchInstruction(0x66c9cc, 0xBD497108, 0xBD400108, &err);  // ldr s8, [x8]
            if (!ok) LOGW("hooks", "could not patch the game's time step: %s (staying at 30 fps)", err.c_str());
        }
        if (ok) {
            g_game_step = step;
            LOGI("hooks", "60 fps: frame limiter at 60, game time step follows the frame time");
        } else {
            // Without a variable time step the game would run fast: undo the limiter patch.
            rt.main.PatchInstruction(0x70bcb0, 0x52800781, 0x321F0FE1, &err);
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
