// Android entry point (NativeActivity + native_app_glue).
//
// Game files: copy the extracted game (the folder containing exefs/, dvdbnd0.bdt, ...) to
//   /sdcard/Android/data/com.dsrnative.app/files/game/
// Save data goes to .../files/save, caches to .../files/cache. Logs: `adb logcat -s 'DSR/*'`.
#include <android/input.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <sys/stat.h>

#include <atomic>
#include <cmath>
#include <string>
#include <thread>

#include "common/log.h"
#include "hle/input.h"
#include "hle/video.h"
#include "runtime/runtime.h"

namespace {
constexpr const char* kTag = "android";

std::atomic<bool> g_game_started{false};
dsr::input::PadState g_pad;

bool FileExists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

void StartGame(android_app* app) {
    if (g_game_started.exchange(true)) return;
    std::string base = app->activity->externalDataPath ? app->activity->externalDataPath : "/sdcard";
    dsr::RuntimeConfig cfg;
    cfg.game_dir = base + "/game";
    cfg.save_dir = base + "/save";
    cfg.cache_dir = base + "/cache";
    mkdir(cfg.save_dir.c_str(), 0775);
    mkdir(cfg.cache_dir.c_str(), 0775);
    if (!FileExists(cfg.game_dir + "/exefs/main")) {
        LOGE(kTag, "game files not found: expected %s/exefs/main", cfg.game_dir.c_str());
        return;
    }
    std::thread([cfg] {
        std::string err;
        if (!dsr::LoadGame(cfg, &err)) {
            LOGE(kTag, "LoadGame failed: %s", err.c_str());
            return;
        }
        dsr::LogImportReport(false);
        int rc = dsr::RunGame();
        LOGI(kTag, "game exited with %d", rc);
    }).detach();
}

int32_t AxisToStick(float v) {
    if (std::fabs(v) < 0.08f) return 0;  // dead zone
    return int32_t(std::lround(std::fmax(-1.0f, std::fmin(1.0f, v)) * 32767.0f));
}

uint64_t KeyToButton(int32_t key) {
    using namespace dsr::input;
    switch (key) {
        // Android gamepads use Xbox-style positions; Switch A/B and X/Y are swapped by position.
        case AKEYCODE_BUTTON_A: return kB;
        case AKEYCODE_BUTTON_B: return kA;
        case AKEYCODE_BUTTON_X: return kY;
        case AKEYCODE_BUTTON_Y: return kX;
        case AKEYCODE_BUTTON_L1: return kL;
        case AKEYCODE_BUTTON_R1: return kR;
        case AKEYCODE_BUTTON_L2: return kZL;
        case AKEYCODE_BUTTON_R2: return kZR;
        case AKEYCODE_BUTTON_THUMBL: return kStickL;
        case AKEYCODE_BUTTON_THUMBR: return kStickR;
        case AKEYCODE_BUTTON_START: return kPlus;
        case AKEYCODE_BUTTON_SELECT: return kMinus;
        case AKEYCODE_DPAD_UP: return kUp;
        case AKEYCODE_DPAD_DOWN: return kDown;
        case AKEYCODE_DPAD_LEFT: return kLeft;
        case AKEYCODE_DPAD_RIGHT: return kRight;
        default: return 0;
    }
}

int32_t OnInput(android_app*, AInputEvent* ev) {
    using namespace dsr::input;
    const int32_t type = AInputEvent_getType(ev);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        uint64_t b = KeyToButton(AKeyEvent_getKeyCode(ev));
        if (!b) return 0;
        if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_DOWN)
            g_pad.buttons |= b;
        else if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_UP)
            g_pad.buttons &= ~b;
        SetPad(g_pad);
        return 1;
    }
    if (type == AINPUT_EVENT_TYPE_MOTION && (AInputEvent_getSource(ev) & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        g_pad.lx = AxisToStick(AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_X, 0));
        g_pad.ly = -AxisToStick(AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_Y, 0));
        g_pad.rx = AxisToStick(AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_Z, 0));
        g_pad.ry = -AxisToStick(AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_RZ, 0));
        // Analog triggers and hat-switch d-pads.
        auto set = [](uint64_t bit, bool on) { g_pad.buttons = on ? (g_pad.buttons | bit) : (g_pad.buttons & ~bit); };
        set(kZL, AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_LTRIGGER, 0) > 0.5f ||
                     AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_BRAKE, 0) > 0.5f);
        set(kZR, AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_RTRIGGER, 0) > 0.5f ||
                     AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_GAS, 0) > 0.5f);
        float hx = AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_HAT_X, 0);
        float hy = AMotionEvent_getAxisValue(ev, AMOTION_EVENT_AXIS_HAT_Y, 0);
        set(kLeft, hx < -0.5f);
        set(kRight, hx > 0.5f);
        set(kUp, hy < -0.5f);
        set(kDown, hy > 0.5f);
        SetPad(g_pad);
        return 1;
    }
    return 0;
}

void OnCmd(android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            LOGI(kTag, "window ready %p", app->window);
            dsr::video::SetNativeWindow(app->window);
            StartGame(app);
            break;
        case APP_CMD_TERM_WINDOW:
            LOGI(kTag, "window lost");
            dsr::video::SetNativeWindow(nullptr);
            break;
        default:
            break;
    }
}
}  // namespace

void android_main(android_app* app) {
    dsr::SetLogLevel(dsr::LogLevel::Debug);
    LOGI(kTag, "Dark Souls Remastered native runtime starting");
    app->onAppCmd = OnCmd;
    app->onInputEvent = OnInput;
    while (!app->destroyRequested) {
        int events;
        android_poll_source* source;
        while (ALooper_pollOnce(-1, nullptr, &events, reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
            if (app->destroyRequested) break;
        }
    }
    LOGI(kTag, "activity destroyed");
}
