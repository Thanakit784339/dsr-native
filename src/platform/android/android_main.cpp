// Android entry point (NativeActivity + native_app_glue).
//
// Game files: copy the extracted game (the folder containing exefs/, dvdbnd0.bdt, ...) to
//   /sdcard/Android/data/com.dsrnative.app/files/game/
// Save data goes to .../files/save, caches to .../files/cache.
//
// Logs: every run writes Download/DSRNative/dsr_<date>_<time>.txt on the phone's shared storage
// (visible to file managers and over USB), plus logcat (`adb logcat -s 'DSR/*'`).
#include <android/input.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <jni.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <thread>

#include "common/crash.h"
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

std::string Prop(const char* name) {
    char v[PROP_VALUE_MAX] = "";
    __system_property_get(name, v);
    return v;
}

// ---- log file ------------------------------------------------------------------------------------
// Android 13 restricts writing to shared folders by plain file path, so the primary route is MediaStore
// (the supported, permission-free way for an app to add a file to Download/). Fallbacks: plain path in
// Download/, Android/media/<package>/ (browsable with file managers), then the app's private folder.

// Clears and describes a pending Java exception.
bool JniFailed(JNIEnv* env, std::string* why, const char* step) {
    if (!env->ExceptionCheck()) return false;
    env->ExceptionClear();
    *why = std::string("exception in ") + step;
    return true;
}

// Creates Download/DSRNative/<name> through MediaStore and returns a writable fd, or -1.
int CreateDownloadViaMediaStore(android_app* app, const char* name, std::string* why) {
    JavaVM* vm = app->activity->vm;
    JNIEnv* env = nullptr;
    if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
        *why = "AttachCurrentThread failed";
        return -1;
    }
    int fd = -1;
    jobject activity = app->activity->clazz;
    do {
        jclass cv_cls = env->FindClass("android/content/ContentValues");
        if (JniFailed(env, why, "FindClass ContentValues") || !cv_cls) break;
        jmethodID cv_ctor = env->GetMethodID(cv_cls, "<init>", "()V");
        jmethodID cv_put = env->GetMethodID(cv_cls, "put", "(Ljava/lang/String;Ljava/lang/String;)V");
        if (JniFailed(env, why, "ContentValues methods")) break;
        jobject values = env->NewObject(cv_cls, cv_ctor);
        if (JniFailed(env, why, "new ContentValues")) break;
        const char* kv[][2] = {{"_display_name", name}, {"mime_type", "text/plain"}, {"relative_path", "Download/DSRNative"}};
        for (auto& p : kv) {
            jstring k = env->NewStringUTF(p[0]);
            jstring v = env->NewStringUTF(p[1]);
            env->CallVoidMethod(values, cv_put, k, v);
            env->DeleteLocalRef(k);
            env->DeleteLocalRef(v);
        }
        if (JniFailed(env, why, "ContentValues.put")) break;

        jclass dl_cls = env->FindClass("android/provider/MediaStore$Downloads");
        if (JniFailed(env, why, "FindClass MediaStore.Downloads") || !dl_cls) break;
        jfieldID uri_field = env->GetStaticFieldID(dl_cls, "EXTERNAL_CONTENT_URI", "Landroid/net/Uri;");
        if (JniFailed(env, why, "EXTERNAL_CONTENT_URI")) break;
        jobject collection = env->GetStaticObjectField(dl_cls, uri_field);

        jclass act_cls = env->GetObjectClass(activity);
        jmethodID get_cr = env->GetMethodID(act_cls, "getContentResolver", "()Landroid/content/ContentResolver;");
        jobject resolver = env->CallObjectMethod(activity, get_cr);
        if (JniFailed(env, why, "getContentResolver") || !resolver) break;
        jclass cr_cls = env->GetObjectClass(resolver);
        jmethodID insert = env->GetMethodID(cr_cls, "insert",
                                            "(Landroid/net/Uri;Landroid/content/ContentValues;)Landroid/net/Uri;");
        jobject item = env->CallObjectMethod(resolver, insert, collection, values);
        if (JniFailed(env, why, "ContentResolver.insert")) break;
        if (!item) {
            *why = "ContentResolver.insert returned null";
            break;
        }
        jmethodID open_fd = env->GetMethodID(cr_cls, "openFileDescriptor",
                                             "(Landroid/net/Uri;Ljava/lang/String;)Landroid/os/ParcelFileDescriptor;");
        jstring mode = env->NewStringUTF("w");
        jobject pfd = env->CallObjectMethod(resolver, open_fd, item, mode);
        env->DeleteLocalRef(mode);
        if (JniFailed(env, why, "openFileDescriptor") || !pfd) break;
        jclass pfd_cls = env->GetObjectClass(pfd);
        jmethodID detach = env->GetMethodID(pfd_cls, "detachFd", "()I");
        fd = env->CallIntMethod(pfd, detach);
        if (JniFailed(env, why, "detachFd")) fd = -1;
    } while (false);
    vm->DetachCurrentThread();
    return fd;
}

// Mirrors this process's logcat (Java exceptions, ART/JNI aborts, libc "Fatal signal" lines — things
// our own logger cannot see) into a second file. A small shell keeps logcat running until a few seconds
// after this process dies, so the end of a crash is captured too. Apps may read their own logs.
void StartLogcatMirror(int fd) {
    char script[512];
    snprintf(script, sizeof(script),
             "logcat -v threadtime --pid=%d & L=$!; "
             "while kill -0 %d 2>/dev/null; do sleep 1; done; sleep 5; kill $L",
             getpid(), getpid());
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fd, STDERR_FILENO);
    char* argv[] = {const_cast<char*>("sh"), const_cast<char*>("-c"), script, nullptr};
    pid_t pid;
    int rc = posix_spawn(&pid, "/system/bin/sh", &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd);
    if (rc == 0)
        LOGI(kTag, "logcat mirror started (pid %d)", pid);
    else
        LOGW(kTag, "logcat mirror failed to start: %s", strerror(rc));
}

void SetupLogFile(android_app* app) {
    time_t now = time(nullptr);
    tm t;
    localtime_r(&now, &t);
    char name[64];
    strftime(name, sizeof(name), "dsr_%Y%m%d_%H%M%S.txt", &t);

    std::vector<std::string> failures;
    std::string why;
    int fd = CreateDownloadViaMediaStore(app, name, &why);
    if (fd >= 0) {
        dsr::SetLogFileFd(fd);
        LOGI(kTag, "log file: Download/DSRNative/%s (via MediaStore)", name);
        char logcat_name[80];
        strftime(logcat_name, sizeof(logcat_name), "dsr_%Y%m%d_%H%M%S_logcat.txt", &t);
        std::string why2;
        int lfd = CreateDownloadViaMediaStore(app, logcat_name, &why2);
        if (lfd >= 0) {
            LOGI(kTag, "logcat file: Download/DSRNative/%s", logcat_name);
            StartLogcatMirror(lfd);
        } else {
            LOGW(kTag, "logcat file not created: %s", why2.c_str());
        }
        return;
    }
    failures.push_back("MediaStore Download/DSRNative: " + why);

    std::string app_dir = app->activity->externalDataPath ? app->activity->externalDataPath : "";
    const std::string package = "com.dsrnative.app";
    const std::string paths[] = {
        "/storage/emulated/0/Download/DSRNative",
        "/storage/emulated/0/Android/media/" + package,
        app_dir,
    };
    for (const std::string& dir : paths) {
        if (dir.empty()) continue;
        mkdir(dir.c_str(), 0775);
        std::string path = dir + "/" + name;
        if (dsr::OpenLogFile(path.c_str())) {
            LOGI(kTag, "log file: %s", path.c_str());
            for (const auto& f : failures) LOGW(kTag, "log location skipped: %s", f.c_str());
            return;
        }
        failures.push_back(path + ": " + strerror(errno));
    }
    for (const auto& f : failures) LOGW(kTag, "log location failed: %s", f.c_str());
    LOGW(kTag, "could not create a log file (logcat only)");
}

void LogDeviceInfo() {
    LOGI(kTag, "device: %s %s (%s), Android %s (API %s), build %s", Prop("ro.product.manufacturer").c_str(),
         Prop("ro.product.model").c_str(), Prop("ro.product.device").c_str(), Prop("ro.build.version.release").c_str(),
         Prop("ro.build.version.sdk").c_str(), Prop("ro.build.display.id").c_str());
    LOGI(kTag, "soc: %s %s, cpus: %ld, page size: %ld", Prop("ro.soc.manufacturer").c_str(), Prop("ro.soc.model").c_str(),
         sysconf(_SC_NPROCESSORS_ONLN), sysconf(_SC_PAGESIZE));
    long pages = sysconf(_SC_PHYS_PAGES);
    LOGI(kTag, "ram: %ld MiB", pages > 0 ? long((int64_t(pages) * sysconf(_SC_PAGESIZE)) >> 20) : -1L);
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
    LOGI(kTag, "game dir: %s", cfg.game_dir.c_str());
    if (!FileExists(cfg.game_dir + "/exefs/main")) {
        LOGE(kTag, "game files not found: expected %s/exefs/main", cfg.game_dir.c_str());
        LOGE(kTag, "copy the extracted game folder (the one containing exefs/ and dvdbnd0.bdt) there, e.g.:");
        LOGE(kTag, "  adb push DarkSoulsR/. %s/", cfg.game_dir.c_str());
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
        static int logged = 0;
        const int32_t code = AKeyEvent_getKeyCode(ev);
        uint64_t b = KeyToButton(code);
        if (logged < 40 && AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_DOWN) {
            logged++;
            LOGI(kTag, "input: key %d down (source 0x%x) -> %s", code, AInputEvent_getSource(ev),
                 b ? "mapped" : "not mapped");
        }
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

const char* CmdName(int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INPUT_CHANGED: return "INPUT_CHANGED";
        case APP_CMD_INIT_WINDOW: return "INIT_WINDOW";
        case APP_CMD_TERM_WINDOW: return "TERM_WINDOW";
        case APP_CMD_WINDOW_RESIZED: return "WINDOW_RESIZED";
        case APP_CMD_WINDOW_REDRAW_NEEDED: return "WINDOW_REDRAW_NEEDED";
        case APP_CMD_CONTENT_RECT_CHANGED: return "CONTENT_RECT_CHANGED";
        case APP_CMD_GAINED_FOCUS: return "GAINED_FOCUS";
        case APP_CMD_LOST_FOCUS: return "LOST_FOCUS";
        case APP_CMD_CONFIG_CHANGED: return "CONFIG_CHANGED";
        case APP_CMD_LOW_MEMORY: return "LOW_MEMORY";
        case APP_CMD_START: return "START";
        case APP_CMD_RESUME: return "RESUME";
        case APP_CMD_SAVE_STATE: return "SAVE_STATE";
        case APP_CMD_PAUSE: return "PAUSE";
        case APP_CMD_STOP: return "STOP";
        case APP_CMD_DESTROY: return "DESTROY";
        default: return "?";
    }
}

void OnCmd(android_app* app, int32_t cmd) {
    LOGI(kTag, "lifecycle: %s", CmdName(cmd));
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
    SetupLogFile(app);
    dsr::InstallCrashHandler();
    LOGI(kTag, "Dark Souls Remastered native runtime starting");
    LogDeviceInfo();
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
    LOGI(kTag, "activity destroyed (android_main returning)");
}
