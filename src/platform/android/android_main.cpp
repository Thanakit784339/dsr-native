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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <thread>

#include "common/crash.h"
#include "common/log.h"
#include "common/platform.h"
#include "common/profiler.h"
#include "hle/input.h"
#include "hle/nvn/backend.h"
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
int CreateDownloadViaMediaStore(android_app* app, const char* name, std::string* why,
                                const char* mime_type = "text/plain") {
    JavaVM* vm = app->activity->vm;
    JNIEnv* env = nullptr;
    // Only detach afterwards if this call attached the thread.
    bool attached_here = false;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) {
            *why = "AttachCurrentThread failed";
            return -1;
        }
        attached_here = true;
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
        const char* kv[][2] = {{"_display_name", name}, {"mime_type", mime_type}, {"relative_path", "Download/DSRNative"}};
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
    if (attached_here) vm->DetachCurrentThread();
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

android_app* g_app_for_files = nullptr;

}  // namespace
extern "C" JNIEXPORT void JNICALL Java_com_dsrnative_app_TextInput_onResult(JNIEnv* env, jclass, jstring text,
                                                                            jboolean ok);
namespace {

// Software keyboard: com.dsrnative.app.TextInput.show(activity, title, initial, maxLength). The class is
// loaded through the activity's class loader because FindClass on a native thread only sees system
// classes. The result comes back through TextInput.onResult (below).
bool RequestTextImpl(const char* title, const std::u16string& initial, int max_length) {
    android_app* app = g_app_for_files;
    if (!app) return false;
    JavaVM* vm = app->activity->vm;
    JNIEnv* env = nullptr;
    bool attached_here = false;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return false;
        attached_here = true;
    }
    bool ok = false;
    do {
        jobject activity = app->activity->clazz;
        jclass act_cls = env->GetObjectClass(activity);
        jmethodID get_loader = env->GetMethodID(act_cls, "getClassLoader", "()Ljava/lang/ClassLoader;");
        jobject loader = env->CallObjectMethod(activity, get_loader);
        if (env->ExceptionCheck() || !loader) break;
        jclass loader_cls = env->FindClass("java/lang/ClassLoader");
        jmethodID load = env->GetMethodID(loader_cls, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
        jstring name = env->NewStringUTF("com.dsrnative.app.TextInput");
        auto cls = static_cast<jclass>(env->CallObjectMethod(loader, load, name));
        env->DeleteLocalRef(name);
        if (env->ExceptionCheck() || !cls) break;
        // NativeActivity loads this library itself, so Java cannot find our JNI functions by name: register
        // the callback explicitly (once).
        static bool registered = false;
        if (!registered) {
            JNINativeMethod m{const_cast<char*>("onResult"), const_cast<char*>("(Ljava/lang/String;Z)V"),
                              reinterpret_cast<void*>(&Java_com_dsrnative_app_TextInput_onResult)};
            if (env->RegisterNatives(cls, &m, 1) != JNI_OK || env->ExceptionCheck()) break;
            registered = true;
        }
        jmethodID show = env->GetStaticMethodID(cls, "show", "(Landroid/app/Activity;Ljava/lang/String;Ljava/lang/String;I)V");
        if (env->ExceptionCheck() || !show) break;
        jstring jtitle = env->NewStringUTF(title);
        jstring jinit = env->NewString(reinterpret_cast<const jchar*>(initial.data()), jsize(initial.size()));
        env->CallStaticVoidMethod(cls, show, activity, jtitle, jinit, jint(max_length));
        env->DeleteLocalRef(jtitle);
        env->DeleteLocalRef(jinit);
        ok = !env->ExceptionCheck();
    } while (false);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    if (attached_here) vm->DetachCurrentThread();
    if (!ok) LOGW(kTag, "could not show the text input dialog");
    return ok;
}
int CreateUserFileImpl(const char* name, const char* mime_type) {
    if (!g_app_for_files) return -1;
    std::string why;
    int fd = CreateDownloadViaMediaStore(g_app_for_files, name, &why, mime_type);
    if (fd < 0) LOGW(kTag, "could not create %s: %s", name, why.c_str());
    return fd;
}

void SetupLogFile(android_app* app) {
    g_app_for_files = app;
    dsr::platform::SetCreateUserFile(&CreateUserFileImpl);
    dsr::platform::SetRequestText(&RequestTextImpl);
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
    LOGI(kTag, "DSR Native build %s", DSR_BUILD_ID);
    LOGI(kTag, "native code: %s", dsr::platform::NativeCodeOptimized() ? "optimised" : "NOT OPTIMISED (-O0): expect very low speed");
    LOGI(kTag, "device: %s %s (%s), Android %s (API %s), build %s", Prop("ro.product.manufacturer").c_str(),
         Prop("ro.product.model").c_str(), Prop("ro.product.device").c_str(), Prop("ro.build.version.release").c_str(),
         Prop("ro.build.version.sdk").c_str(), Prop("ro.build.display.id").c_str());
    LOGI(kTag, "soc: %s %s, cpus: %ld, page size: %ld", Prop("ro.soc.manufacturer").c_str(), Prop("ro.soc.model").c_str(),
         sysconf(_SC_NPROCESSORS_ONLN), sysconf(_SC_PAGESIZE));
    long pages = sysconf(_SC_PHYS_PAGES);
    LOGI(kTag, "ram: %ld MiB", pages > 0 ? long((int64_t(pages) * sysconf(_SC_PAGESIZE)) >> 20) : -1L);
}

#define PRESENT_MODE_HELP                                                                                    \
    "# present_mode: how finished frames reach the screen: mailbox (newest frame shown at each refresh,\n"   \
    "# lower latency), fifo (classic vsync queue), fifo_relaxed (vsync, tears when a frame is late),\n"      \
    "# immediate (no vsync, can tear). Falls back to fifo if the device does not offer the mode.\n"

bool g_face_by_position = false;  // `face_buttons` setting (see KeyToButton)

#define FACE_BUTTONS_HELP                                                                                    \
    "# face_buttons: label = each button does what its label says on Switch (A is A); position = by\n"      \
    "# position, as on a Switch pad (for Xbox-style controllers whose A sits where the Switch's B is)\n"

// settings.txt beside the game folder: "key = value" lines, '#' comments. Written with defaults if missing.
void LoadSettings(const std::string& path, dsr::RuntimeConfig* cfg) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        if (FILE* w = fopen(path.c_str(), "w")) {
            fprintf(w,
                    "# DSR Native settings (read at start-up)\n"
                    "# fps: frame rate cap, 30 (as on Switch) to 60; e.g. 45 for a steady rate in heavy areas\n"
                    "fps = %d\n"
                    "# async_pipelines: 1 = compile new GPU pipelines in the background (no hitch; a new object may\n"
                    "# appear a frame late), 0 = compile on the spot (can hitch the first time something is seen)\n"
                    "async_pipelines = %d\n"
                    "# auto_capture: 1 = save a few frame captures (dsr_frame_*.bin) during play, for diagnostics\n"
                    "# (pressing both sticks in, L3 + R3, always saves one)\n"
                    "auto_capture = %d\n"
                    "# profile: 1 = log where the CPU time goes every 10 s (diagnostics, costs ~1%%)\n"
                    "profile = %d\n"
                    "# gamma: extra display gamma on top of the in-game brightness (1.0 = as the game draws it;\n"
                    "# 1.2 - 1.6 lifts dark areas)\n"
                    "gamma = %.2f\n"
                    "# exposure: scales the scene's exposure before tone mapping (1.0 = as the game draws it; 1.45\n"
                    "# is what the game's own light adaptation computes in dark areas)\n"
                    "exposure = %.2f\n"
                    "# pin_render_thread: 1 = keep the render thread on the fastest CPU core (was slower on an\n"
                    "# 8 Gen 2: the GPU then waited; 0 = let Android choose)\n"
                    "pin_render_thread = 0\n"
                    "# record_thread: 1 = record GPU commands on a second CPU thread (faster), 0 = all on one thread\n"
                    "record_thread = 1\n"
                    "# fast_cores: 1 = keep the game's main, recording and submit threads on the performance cores\n"
                    "fast_cores = 1\n"
                    "# audio_smooth: 1 = release audio buffers to the game at a smoothed pace (experimental)\n"
                    "audio_smooth = 0\n"
                    PRESENT_MODE_HELP
                    "present_mode = mailbox\n"
                    FACE_BUTTONS_HELP
                    "face_buttons = label\n",
                    cfg->target_fps, cfg->async_pipelines ? 1 : 0, cfg->auto_capture ? 1 : 0, cfg->profile ? 1 : 0,
                    cfg->gamma, cfg->exposure);
            fclose(w);
            LOGI(kTag, "settings: wrote defaults to %s", path.c_str());
        }
    } else {
        bool have_present_mode = false, have_face_buttons = false;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            char key[64], value[64];
            if (line[0] == '#' || sscanf(line, " %63[a-z_0-9] = %63s", key, value) != 2) continue;
            const int v = atoi(value);
            if (!strcmp(key, "fps")) cfg->target_fps = std::clamp(v, 30, 60);  // e.g. 45: a steady cap below 60
            else if (!strcmp(key, "async_pipelines")) cfg->async_pipelines = v != 0;
            else if (!strcmp(key, "auto_capture")) cfg->auto_capture = v != 0;
            else if (!strcmp(key, "profile")) cfg->profile = v != 0;
            else if (!strcmp(key, "pin_render_thread")) cfg->pin_render_thread = v != 0;
            else if (!strcmp(key, "record_thread")) cfg->record_thread = v != 0;
            else if (!strcmp(key, "fast_cores")) cfg->fast_cores = v != 0;
            else if (!strcmp(key, "audio_smooth")) cfg->audio_smooth = v != 0;
            else if (!strcmp(key, "face_buttons")) {
                g_face_by_position = !strcmp(value, "position");
                have_face_buttons = true;
            } else if (!strcmp(key, "present_mode")) {
                cfg->present_mode = value;
                have_present_mode = true;
            }
            else if (!strcmp(key, "gamma")) {
                const float g = float(atof(value));
                cfg->gamma = g >= 0.5f && g <= 3.0f ? g : 1.0f;
            } else if (!strcmp(key, "exposure")) {
                const float e = float(atof(value));
                cfg->exposure = e >= 0.25f && e <= 4.0f ? e : 1.0f;
            }
        }
        fclose(f);
        // Settings added since the file was written are appended with their defaults, so they can be found.
        if (!have_present_mode)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" PRESENT_MODE_HELP "present_mode = mailbox\n", w);
                fclose(w);
                LOGI(kTag, "settings: added present_mode to %s", path.c_str());
            }
        if (!have_face_buttons)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" FACE_BUTTONS_HELP "face_buttons = label\n", w);
                fclose(w);
                LOGI(kTag, "settings: added face_buttons to %s", path.c_str());
            }
    }
    LOGI(kTag, "settings (%s): fps %d, async pipelines %s, auto capture %s, profile %s, gamma %.2f, exposure %.2f, "
         "present mode %s, face buttons by %s",
         path.c_str(), cfg->target_fps, cfg->async_pipelines ? "on" : "off", cfg->auto_capture ? "on" : "off",
         cfg->profile ? "on" : "off", cfg->gamma, cfg->exposure, cfg->present_mode.c_str(),
         g_face_by_position ? "position" : "label");
    if (cfg->profile) dsr::profiler::Enable();
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
    LoadSettings(base + "/settings.txt", &cfg);
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
        // By label (default): the button Android reports as A is the Switch's A, and so on.
        // `face_buttons = position` maps by position instead (Xbox-style pads, whose A sits where the Switch's
        // B is; this was the only behaviour before).
        case AKEYCODE_BUTTON_A: return g_face_by_position ? kB : kA;
        case AKEYCODE_BUTTON_B: return g_face_by_position ? kA : kB;
        case AKEYCODE_BUTTON_X: return g_face_by_position ? kY : kX;
        case AKEYCODE_BUTTON_Y: return g_face_by_position ? kX : kY;
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
        const uint64_t before = g_pad.buttons;
        if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_DOWN)
            g_pad.buttons |= b;
        else if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_UP)
            g_pad.buttons &= ~b;
        // Diagnostics: pressing both sticks in (L3 + R3) saves a frame capture (dsr_frame_*.bin) of the next frame.
        {
            using namespace dsr::input;
            const uint64_t both = kStickL | kStickR;
            if ((g_pad.buttons & both) == both && (before & both) != both) {
                LOGI(kTag, "L3 + R3: frame capture requested");
                dsr::hle::nvn::GetBackend().RequestCapture(0);
            }
        }
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

// Called by com.dsrnative.app.TextInput when the player confirms or cancels the dialog.
extern "C" JNIEXPORT void JNICALL Java_com_dsrnative_app_TextInput_onResult(JNIEnv* env, jclass, jstring text,
                                                                            jboolean ok) {
    std::u16string s;
    if (text) {
        const jchar* chars = env->GetStringChars(text, nullptr);
        s.assign(reinterpret_cast<const char16_t*>(chars), size_t(env->GetStringLength(text)));
        env->ReleaseStringChars(text, chars);
    }
    LOGI("swkbd", "text input %s (%zu characters)", ok ? "confirmed" : "cancelled", s.size());
    dsr::platform::DeliverText(ok, s);
}
