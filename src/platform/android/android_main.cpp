// Android entry point (NativeActivity + native_app_glue).
//
// Files: everything lives in Download/DSRNative on the phone's shared storage (needs "All files access", asked
// for on first launch; refused -> the app's private folder Android/data/com.dsrnative.app/files instead):
//   game/          the extracted game (the folder containing exefs/, dvdbnd0.bdt, ...)
//   save/ cache/   save data, GPU pipeline cache
//   settings.txt
//   drivers/       custom Vulkan drivers (.zip as for AdrenoTools, or a bare .so), chosen with gpu_driver
//   logs/          dsr_<date>_<time>.txt run logs, frame captures, shader dumps (also logcat: adb logcat -s 'DSR/*')
#include <android/input.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <spawn.h>
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <jni.h>
#include <zlib.h>

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

// mkdir -p.
void MakeDirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') mkdir(path.substr(0, i).c_str(), 0775);
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

// Creates Download/DSRNative/logs/<name> through MediaStore and returns a writable fd, or -1.
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
        const char* kv[][2] = {{"_display_name", name}, {"mime_type", mime_type}, {"relative_path", "Download/DSRNative/logs"}};
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
        LOGI(kTag, "log file: Download/DSRNative/logs/%s (via MediaStore)", name);
        char logcat_name[80];
        strftime(logcat_name, sizeof(logcat_name), "dsr_%Y%m%d_%H%M%S_logcat.txt", &t);
        std::string why2;
        int lfd = CreateDownloadViaMediaStore(app, logcat_name, &why2);
        if (lfd >= 0) {
            LOGI(kTag, "logcat file: Download/DSRNative/logs/%s", logcat_name);
            StartLogcatMirror(lfd);
        } else {
            LOGW(kTag, "logcat file not created: %s", why2.c_str());
        }
        return;
    }
    failures.push_back("MediaStore Download/DSRNative/logs: " + why);

    std::string app_dir = app->activity->externalDataPath ? app->activity->externalDataPath : "";
    const std::string package = "com.dsrnative.app";
    const std::string paths[] = {
        "/storage/emulated/0/Download/DSRNative/logs",
        "/storage/emulated/0/Android/media/" + package,
        app_dir,
    };
    for (const std::string& dir : paths) {
        if (dir.empty()) continue;
        MakeDirs(dir);
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
// Touch controls settings (see TOUCH_HELP).
struct TouchSettings {
    bool enabled = true;
    float opacity = 0.7f;
    float camera_speed = 1.0f;
    bool camera_invert = false;
} g_touch;
std::atomic<bool> g_touch_visible{false};  // the on-screen controls show (set by TouchControls.java)

#define TOUCH_HELP                                                                                            \
    "# Touch controls: a stick at the bottom left, the Switch buttons around the edges (LOCK = lock-on, the\n" \
    "# right stick press); touch the right side anywhere else for a camera stick that follows your finger.\n"  \
    "# They hide when a controller is connected or used; double-tap the screen to bring them back.\n"         \
    "# touch_controls: 1 = on, 0 = off. touch_opacity: 0.1 to 1. touch_camera_speed: how far the camera\n"   \
    "# stick moves for a full turn (1 = to its edge, 2 = half that). touch_camera_invert: 1 = up looks down.\n"
#define TOUCH_DEFAULTS "touch_controls = 1\ntouch_opacity = 0.7\ntouch_camera_speed = 1.0\ntouch_camera_invert = 0\n"

#define GPU_DRIVER_HELP                                                                                      \
    "# gpu_driver: system = the phone's own Vulkan driver; or the name of a file in the drivers folder next\n" \
    "# to this file: a driver .zip as made for AdrenoTools (e.g. Mesa Turnip) or a bare .so\n"                 \
    "# (e.g. gpu_driver = turnip.zip). If it cannot be loaded, the system driver is used.\n"
#define BC_ENABLER_HELP                                                                                      \
    "# bc_enabler: 1 = on Adreno 6xx system drivers, patch in the compressed (BC) texture formats the GPU\n"   \
    "# supports but the driver hides (faster, and some textures look right); 0 = decode them on the CPU\n"

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
                    "# exposure: scales the scene's exposure before tone mapping (default 1.37; 1.0 = exactly as the\n"
                    "# game draws it; higher is brighter)\n"
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
                    "face_buttons = label\n"
                    "# lrz_workaround: 1 = refresh depth buffers written by shaders before reuse (experiment, off)\n"
                    "lrz_workaround = 0\n"
                    "# shader_depth_test: 1 = fix for see-through squares in smoke, fog and glow (0 = off, for testing)\n"
                    "shader_depth_test = 1\n"
                    GPU_DRIVER_HELP
                    "gpu_driver = system\n"
                    BC_ENABLER_HELP
                    "bc_enabler = 1\n"
                    TOUCH_HELP TOUCH_DEFAULTS,
                    cfg->target_fps, cfg->async_pipelines ? 1 : 0, cfg->auto_capture ? 1 : 0, cfg->profile ? 1 : 0,
                    cfg->gamma, cfg->exposure);
            fclose(w);
            LOGI(kTag, "settings: wrote defaults to %s", path.c_str());
        } else {
            LOGW(kTag, "settings: cannot create %s (%s): using defaults", path.c_str(), strerror(errno));
        }
    } else {
        bool have_present_mode = false, have_face_buttons = false, have_gpu_driver = false, have_bc_enabler = false;
        bool have_touch = false;
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
            else if (!strcmp(key, "gpu_compression")) cfg->gpu_compression = v != 0;
            else if (!strcmp(key, "audio_smooth")) cfg->audio_smooth = v != 0;
            else if (!strcmp(key, "lrz_workaround")) cfg->lrz_workaround = v != 0;
            else if (!strcmp(key, "shader_depth_test")) cfg->shader_depth_test = v != 0;
            else if (!strcmp(key, "hole_test")) cfg->hole_test = v;
            else if (!strcmp(key, "touch_controls")) {
                g_touch.enabled = v != 0;
                have_touch = true;
            } else if (!strcmp(key, "touch_opacity")) {
                const float o = float(atof(value));
                g_touch.opacity = o >= 0.1f && o <= 1.0f ? o : 0.7f;
            } else if (!strcmp(key, "touch_camera_speed")) {
                const float sp = float(atof(value));
                g_touch.camera_speed = sp >= 0.1f && sp <= 10.0f ? sp : 1.0f;
            } else if (!strcmp(key, "touch_camera_invert")) {
                g_touch.camera_invert = v != 0;
            } else if (!strcmp(key, "bc_enabler")) {
                cfg->bc_enabler = v != 0;
                have_bc_enabler = true;
            } else if (!strcmp(key, "gpu_driver")) {
                // The whole rest of the line (file names may contain spaces).
                std::string val = strchr(line, '=') + 1;
                while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == ' ' || val.back() == '\t'))
                    val.pop_back();
                size_t b = val.find_first_not_of(" \t");
                cfg->gpu_driver = b == std::string::npos ? "system" : val.substr(b);
                have_gpu_driver = true;
            }
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
                cfg->exposure = e >= 0.25f && e <= 4.0f ? e : dsr::RuntimeConfig{}.exposure;
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
        if (!have_touch)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" TOUCH_HELP TOUCH_DEFAULTS, w);
                fclose(w);
                LOGI(kTag, "settings: added the touch controls settings to %s", path.c_str());
            }
        if (!have_gpu_driver)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" GPU_DRIVER_HELP "gpu_driver = system\n", w);
                fclose(w);
                LOGI(kTag, "settings: added gpu_driver to %s", path.c_str());
            }
        if (!have_bc_enabler)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" BC_ENABLER_HELP "bc_enabler = 1\n", w);
                fclose(w);
                LOGI(kTag, "settings: added bc_enabler to %s", path.c_str());
            }
    }
    LOGI(kTag, "settings (%s): fps %d, async pipelines %s, auto capture %s, profile %s, gamma %.2f, exposure %.2f, "
         "present mode %s, face buttons by %s",
         path.c_str(), cfg->target_fps, cfg->async_pipelines ? "on" : "off", cfg->auto_capture ? "on" : "off",
         cfg->profile ? "on" : "off", cfg->gamma, cfg->exposure, cfg->present_mode.c_str(),
         g_face_by_position ? "position" : "label");
    LOGI(kTag, "settings: gpu_driver %s, bc_enabler %d", cfg->gpu_driver.c_str(), int(cfg->bc_enabler));
    LOGI(kTag, "settings: touch controls %s, opacity %.2f, camera speed %.2f%s", g_touch.enabled ? "on" : "off",
         g_touch.opacity, g_touch.camera_speed, g_touch.camera_invert ? ", camera inverted" : "");
    if (!cfg->shader_depth_test) LOGI(kTag, "settings: shader_depth_test off");
    if (cfg->hole_test || cfg->lrz_workaround)
        LOGI(kTag, "settings: hole_test %d, lrz_workaround %d", cfg->hole_test, int(cfg->lrz_workaround));
    if (cfg->profile) dsr::profiler::Enable();
}

// The app's folder for everything the player handles: game/, save/, cache/, settings.txt (logs, frame captures
// and shader dumps go to logs/ inside it, through MediaStore).
constexpr const char* kSharedDir = "/storage/emulated/0/Download/DSRNative";

// GameActivity.storageAccess(): 0 = still waiting for the player's answer, 1 = granted, 2 = refused.
int StorageAccess(android_app* app) {
    JavaVM* vm = app->activity->vm;
    JNIEnv* env = nullptr;
    bool attached_here = false;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return 2;
        attached_here = true;
    }
    int r = 2;
    jclass cls = env->GetObjectClass(app->activity->clazz);
    jmethodID m = cls ? env->GetStaticMethodID(cls, "storageAccess", "()I") : nullptr;
    if (m && !env->ExceptionCheck()) r = env->CallStaticIntMethod(cls, m);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        r = 2;
    }
    if (attached_here) vm->DetachCurrentThread();
    return r;
}

bool CopyFile(const std::string& from, const std::string& to) {
    FILE* in = fopen(from.c_str(), "rb");
    if (!in) return false;
    FILE* out = fopen(to.c_str(), "wb");
    if (!out) {
        fclose(in);
        return false;
    }
    std::vector<char> buf(1 << 16);
    size_t n;
    bool ok = true;
    while ((n = fread(buf.data(), 1, buf.size(), in)) > 0)
        if (fwrite(buf.data(), 1, n, out) != n) ok = false;
    fclose(in);
    if (fclose(out) != 0) ok = false;
    return ok;
}

// Copies a folder tree; files already present at the destination are kept. Returns the number of files copied.
int CopyTree(const std::string& from, const std::string& to) {
    DIR* d = opendir(from.c_str());
    if (!d) return 0;
    mkdir(to.c_str(), 0775);
    int copied = 0;
    while (dirent* e = readdir(d)) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        const std::string a = from + "/" + e->d_name, b = to + "/" + e->d_name;
        struct stat st;
        if (stat(a.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode))
            copied += CopyTree(a, b);
        else if (!FileExists(b) && CopyFile(a, b))
            copied++;
    }
    closedir(d);
    return copied;
}

// Brings saves, settings and caches over from the folder used before (Android/data/<package>/files). The game
// files are not copied (several GB): if they are only there, they are used from there until moved.
void MigrateFromAppFolder(const std::string& old_dir, const std::string& base) {
    if (old_dir.empty() || old_dir == base || !FileExists(old_dir)) return;
    if (!FileExists(base + "/settings.txt") && FileExists(old_dir + "/settings.txt") &&
        CopyFile(old_dir + "/settings.txt", base + "/settings.txt"))
        LOGI(kTag, "moved over: settings.txt from %s", old_dir.c_str());
    if (int n = CopyTree(old_dir + "/save", base + "/save"))
        LOGI(kTag, "moved over: %d save file(s) from %s/save (the originals are kept)", n, old_dir.c_str());
    if (int n = CopyTree(old_dir + "/cache", base + "/cache"))
        LOGI(kTag, "moved over: %d cache file(s) from %s/cache", n, old_dir.c_str());
}

bool CanWriteIn(const std::string& dir) {
    const std::string probe = dir + "/.write_test";
    FILE* f = fopen(probe.c_str(), "w");
    if (!f) return false;
    fclose(f);
    unlink(probe.c_str());
    return true;
}

// The app's native library folder (getApplicationInfo().nativeLibraryDir): libadrenotools loads its hook
// libraries from there.
std::string NativeLibraryDir(android_app* app) {
    JavaVM* vm = app->activity->vm;
    JNIEnv* env = nullptr;
    bool attached_here = false;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return "";
        attached_here = true;
    }
    std::string dir;
    jobject act = app->activity->clazz;
    jclass cls = env->GetObjectClass(act);
    jmethodID get_info = env->GetMethodID(cls, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
    jobject info = get_info ? env->CallObjectMethod(act, get_info) : nullptr;
    if (info && !env->ExceptionCheck()) {
        jclass icls = env->GetObjectClass(info);
        jfieldID fid = env->GetFieldID(icls, "nativeLibraryDir", "Ljava/lang/String;");
        auto js = fid ? static_cast<jstring>(env->GetObjectField(info, fid)) : nullptr;
        if (js) {
            const char* c = env->GetStringUTFChars(js, nullptr);
            if (c) dir = c;
            env->ReleaseStringUTFChars(js, c);
        }
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (attached_here) vm->DetachCurrentThread();
    return dir;
}

// Minimal zip reader (stored and deflated entries): extracts the files of `zip` whose names end in one of
// `suffixes` into `dir` (flattened: the base name only). Returns the names extracted.
std::vector<std::string> Unzip(const std::string& zip, const std::string& dir, const std::vector<std::string>& suffixes,
                               std::string* err) {
    std::vector<std::string> out;
    FILE* f = fopen(zip.c_str(), "rb");
    if (!f) {
        *err = strerror(errno);
        return out;
    }
    std::vector<uint8_t> data;
    {
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (n > 0 && n < (512l << 20)) {
            data.resize(size_t(n));
            if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        }
        fclose(f);
    }
    auto u16 = [&](size_t o) { return uint32_t(data[o] | data[o + 1] << 8); };
    auto u32 = [&](size_t o) { return uint32_t(data[o] | data[o + 1] << 8 | data[o + 2] << 16 | uint32_t(data[o + 3]) << 24); };
    size_t eocd = std::string::npos;
    for (size_t i = data.size() >= 22 ? data.size() - 22 : 0; data.size() >= 22; i--) {
        if (u32(i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (i == 0 || data.size() - i > 66000) break;
    }
    if (eocd == std::string::npos) {
        *err = "not a zip file";
        return out;
    }
    const uint32_t count = u16(eocd + 10);
    size_t cd = u32(eocd + 16);
    for (uint32_t e = 0; e < count && cd + 46 <= data.size() && u32(cd) == 0x02014b50; e++) {
        const uint32_t method = u16(cd + 10), csize = u32(cd + 20), usize = u32(cd + 24);
        const uint32_t nlen = u16(cd + 28), xlen = u16(cd + 30), clen = u16(cd + 32), lho = u32(cd + 42);
        std::string name(reinterpret_cast<const char*>(&data[cd + 46]), nlen);
        cd += 46 + nlen + xlen + clen;
        if (name.empty() || name.back() == '/') continue;
        bool wanted = false;
        for (const std::string& suf : suffixes)
            if (name.size() >= suf.size() && name.compare(name.size() - suf.size(), suf.size(), suf) == 0) wanted = true;
        if (!wanted || lho + 30 > data.size() || u32(lho) != 0x04034b50) continue;
        const size_t src = lho + 30 + u16(lho + 26) + u16(lho + 28);
        if (src + csize > data.size()) continue;
        std::vector<uint8_t> buf(usize);
        if (method == 0 && csize == usize) {
            memcpy(buf.data(), &data[src], usize);
        } else if (method == 8) {
            z_stream z{};
            if (inflateInit2(&z, -MAX_WBITS) != Z_OK) continue;
            z.next_in = &data[src];
            z.avail_in = csize;
            z.next_out = buf.data();
            z.avail_out = usize;
            const int r = inflate(&z, Z_FINISH);
            inflateEnd(&z);
            if (r != Z_STREAM_END) continue;
        } else {
            continue;
        }
        const size_t slash = name.find_last_of('/');
        const std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
        FILE* w = fopen((dir + base).c_str(), "wb");
        if (!w) continue;
        const bool ok = fwrite(buf.data(), 1, buf.size(), w) == buf.size();
        fclose(w);
        if (ok) out.push_back(base);
    }
    if (out.empty() && err->empty()) *err = "no matching files in the zip";
    return out;
}

// settings.txt `gpu_driver`: the file in <base>/drivers is copied (a .so) or unpacked (a .zip) into the app's
// private folder, since libraries cannot be loaded from shared storage. Sets cfg->gpu_driver_dir/_lib.
void PrepareGpuDriver(android_app* app, const std::string& base, dsr::RuntimeConfig* cfg) {
    const std::string drivers = base + "/drivers";
    mkdir(drivers.c_str(), 0775);
    cfg->native_lib_dir = NativeLibraryDir(app);
    if (cfg->gpu_driver.empty() || cfg->gpu_driver == "system") return;
    const std::string src = drivers + "/" + cfg->gpu_driver;
    if (!FileExists(src)) {
        LOGW(kTag, "gpu_driver: %s not found: using the system driver", src.c_str());
        return;
    }
    const char* priv = app->activity->internalDataPath;
    if (!priv || cfg->native_lib_dir.empty()) {
        LOGW(kTag, "gpu_driver: no private folder or native library folder: using the system driver");
        return;
    }
    const std::string dir = std::string(priv) + "/gpu_driver/";
    MakeDirs(dir);
    if (DIR* d = opendir(dir.c_str())) {  // remove the previous driver's files
        while (dirent* e = readdir(d))
            if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) unlink((dir + e->d_name).c_str());
        closedir(d);
    }
    std::string lib;
    const auto ends = [](const std::string& s, const char* suf) {
        const size_t n = strlen(suf);
        return s.size() >= n && strcasecmp(s.c_str() + s.size() - n, suf) == 0;
    };
    if (ends(src, ".zip")) {
        std::string err;
        const std::vector<std::string> files = Unzip(src, dir, {".so", "meta.json"}, &err);
        // meta.json (AdrenoTools format) names the driver library: "libraryName": "vulkan.ad07xx.so"
        if (FILE* m = fopen((dir + "meta.json").c_str(), "r")) {
            char buf[4096] = {};
            fread(buf, 1, sizeof(buf) - 1, m);
            fclose(m);
            if (const char* k = strstr(buf, "\"libraryName\"")) {
                const char* q1 = strchr(k + 13, '"');
                const char* q2 = q1 ? strchr(q1 + 1, '"') : nullptr;
                if (q1 && q2) lib.assign(q1 + 1, q2);
            }
        }
        if (lib.empty())
            for (const std::string& f : files)
                if (ends(f, ".so")) {
                    lib = f;
                    break;
                }
        if (lib.empty()) {
            LOGW(kTag, "gpu_driver: %s: no driver library found (%s): using the system driver", src.c_str(),
                 err.empty() ? "no .so in it" : err.c_str());
            return;
        }
        LOGI(kTag, "gpu_driver: unpacked %zu file(s) from %s, driver library %s", files.size(), src.c_str(), lib.c_str());
    } else {
        const size_t slash = src.find_last_of('/');
        lib = src.substr(slash + 1);
        if (!CopyFile(src, dir + lib)) {
            LOGW(kTag, "gpu_driver: could not copy %s: using the system driver", src.c_str());
            return;
        }
    }
    if (!FileExists(dir + lib)) {
        LOGW(kTag, "gpu_driver: %s missing after unpacking: using the system driver", lib.c_str());
        return;
    }
    cfg->gpu_driver_dir = dir;
    cfg->gpu_driver_lib = lib;
}

// ---- touch controls (com.dsrnative.app.TouchControls) -------------------------------------------------------
void JNICALL TouchNativeState(JNIEnv*, jclass, jint buttons, jfloat lx, jfloat ly, jfloat rx, jfloat ry) {
    dsr::input::SetTouch(uint64_t(uint32_t(buttons)), lx, ly, rx, g_touch.camera_invert ? -ry : ry);
}
void JNICALL TouchNativeVisible(JNIEnv*, jclass, jboolean visible) {
    g_touch_visible = visible != JNI_FALSE;
    if (!visible) dsr::input::SetTouch(0, 0, 0, 0, 0);
    LOGI(kTag, "touch controls %s", visible ? "shown" : "hidden");
}
jclass g_touch_cls = nullptr;  // global ref
JNIEnv* g_glue_env = nullptr;  // the app glue thread's JNI environment (attached once, for the whole run)

JNIEnv* GlueEnv(android_app* app) {
    if (!g_glue_env) {
        JavaVM* vm = app->activity->vm;
        if (vm->GetEnv(reinterpret_cast<void**>(&g_glue_env), JNI_VERSION_1_6) != JNI_OK || !g_glue_env)
            if (vm->AttachCurrentThread(&g_glue_env, nullptr) != JNI_OK) g_glue_env = nullptr;
    }
    return g_glue_env;
}

// Creates the overlay (TouchControls.attach) on the app glue thread, once the game starts.
void StartTouchControls(android_app* app) {
    JNIEnv* env = GlueEnv(app);
    if (!env) return;
    jobject activity = app->activity->clazz;
    jclass act_cls = env->GetObjectClass(activity);
    jmethodID get_loader = env->GetMethodID(act_cls, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject loader = env->CallObjectMethod(activity, get_loader);
    jclass loader_cls = env->FindClass("java/lang/ClassLoader");
    jmethodID load = env->GetMethodID(loader_cls, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    jstring name = env->NewStringUTF("com.dsrnative.app.TouchControls");
    auto cls = static_cast<jclass>(env->CallObjectMethod(loader, load, name));
    env->DeleteLocalRef(name);
    if (env->ExceptionCheck() || !cls) {
        env->ExceptionClear();
        LOGW(kTag, "touch controls: class not found");
        return;
    }
    const JNINativeMethod methods[] = {
        {const_cast<char*>("nativeState"), const_cast<char*>("(IFFFF)V"), reinterpret_cast<void*>(&TouchNativeState)},
        {const_cast<char*>("nativeVisible"), const_cast<char*>("(Z)V"), reinterpret_cast<void*>(&TouchNativeVisible)},
    };
    if (env->RegisterNatives(cls, methods, 2) != JNI_OK || env->ExceptionCheck()) {
        env->ExceptionClear();
        LOGW(kTag, "touch controls: could not register natives");
        return;
    }
    g_touch_cls = static_cast<jclass>(env->NewGlobalRef(cls));
    jmethodID attach = env->GetStaticMethodID(cls, "attach", "(Landroid/app/Activity;FFZ)V");
    if (attach)
        env->CallStaticVoidMethod(cls, attach, activity, jfloat(g_touch.opacity), jfloat(g_touch.camera_speed),
                                  jboolean(g_touch.enabled));
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

// A controller was used while the touch controls show: hide them.
void HideTouchControlsForController(android_app* app) {
    if (!g_touch_visible.exchange(false) || !g_touch_cls) return;
    JNIEnv* env = GlueEnv(app);
    if (!env) return;
    static jmethodID used = env->GetStaticMethodID(g_touch_cls, "controllerUsed", "(Landroid/app/Activity;)V");
    if (used) env->CallStaticVoidMethod(g_touch_cls, used, app->activity->clazz);
    if (env->ExceptionCheck()) env->ExceptionClear();
}

void StartGame(android_app* app) {
    if (g_game_started) return;
    // Wait for the answer to the storage access request (the settings screen is up on a first launch).
    const int access = StorageAccess(app);
    if (access == 0) {
        static bool logged = false;
        if (!logged) LOGI(kTag, "waiting for storage access (\"All files access\" for Download/DSRNative)");
        logged = true;
        return;
    }
    if (g_game_started.exchange(true)) return;
    const std::string old_dir = app->activity->externalDataPath ? app->activity->externalDataPath : "";
    std::string base = kSharedDir;
    if (access == 1) MakeDirs(base);
    if (access != 1 || !CanWriteIn(base)) {
        LOGW(kTag, "no access to %s (%s): using the app's private folder %s instead", kSharedDir,
             access == 1 ? strerror(errno) : "storage access refused", old_dir.c_str());
        base = old_dir.empty() ? "/sdcard" : old_dir;
    } else {
        LOGI(kTag, "app folder: %s", base.c_str());
        MigrateFromAppFolder(old_dir, base);
    }
    dsr::RuntimeConfig cfg;
    cfg.game_dir = base + "/game";
    cfg.save_dir = base + "/save";
    cfg.cache_dir = base + "/cache";
    // On a first launch the app's own folder may not exist yet (Android creates it lazily), and then nothing
    // below could be written. Create it and everything the user is expected to find in it.
    MakeDirs(base);
    mkdir(cfg.game_dir.c_str(), 0775);
    mkdir(cfg.save_dir.c_str(), 0775);
    mkdir(cfg.cache_dir.c_str(), 0775);
    LoadSettings(base + "/settings.txt", &cfg);
    PrepareGpuDriver(app, base, &cfg);
    if (!FileExists(cfg.game_dir + "/exefs/main") && base != old_dir && FileExists(old_dir + "/game/exefs/main")) {
        LOGW(kTag, "game files are still in %s/game: using them from there; move them to %s/game", old_dir.c_str(),
             base.c_str());
        cfg.game_dir = old_dir + "/game";
    }
    LOGI(kTag, "game dir: %s", cfg.game_dir.c_str());
    if (!FileExists(cfg.game_dir + "/exefs/main")) {
        const std::string note = cfg.game_dir + "/PUT_GAME_FILES_HERE.txt";
        if (!FileExists(note))
            if (FILE* w = fopen(note.c_str(), "w")) {
                fputs("Copy the contents of the extracted Dark Souls Remastered (Switch) folder here: the folder\n"
                      "that contains exefs/, dvdbnd0.bdt, dvdbnd0.bhd5 and so on, so that this folder holds\n"
                      "exefs/main directly. Then start the app again.\n"
                      "Settings are in settings.txt next to this folder.\n",
                      w);
                fclose(w);
            }
        LOGE(kTag, "game files not found: expected %s/exefs/main", cfg.game_dir.c_str());
        LOGE(kTag, "copy the extracted game folder (the one containing exefs/ and dvdbnd0.bdt) there, e.g.:");
        LOGE(kTag, "  adb push DarkSoulsR/. %s/", cfg.game_dir.c_str());
        return;
    }
    StartTouchControls(app);
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

int32_t OnInput(android_app* app, AInputEvent* ev) {
    using namespace dsr::input;
    const int32_t type = AInputEvent_getType(ev);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        const int32_t src = AInputEvent_getSource(ev);
        if (g_touch_visible && KeyToButton(AKeyEvent_getKeyCode(ev)) &&
            ((src & AINPUT_SOURCE_GAMEPAD) == AINPUT_SOURCE_GAMEPAD || (src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK ||
             (src & AINPUT_SOURCE_DPAD) == AINPUT_SOURCE_DPAD))
            HideTouchControlsForController(app);
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
        if (g_touch_visible) {
            float m = 0;
            for (int32_t axis : {AMOTION_EVENT_AXIS_X, AMOTION_EVENT_AXIS_Y, AMOTION_EVENT_AXIS_Z, AMOTION_EVENT_AXIS_RZ})
                m = std::fmax(m, std::fabs(AMotionEvent_getAxisValue(ev, axis, 0)));
            if (m > 0.5f) HideTouchControlsForController(app);  // a real push, not stick noise
        }
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
        case APP_CMD_RESUME:
            dsr::platform::SetPaused(false);
            if (app->window) StartGame(app);  // back from the storage access screen
            break;
        case APP_CMD_GAINED_FOCUS:
            if (app->window) StartGame(app);  // back from the storage access screen
            break;
        case APP_CMD_PAUSE:
            // Home, app switcher, screen off: the game is held (no frames, sound or busy threads) until RESUME.
            if (g_game_started) dsr::platform::SetPaused(true);
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
    if (g_glue_env) app->activity->vm->DetachCurrentThread();  // attached for the touch controls
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
