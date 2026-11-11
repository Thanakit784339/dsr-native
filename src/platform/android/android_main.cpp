// Android entry point (NativeActivity + native_app_glue).
//
// Files: everything lives in Download/DSRNative on the phone's shared storage (needs "All files access", asked
// for on first launch; refused -> the app's private folder Android/data/com.dsrnative.app/files instead):
//   game/          the extracted game (the folder containing exefs/, dvdbnd0.bdt, ...)
//   save/ cache/   save data, GPU pipeline cache
//   settings.txt
//   logs/          dsr_<date>_<time>.txt run logs, frame captures, shader dumps (also logcat: adb logcat -s 'DSR/*')
#include <android/input.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <spawn.h>
#include <dirent.h>
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
#include <mutex>
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
std::mutex g_pad_mutex;  // guards g_pad: written by the input-event thread, read by the touch UI thread too
dsr::input::PadState g_pad;

// On-screen touch controls (TouchControls.java, over JNI): a separate input source, merged with the
// physical pad below so either can drive the game regardless of which one last moved.
std::atomic<uint64_t> g_touch_buttons{0};
std::atomic<int32_t> g_touch_lx{0}, g_touch_ly{0}, g_touch_rx{0}, g_touch_ry{0};

// Recomputes the combined state and pushes it to nn::hid. Touch sticks only override the physical ones
// while actually being dragged (zero means "not touched"), so a connected gamepad still works normally.
void PushMergedPad() {
    using namespace dsr::input;
    PadState s;
    {
        std::lock_guard<std::mutex> lk(g_pad_mutex);
        s = g_pad;
    }
    s.buttons |= g_touch_buttons.load(std::memory_order_relaxed);
    const int32_t tlx = g_touch_lx.load(std::memory_order_relaxed), tly = g_touch_ly.load(std::memory_order_relaxed);
    const int32_t trx = g_touch_rx.load(std::memory_order_relaxed), try_ = g_touch_ry.load(std::memory_order_relaxed);
    if (tlx || tly) { s.lx = tlx; s.ly = tly; }
    if (trx || try_) { s.rx = trx; s.ry = try_; }
    SetPad(s);
}

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
                    "# max_anisotropy: caps anisotropic filtering regardless of what the game asks for (1.0-16.0).\n"
                    "# Past ~4x the visual gain is marginal but GPU sampling cost keeps climbing -- lower trades\n"
                    "# sharpness on textures at a grazing angle (floors, ground) for performance.\n"
                    "max_anisotropy = %.1f\n"
                    "# pin_render_thread: 1 = keep the render thread on the fastest CPU core (was slower on an\n"
                    "# 8 Gen 2: the GPU then waited; 0 = let Android choose)\n"
                    "pin_render_thread = 0\n"
                    "# record_thread: 1 = record GPU commands on a second CPU thread (faster), 0 = all on one thread\n"
                    "record_thread = 1\n"
                    "# fast_cores: 1 = keep the game's main, recording and submit threads on the performance cores\n"
                    "fast_cores = 1\n"
                    "# audio_smooth: 1 = release audio buffers to the game at a smoothed pace (experimental)\n"
                    "audio_smooth = 0\n"
                    "# mali_barriers: 1 = narrow layout-transition barriers to just the stages that actually read\n"
                    "# the image, instead of always blocking everything (lets a tile-based GPU overlap more\n"
                    "# between passes); 0 = the old maximally-conservative behavior\n"
                    "mali_barriers = 1\n"
                    PRESENT_MODE_HELP
                    "present_mode = mailbox\n"
                    FACE_BUTTONS_HELP
                    "face_buttons = label\n"
                    "# docked: nn::oe::GetOperationMode -- the same flag real hardware uses to tell a Switch\n"
                    "# game it's docked vs handheld. Most games (including this one) render at a higher internal\n"
                    "# resolution when docked. 0 = handheld (720p), 1 = docked (usually ~1080p, more demanding)\n"
                    "docked = %d\n"
                    "# render_scale: internal render resolution, as a fraction of the game's normal frame\n"
                    "# (0.1-1.0). The result is still shown at full screen size (upscaled, same as Present()\n"
                    "# always did for a mismatched size) -- lower trades sharpness for GPU headroom.\n"
                    "render_scale = %.2f\n"
                    "# texture_mip_skip: skip this many of a big game texture's own top mip levels (0 = off, full\n"
                    "# quality; 1 = roughly 75%% less data to load per big texture -- softer up close, cuts\n"
                    "# upload bandwidth hard during a zone load)\n"
                    "texture_mip_skip = %d\n"
                    "# astc_transcode: 1 = transcode BC textures to ASTC instead of RGBA8 when the GPU has no\n"
                    "# native BC support (real upload bandwidth win); 0 = always use the RGBA8 fallback. Off by\n"
                    "# default -- reproducibly crashes on some devices right now (see vk_backend.cpp)\n"
                    "astc_transcode = %d\n",
                    cfg->target_fps, cfg->async_pipelines ? 1 : 0, cfg->auto_capture ? 1 : 0, cfg->profile ? 1 : 0,
                    cfg->gamma, cfg->exposure, cfg->max_anisotropy, cfg->docked ? 1 : 0, cfg->render_scale,
                    cfg->texture_mip_skip, cfg->astc_transcode ? 1 : 0);
            fclose(w);
            LOGI(kTag, "settings: wrote defaults to %s", path.c_str());
        } else {
            LOGW(kTag, "settings: cannot create %s (%s): using defaults", path.c_str(), strerror(errno));
        }
    } else {
        bool have_present_mode = false, have_face_buttons = false, have_docked = false, have_render_scale = false,
             have_texture_mip_skip = false, have_astc_transcode = false;
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
            else if (!strcmp(key, "mali_barriers")) cfg->mali_barriers = v != 0;
            else if (!strcmp(key, "docked")) {
                cfg->docked = v != 0;
                have_docked = true;
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
            } else if (!strcmp(key, "max_anisotropy")) {
                const float a = float(atof(value));
                cfg->max_anisotropy = a >= 1.0f && a <= 16.0f ? a : dsr::RuntimeConfig{}.max_anisotropy;
            } else if (!strcmp(key, "render_scale")) {
                const float rs = float(atof(value));
                cfg->render_scale = rs >= 0.1f && rs <= 1.0f ? rs : 1.0f;
                have_render_scale = true;
            } else if (!strcmp(key, "texture_mip_skip")) {
                cfg->texture_mip_skip = std::clamp(v, 0, 4);
                have_texture_mip_skip = true;
            } else if (!strcmp(key, "astc_transcode")) {
                cfg->astc_transcode = v != 0;
                have_astc_transcode = true;
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
        if (!have_docked)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n# docked: nn::oe::GetOperationMode -- the same flag real hardware uses to tell a Switch\n"
                      "# game it's docked vs handheld. Most games (including this one) render at a higher internal\n"
                      "# resolution when docked. 0 = handheld (720p), 1 = docked (usually ~1080p, more demanding)\n"
                      "docked = 0\n",
                      w);
                fclose(w);
                LOGI(kTag, "settings: added docked to %s", path.c_str());
            }
        if (!have_face_buttons)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n" FACE_BUTTONS_HELP "face_buttons = label\n", w);
                fclose(w);
                LOGI(kTag, "settings: added face_buttons to %s", path.c_str());
            }
        if (!have_render_scale)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n# render_scale: internal render resolution, as a fraction of the game's normal frame\n"
                      "# (0.1-1.0). The result is still shown at full screen size (upscaled, same as Present()\n"
                      "# always did for a mismatched size) -- lower trades sharpness for GPU headroom.\n"
                      "render_scale = 1.00\n",
                      w);
                fclose(w);
                LOGI(kTag, "settings: added render_scale to %s", path.c_str());
            }
        if (!have_texture_mip_skip)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n# texture_mip_skip: skip this many of a big game texture's own top mip levels (0 = off,\n"
                      "# full quality; 1 = roughly 75% less data to load per big texture -- softer up close, cuts\n"
                      "# upload bandwidth hard during a zone load)\n"
                      "texture_mip_skip = 1\n",
                      w);
                fclose(w);
                LOGI(kTag, "settings: added texture_mip_skip to %s", path.c_str());
            }
        if (!have_astc_transcode)
            if (FILE* w = fopen(path.c_str(), "a")) {
                fputs("\n# astc_transcode: 1 = transcode BC textures to ASTC instead of RGBA8 when the GPU has\n"
                      "# no native BC support; 0 = always use the RGBA8 fallback. Off by default -- see\n"
                      "# vk_backend.cpp for why.\n"
                      "astc_transcode = 0\n",
                      w);
                fclose(w);
                LOGI(kTag, "settings: added astc_transcode to %s", path.c_str());
            }
    }
    LOGI(kTag, "settings (%s): fps %d, async pipelines %s, auto capture %s, profile %s, gamma %.2f, exposure %.2f, "
         "max anisotropy %.1fx, present mode %s, face buttons by %s, docked %s, render scale %.2f, texture mip "
         "skip %d, astc transcode %s, mali barriers %s",
         path.c_str(), cfg->target_fps, cfg->async_pipelines ? "on" : "off", cfg->auto_capture ? "on" : "off",
         cfg->profile ? "on" : "off", cfg->gamma, cfg->exposure, cfg->max_anisotropy, cfg->present_mode.c_str(),
         g_face_by_position ? "position" : "label", cfg->docked ? "on" : "off", cfg->render_scale,
         cfg->texture_mip_skip, cfg->astc_transcode ? "on" : "off", cfg->mali_barriers ? "on" : "off");
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
        {
            std::lock_guard<std::mutex> lk(g_pad_mutex);
            const uint64_t before = g_pad.buttons;
            if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_DOWN)
                g_pad.buttons |= b;
            else if (AKeyEvent_getAction(ev) == AKEY_EVENT_ACTION_UP)
                g_pad.buttons &= ~b;
            // Diagnostics: pressing both sticks in (L3 + R3) saves a frame capture (dsr_frame_*.bin) of the next
            // frame.
            const uint64_t both = kStickL | kStickR;
            if ((g_pad.buttons & both) == both && (before & both) != both) {
                LOGI(kTag, "L3 + R3: frame capture requested");
                dsr::hle::nvn::GetBackend().RequestCapture(0);
            }
        }
        PushMergedPad();
        return 1;
    }
    if (type == AINPUT_EVENT_TYPE_MOTION && (AInputEvent_getSource(ev) & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        {
            std::lock_guard<std::mutex> lk(g_pad_mutex);
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
        }
        PushMergedPad();
        return 1;
    }
    return 0;
}

// ---- on-screen touch controls (TouchControls.java) ----------------------------------------------------
extern "C" JNIEXPORT void JNICALL Java_com_dsrnative_app_TouchControls_nativeSetTouchButtons(JNIEnv*, jobject,
                                                                                              jlong mask) {
    g_touch_buttons.store(uint64_t(mask), std::memory_order_relaxed);
    PushMergedPad();
}
extern "C" JNIEXPORT void JNICALL Java_com_dsrnative_app_TouchControls_nativeSetTouchStick(JNIEnv*, jobject,
                                                                                            jint which, jint x,
                                                                                            jint y) {
    if (which == 0) {
        g_touch_lx.store(x, std::memory_order_relaxed);
        g_touch_ly.store(y, std::memory_order_relaxed);
    } else {
        g_touch_rx.store(x, std::memory_order_relaxed);
        g_touch_ry.store(y, std::memory_order_relaxed);
    }
    PushMergedPad();
}

// ---- on-screen FPS counter (GameActivity's overlay) --------------------------------------------------
extern "C" JNIEXPORT jfloat JNICALL Java_com_dsrnative_app_GameActivity_nativeGetFps(JNIEnv*, jobject) {
    return dsr::hle::nvn::CurrentFps();
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
        case APP_CMD_GAINED_FOCUS:
            if (app->window) StartGame(app);  // back from the storage access screen
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
