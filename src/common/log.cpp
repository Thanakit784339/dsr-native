#include "common/log.h"

#include <atomic>
#include <cstdlib>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace dsr {

static std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};

void SetLogLevel(LogLevel lvl) { g_level = static_cast<int>(lvl); }

void LogV(LogLevel lvl, const char* tag, const char* fmt, va_list ap) {
    if (static_cast<int>(lvl) < g_level.load(std::memory_order_relaxed)) return;
#if defined(__ANDROID__)
    static const int prio[] = {ANDROID_LOG_VERBOSE, ANDROID_LOG_DEBUG, ANDROID_LOG_INFO,
                               ANDROID_LOG_WARN, ANDROID_LOG_ERROR, ANDROID_LOG_FATAL};
    char full[64];
    snprintf(full, sizeof(full), "DSR/%s", tag);
    __android_log_vprint(prio[static_cast<int>(lvl)], full, fmt, ap);
#else
    static const char* names = "TDIWEF";
    fprintf(stderr, "[%c] %-8s ", names[static_cast<int>(lvl)], tag);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
#endif
}

void Log(LogLevel lvl, const char* tag, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    LogV(lvl, tag, fmt, ap);
    va_end(ap);
}

void Fatal(const char* tag, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    LogV(LogLevel::Fatal, tag, fmt, ap);
    va_end(ap);
    abort();
}

}  // namespace dsr
