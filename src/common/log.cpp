#include "common/log.h"

#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace dsr {

static std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
static std::atomic<int> g_file_fd{-1};
static std::mutex g_file_mutex;

void SetLogLevel(LogLevel lvl) { g_level = static_cast<int>(lvl); }

bool OpenLogFile(const char* path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return false;
    int old = g_file_fd.exchange(fd);
    if (old >= 0) close(old);
    return true;
}

int LogFileFd() { return g_file_fd.load(std::memory_order_relaxed); }

static void WriteFileLine(LogLevel lvl, const char* tag, const char* msg) {
    int fd = g_file_fd.load(std::memory_order_relaxed);
    if (fd < 0) return;
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm t;
    localtime_r(&ts.tv_sec, &t);
    char thread[16] = "?";
    pthread_getname_np(pthread_self(), thread, sizeof(thread));
    static const char* names = "TDIWEF";
    char line[4096];
    int n = snprintf(line, sizeof(line), "%02d:%02d:%02d.%03ld %c %-10s [%s] %s\n", t.tm_hour, t.tm_min, t.tm_sec,
                     ts.tv_nsec / 1'000'000, names[static_cast<int>(lvl)], tag, thread, msg);
    if (n < 0) return;
    if (n >= int(sizeof(line))) {
        n = sizeof(line);
        line[n - 1] = '\n';
    }
    // One write() per line and no buffering, so the file is complete even if the process crashes.
    std::lock_guard<std::mutex> lk(g_file_mutex);
    ssize_t r = write(fd, line, size_t(n));
    (void)r;
}

void LogV(LogLevel lvl, const char* tag, const char* fmt, va_list ap) {
    if (static_cast<int>(lvl) < g_level.load(std::memory_order_relaxed)) return;
    char msg[3072];
    vsnprintf(msg, sizeof(msg), fmt, ap);
#if defined(__ANDROID__)
    static const int prio[] = {ANDROID_LOG_VERBOSE, ANDROID_LOG_DEBUG, ANDROID_LOG_INFO,
                               ANDROID_LOG_WARN, ANDROID_LOG_ERROR, ANDROID_LOG_FATAL};
    char full[64];
    snprintf(full, sizeof(full), "DSR/%s", tag);
    __android_log_write(prio[static_cast<int>(lvl)], full, msg);
#else
    static const char* names = "TDIWEF";
    fprintf(stderr, "[%c] %-8s %s\n", names[static_cast<int>(lvl)], tag, msg);
#endif
    WriteFileLine(lvl, tag, msg);
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
