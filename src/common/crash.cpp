// Crash reporting. Everything in the signal handler is async-signal-safe: no malloc, no stdio, only
// write() into a stack buffer.
#include "common/crash.h"

#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "common/log.h"

namespace dsr {
namespace {

struct ModuleInfo {
    const char* name;
    uintptr_t base;
    size_t size;
};
constexpr int kMaxModules = 8;
ModuleInfo g_modules[kMaxModules];
volatile int g_num_modules = 0;

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP};
struct sigaction g_old[sizeof(kSignals) / sizeof(kSignals[0])];
volatile sig_atomic_t g_in_handler = 0;

// ---- tiny async-signal-safe string builder ----------------------------------------------------------
struct Buf {
    char data[8192];
    size_t len = 0;
    void Str(const char* s) {
        while (*s && len < sizeof(data) - 1) data[len++] = *s++;
    }
    void Hex(uint64_t v, int digits = 16) {
        char tmp[17];
        for (int i = digits - 1; i >= 0; i--) {
            tmp[i] = "0123456789abcdef"[v & 0xF];
            v >>= 4;
        }
        tmp[digits] = 0;
        Str("0x");
        Str(tmp);
    }
    void Dec(int64_t v) {
        char tmp[24];
        int i = 23;
        tmp[i] = 0;
        bool neg = v < 0;
        uint64_t u = neg ? uint64_t(-v) : uint64_t(v);
        do {
            tmp[--i] = char('0' + u % 10);
            u /= 10;
        } while (u);
        if (neg) tmp[--i] = '-';
        Str(tmp + i);
    }
    // "0x... (main+0x1234)" when the address is inside a registered module
    void Addr(uint64_t a) {
        Hex(a);
        for (int i = 0; i < g_num_modules; i++) {
            const ModuleInfo& m = g_modules[i];
            if (a >= m.base && a < m.base + m.size) {
                Str(" (");
                Str(m.name);
                Str("+");
                Hex(a - m.base, 8);
                Str(")");
                break;
            }
        }
    }
};

const char* SignalName(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (invalid memory access)";
        case SIGBUS: return "SIGBUS (bus error / misaligned access)";
        case SIGILL: return "SIGILL (illegal instruction)";
        case SIGFPE: return "SIGFPE (arithmetic error)";
        case SIGABRT: return "SIGABRT (abort)";
        case SIGTRAP: return "SIGTRAP (breakpoint / trap)";
        default: return "signal";
    }
}

void Emit(const Buf& b) {
    int fd = LogFileFd();
    if (fd >= 0) {
        ssize_t r = write(fd, b.data, b.len);
        (void)r;
    }
#if defined(__ANDROID__)
    // __android_log_write is not formally async-signal-safe but is what debuggerd-style handlers use.
    char copy[sizeof(b.data)];
    memcpy(copy, b.data, b.len);
    copy[b.len] = 0;
    __android_log_write(ANDROID_LOG_FATAL, "DSR/crash", copy);
#else
    ssize_t r = write(STDERR_FILENO, b.data, b.len);
    (void)r;
#endif
}

void Handler(int sig, siginfo_t* info, void* ctx) {
    if (!g_in_handler) {
        g_in_handler = 1;
        Buf b;
        char thread[16] = "?";
        pthread_getname_np(pthread_self(), thread, sizeof(thread));
        b.Str("\n================ CRASH ================\n");
        b.Str("signal: ");
        b.Str(SignalName(sig));
        b.Str("  code ");
        b.Dec(info ? info->si_code : 0);
        b.Str("\nthread: ");
        b.Str(thread);
        b.Str("  tid ");
        b.Dec(gettid());
        if (sig == SIGSEGV || sig == SIGBUS) {
            b.Str("\nfault address: ");
            b.Addr(reinterpret_cast<uint64_t>(info ? info->si_addr : nullptr));
        }
#if defined(__aarch64__)
        const auto* uc = static_cast<const ucontext_t*>(ctx);
        const auto& mc = uc->uc_mcontext;
        b.Str("\npc: ");
        b.Addr(mc.pc);
        b.Str("\nlr: ");
        b.Addr(mc.regs[30]);
        b.Str("\nsp: ");
        b.Hex(mc.sp);
        b.Str("\n");
        for (int i = 0; i < 30; i++) {
            b.Str(i < 10 ? "x" : "x");
            b.Dec(i);
            b.Str(i < 10 ? ":  " : ": ");
            b.Addr(mc.regs[i]);
            b.Str((i % 2) ? "\n" : "    ");
        }
        // Return addresses on the stack that point into guest code: a cheap backtrace for frame-pointer
        // based guest code (the Switch toolchain keeps x29 frame records).
        b.Str("\nframe chain (x29):\n");
        uint64_t fp = mc.regs[29];
        for (int depth = 0; depth < 24 && fp && (fp & 0xF) == 0; depth++) {
            // Only follow frames inside the current stack area to avoid faulting in the handler.
            if (fp < mc.sp || fp - mc.sp > (64u << 20)) break;
            const uint64_t* rec = reinterpret_cast<const uint64_t*>(fp);
            b.Str("  #");
            b.Dec(depth);
            b.Str(" ");
            b.Addr(rec[1]);
            b.Str("\n");
            if (rec[0] <= fp) break;
            fp = rec[0];
        }
#else
        (void)ctx;
        b.Str("\n");
#endif
        b.Str("=======================================\n");
        Emit(b);
    }
    // Hand over to the previous handler (debuggerd tombstone, etc.).
    for (size_t i = 0; i < sizeof(kSignals) / sizeof(kSignals[0]); i++)
        if (kSignals[i] == sig) sigaction(sig, &g_old[i], nullptr);
    // Faults re-trigger when the instruction re-executes; re-raise the others explicitly.
    if (sig == SIGABRT || sig == SIGTRAP || (info && info->si_code <= 0)) raise(sig);
}

}  // namespace

void InstallCrashHandler() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = Handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (size_t i = 0; i < sizeof(kSignals) / sizeof(kSignals[0]); i++) sigaction(kSignals[i], &sa, &g_old[i]);
}

void RegisterCrashModule(const char* name, uintptr_t base, size_t size) {
    int n = g_num_modules;
    if (n >= kMaxModules) return;
    g_modules[n] = {name, base, size};
    g_num_modules = n + 1;
}

}  // namespace dsr
