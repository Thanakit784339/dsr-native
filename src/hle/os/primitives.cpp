#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <climits>
#include <thread>

#include "hle/os/os.h"

namespace dsr::hle::os {

int64_t NowNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

// 19.2 MHz: ticks = ns * 12 / 625
int64_t NsToTicks(int64_t ns) { return int64_t((__int128)ns * 12 / 625); }
int64_t TicksToNs(int64_t t) { return int64_t((__int128)t * 625 / 12); }
int64_t NowTicks() { return NsToTicks(NowNs()); }

bool FutexWait(std::atomic<uint32_t>* addr, uint32_t expected, int64_t timeout_ns) {
    timespec ts, *pts = nullptr;
    if (timeout_ns >= 0) {
        ts.tv_sec = timeout_ns / 1'000'000'000;
        ts.tv_nsec = timeout_ns % 1'000'000'000;
        pts = &ts;
    }
    long r = syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAIT_PRIVATE, expected, pts, nullptr, 0);
    return !(r == -1 && errno == ETIMEDOUT);
}

void FutexWake(std::atomic<uint32_t>* addr, int count) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAKE_PRIVATE, count, nullptr, nullptr, 0);
}

// Classic three-state futex mutex (0 = free, 1 = locked, 2 = locked with waiters).
void CritSec::Lock(std::atomic<uint32_t>* w) {
    uint32_t c = 0;
    if (w->compare_exchange_strong(c, 1, std::memory_order_acquire)) return;
    if (c != 2) c = w->exchange(2, std::memory_order_acquire);
    while (c != 0) {
        FutexWait(w, 2, -1);
        c = w->exchange(2, std::memory_order_acquire);
    }
}

bool CritSec::TryLock(std::atomic<uint32_t>* w) {
    uint32_t c = 0;
    return w->compare_exchange_strong(c, 1, std::memory_order_acquire);
}

void CritSec::Unlock(std::atomic<uint32_t>* w) {
    if (w->exchange(0, std::memory_order_release) == 2) FutexWake(w, 1);
}

bool CondVar::Wait(std::atomic<uint32_t>* cv, std::atomic<uint32_t>* cs, int64_t timeout_ns) {
    uint32_t seq = cv->load(std::memory_order_relaxed);
    CritSec::Unlock(cs);
    bool ok = FutexWait(cv, seq, timeout_ns);
    CritSec::Lock(cs);
    return ok;
}

void CondVar::Signal(std::atomic<uint32_t>* cv) {
    cv->fetch_add(1, std::memory_order_release);
    FutexWake(cv, 1);
}

void CondVar::Broadcast(std::atomic<uint32_t>* cv) {
    cv->fetch_add(1, std::memory_order_release);
    FutexWake(cv, INT_MAX);
}

// ---- wait epoch ----------------------------------------------------------------------------------
static std::atomic<uint32_t> g_epoch{0};

void BumpWaitEpoch() {
    g_epoch.fetch_add(1, std::memory_order_release);
    FutexWake(&g_epoch, INT_MAX);
}
uint32_t WaitEpoch() { return g_epoch.load(std::memory_order_acquire); }
bool WaitForEpochChange(uint32_t seen, int64_t timeout_ns) { return FutexWait(&g_epoch, seen, timeout_ns); }

}  // namespace dsr::hle::os
