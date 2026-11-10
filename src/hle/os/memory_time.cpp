// nn::os memory heap, ticks and randomness.
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <map>
#include <mutex>
#include <random>

#include "common/log.h"
#include "hle/os/heap.h"
#include "hle/os/os.h"
#include "hle/registry.h"

namespace dsr::hle::os {
namespace {
constexpr const char* kTag = "os.mem";
constexpr size_t kHeapReserve = size_t(6) << 30;  // address space reserved for the guest heap
constexpr size_t kBlockUnit = 2 << 20;            // nn::os::MemoryBlockUnitSize
constexpr uint32_t kResultOutOfMemory = 0x1A01 + 0;  // approximation; game only checks IsSuccess

std::mutex g_heap_mutex;
uintptr_t g_heap_base = 0;
size_t g_heap_size = 0;
std::map<uintptr_t, size_t> g_free;   // start -> size (within [base, base+size))
std::map<uintptr_t, size_t> g_used;
size_t g_used_bytes = 0;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

void InsertFree(uintptr_t start, size_t size) {
    auto it = g_free.emplace(start, size).first;
    if (it != g_free.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            g_free.erase(it);
            it = prev;
        }
    }
    auto next = std::next(it);
    if (next != g_free.end() && it->first + it->second == next->first) {
        it->second += next->second;
        g_free.erase(next);
    }
}
}  // namespace

void GuestHeapRange(uintptr_t* base, size_t* size) {
    std::lock_guard<std::mutex> lk(g_heap_mutex);
    *base = g_heap_base;
    *size = g_heap_size;
}

bool GuestHeapContains(const void* p) {
    auto a = reinterpret_cast<uintptr_t>(p);
    return g_heap_base && a >= g_heap_base && a < g_heap_base + g_heap_size;
}

}  // namespace dsr::hle::os

using namespace dsr::hle::os;

HLE_FUNC(_ZN2nn2os17SetMemoryHeapSizeEm, uint32_t, nn_os_SetMemoryHeapSize, (size_t size)) {
    std::lock_guard<std::mutex> lk(g_heap_mutex);
    size = AlignUp(size, kBlockUnit);
    if (!g_heap_base) {
        void* p = mmap(nullptr, kHeapReserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) FATAL(kTag, "cannot reserve %zu bytes of address space for the guest heap", kHeapReserve);
        // Align base to the block unit so block addresses match Switch alignment expectations.
        uintptr_t b = AlignUp(reinterpret_cast<uintptr_t>(p), kBlockUnit);
        g_heap_base = b;
    }
    if (size > kHeapReserve - kBlockUnit) return kResultOutOfMemory;
    if (size > g_heap_size) {
        if (mprotect(reinterpret_cast<void*>(g_heap_base + g_heap_size), size - g_heap_size, PROT_READ | PROT_WRITE))
            FATAL(kTag, "mprotect heap growth failed");
        InsertFree(g_heap_base + g_heap_size, size - g_heap_size);
    } else if (size < g_heap_size) {
        LOGW(kTag, "heap shrink request ignored (%zu -> %zu)", g_heap_size, size);
        return 0;
    }
    g_heap_size = size;
    LOGI(kTag, "guest heap: base 0x%lx size %zu MiB", (unsigned long)g_heap_base, size >> 20);
    return 0;
}

HLE_FUNC(_ZN2nn2os19AllocateMemoryBlockEPmm, uint32_t, nn_os_AllocateMemoryBlock, (uintptr_t * out, size_t size)) {
    std::lock_guard<std::mutex> lk(g_heap_mutex);
    size = AlignUp(size, kBlockUnit);
    for (auto it = g_free.begin(); it != g_free.end(); ++it) {
        if (it->second >= size) {
            uintptr_t addr = it->first;
            size_t rest = it->second - size;
            g_free.erase(it);
            if (rest) g_free.emplace(addr + size, rest);
            g_used[addr] = size;
            g_used_bytes += size;
            *out = addr;
            return 0;
        }
    }
    LOGE(kTag, "AllocateMemoryBlock(%zu) failed: heap exhausted", size);
    return kResultOutOfMemory;
}

HLE_FUNC(_ZN2nn2os15FreeMemoryBlockEmm, void, nn_os_FreeMemoryBlock, (uintptr_t addr, size_t size)) {
    std::lock_guard<std::mutex> lk(g_heap_mutex);
    auto it = g_used.find(addr);
    if (it == g_used.end()) {
        LOGW(kTag, "FreeMemoryBlock of unknown block 0x%lx", (unsigned long)addr);
        return;
    }
    (void)size;
    g_used_bytes -= it->second;
    // Give the pages back to the OS but keep the range mapped.
    madvise(reinterpret_cast<void*>(addr), it->second, MADV_DONTNEED);
    InsertFree(addr, it->second);
    g_used.erase(it);
}

struct MemoryInfo {
    uint64_t total_available;
    size_t total_used;
    size_t total_heap;
    size_t allocated_heap;
    size_t program_size;
    size_t total_thread_stack;
    int thread_count;
};

HLE_FUNC(_ZN2nn2os15QueryMemoryInfoEPNS0_10MemoryInfoE, void, nn_os_QueryMemoryInfo, (MemoryInfo * mi)) {
    std::lock_guard<std::mutex> lk(g_heap_mutex);
    mi->total_available = uint64_t(3264) << 20;  // what a retail Switch reports to applications
    mi->total_heap = g_heap_size;
    mi->allocated_heap = g_used_bytes;
    mi->program_size = 0x6000000;
    mi->total_thread_stack = 0;
    mi->thread_count = 0;
    mi->total_used = mi->program_size + g_heap_size;
}

// ---- ticks ---------------------------------------------------------------------------------------
// Some game threads poll the clock in a tight loop (the NexusRevolution network thread spent ~45% of a core in
// clock_gettime). A thread that has read the clock back to back (under 5 us apart) for more than 1 ms gets a
// 0.5 ms nap every 32 reads from then on, so it stops taking a fast core from the frame and render threads.
// A thread that reads the clock at a normal rate never qualifies; a spin-wait may overshoot by up to 0.5 ms.
HLE_FUNC(_ZN2nn2os13GetSystemTickEv, int64_t, nn_os_GetSystemTick, ()) {
    const int64_t t = NowTicks();
    thread_local int64_t last = 0, streak_start = 0;
    thread_local int dense = 0;
    thread_local bool napped = false;
    if (t - last < 96 || napped) {  // 96 ticks = 5 us at 19.2 MHz; our own nap does not end a streak
        napped = false;
        if (!streak_start) streak_start = t;
        if (++dense >= 32 && t - streak_start > 19200) {
            dense = 0;
            timespec ts{0, 500'000};
            nanosleep(&ts, nullptr);
            napped = true;
        }
    } else {
        dense = 0;
        streak_start = 0;
    }
    last = t;
    return t;
}
HLE_ALIAS(nnosGetSystemTick, nn_os_GetSystemTick);
HLE_FUNC(_ZN2nn2os22GetSystemTickFrequencyEv, int64_t, nn_os_GetSystemTickFrequency, ()) { return kTickFrequency; }
HLE_ALIAS(nnosGetSystemTickFrequency, nn_os_GetSystemTickFrequency);
HLE_FUNC(_ZN2nn2os13ConvertToTickENS_8TimeSpanE, int64_t, nn_os_ConvertToTick, (int64_t ns)) { return NsToTicks(ns); }
HLE_FUNC(_ZN2nn2os17ConvertToTimeSpanENS0_4TickE, int64_t, nn_os_ConvertToTimeSpan, (int64_t t)) {
    return TicksToNs(t);
}
HLE_FUNC(nnosConvertTickToNanoSeconds, int64_t, nnosConvertTickToNanoSeconds, (int64_t t)) { return TicksToNs(t); }

HLE_FUNC(_ZN2nn2os19GenerateRandomBytesEPvm, void, nn_os_GenerateRandomBytes, (void* buf, size_t n)) {
    static std::mutex m;
    static std::random_device rd;
    std::lock_guard<std::mutex> lk(m);
    auto* p = static_cast<uint8_t*>(buf);
    while (n) {
        uint32_t v = rd();
        size_t k = n < 4 ? n : 4;
        memcpy(p, &v, k);
        p += k;
        n -= k;
    }
}
