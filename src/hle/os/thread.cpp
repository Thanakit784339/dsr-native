// nn::os threads and thread-local storage.
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "arch/arch.h"
#include "common/log.h"
#include "hle/os/os.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

namespace dsr::hle::os {

namespace {
constexpr const char* kTag = "os.thread";
constexpr uint32_t kResultSuccess = 0;
constexpr uint32_t kResultOutOfResource = 0x1003;  // module 3 (os), desc 8 (approximation)
constexpr size_t kHostStackExtra = 256 * 1024;      // headroom for host code (drivers, logging) on guest threads
constexpr int kTlsSlots = 32;

enum class State : int { Created, Started, Exited };

struct HostThread {
    GuestThread* guest = nullptr;
    const void* entry = nullptr;
    void* arg = nullptr;
    size_t stack_size = 0;
    int priority = 16;
    int ideal_core = 0;
    uint64_t id = 0;
    std::string name;
    const char* name_ptr = nullptr;  // guest-provided pointer (SetThreadNamePointer)
    pthread_t pthread{};
    uintptr_t stack_lo = 0, stack_hi = 0;
    std::mutex m;
    std::atomic<uint32_t> state{0};
};

std::mutex g_threads_mutex;
std::unordered_map<GuestThread*, std::unique_ptr<HostThread>> g_threads;
std::atomic<uint64_t> g_next_id{1};

thread_local HostThread* t_current = nullptr;
thread_local uintptr_t t_tls_values[kTlsSlots];

struct TlsSlotInfo {
    bool used = false;
    const void* destructor = nullptr;  // guest function void(uintptr_t)
};
std::mutex g_tls_mutex;
TlsSlotInfo g_tls_slots[kTlsSlots];

HostThread* Lookup(GuestThread* t) {
    std::lock_guard<std::mutex> lk(g_threads_mutex);
    auto it = g_threads.find(t);
    return it == g_threads.end() ? nullptr : it->second.get();
}

void QueryStack(HostThread* ht) {
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
        void* lo;
        size_t sz;
        pthread_attr_getstack(&attr, &lo, &sz);
        ht->stack_lo = reinterpret_cast<uintptr_t>(lo);
        ht->stack_hi = ht->stack_lo + sz;
        pthread_attr_destroy(&attr);
    }
}

void RunTlsDestructors() {
    for (int i = 0; i < kTlsSlots; i++) {
        const void* dtor;
        {
            std::lock_guard<std::mutex> lk(g_tls_mutex);
            dtor = g_tls_slots[i].used ? g_tls_slots[i].destructor : nullptr;
        }
        if (dtor && t_tls_values[i]) {
            uintptr_t v = t_tls_values[i];
            t_tls_values[i] = 0;
            arch::GuestCall<void>(dtor, v);
        }
    }
}

void* ThreadTrampoline(void* p) {
    auto* ht = static_cast<HostThread*>(p);
    t_current = ht;
    QueryStack(ht);
    pthread_setname_np(pthread_self(), ht->name.empty() ? "guest" : ht->name.substr(0, 15).c_str());
    LOGD(kTag, "thread %llu '%s' start entry=%p arg=%p", (unsigned long long)ht->id, ht->name.c_str(), ht->entry,
         ht->arg);
    arch::GuestCall<void>(ht->entry, ht->arg);
    RunTlsDestructors();
    ht->state.store(uint32_t(State::Exited), std::memory_order_release);
    FutexWake(&ht->state, INT32_MAX);
    LOGD(kTag, "thread %llu exited", (unsigned long long)ht->id);
    return nullptr;
}

uint32_t CreateThreadImpl(GuestThread* t, const void* entry, void* arg, void* stack, size_t stack_size, int prio,
                          int core) {
    (void)stack;  // the guest stack is not used: bionic needs its own guard/TLS layout
    auto ht = std::make_unique<HostThread>();
    ht->guest = t;
    ht->entry = entry;
    ht->arg = arg;
    ht->stack_size = stack_size;
    ht->priority = prio;
    ht->ideal_core = core;
    ht->id = g_next_id.fetch_add(1);
    std::lock_guard<std::mutex> lk(g_threads_mutex);
    g_threads[t] = std::move(ht);
    return kResultSuccess;
}
}  // namespace

GuestThread* CurrentThread() {
    if (!t_current) {
        // Adopt foreign threads lazily (e.g. host callback threads that call into guest code).
        auto* buf = calloc(1, 0x200);
        auto ht = std::make_unique<HostThread>();
        ht->guest = buf;
        ht->id = g_next_id.fetch_add(1);
        ht->name = "adopted";
        ht->state = uint32_t(State::Started);
        QueryStack(ht.get());
        t_current = ht.get();
        std::lock_guard<std::mutex> lk(g_threads_mutex);
        g_threads[buf] = std::move(ht);
    }
    return t_current->guest;
}

void AdoptMainThread(const char* name, size_t stack_size, int priority) {
    GuestThread* t = CurrentThread();
    t_current->name = name;
    t_current->stack_size = stack_size;
    t_current->priority = priority;
    (void)t;
}

}  // namespace dsr::hle::os

using namespace dsr::hle::os;
using dsr::hle::os::CurrentThread;

// ------------------------------------------------------------------------------------------------
HLE_FUNC(_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi, uint32_t, nn_os_CreateThread,
         (GuestThread * t, const void* entry, void* arg, void* stack, size_t size, int prio)) {
    return CreateThreadImpl(t, entry, arg, stack, size, prio, 0);
}

HLE_FUNC(_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mii, uint32_t, nn_os_CreateThreadCore,
         (GuestThread * t, const void* entry, void* arg, void* stack, size_t size, int prio, int core)) {
    return CreateThreadImpl(t, entry, arg, stack, size, prio, core);
}

HLE_FUNC(nnosCreateThread, uint32_t, nnosCreateThread,
         (GuestThread * t, const void* entry, void* arg, void* stack, size_t size, int prio, int core)) {
    return CreateThreadImpl(t, entry, arg, stack, size, prio, core);
}

HLE_FUNC(_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE, void, nn_os_StartThread, (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    if (!ht) FATAL("os.thread", "StartThread on unknown thread %p", t);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, std::max<size_t>(ht->stack_size, 64 * 1024) + kHostStackExtra);
    ht->state.store(uint32_t(State::Started));
    if (pthread_create(&ht->pthread, &attr, ThreadTrampoline, ht) != 0)
        FATAL("os.thread", "pthread_create failed for '%s'", ht->name.c_str());
    pthread_attr_destroy(&attr);
}
HLE_ALIAS(nnosStartThread, nn_os_StartThread);

HLE_FUNC(_ZN2nn2os10WaitThreadEPNS0_10ThreadTypeE, void, nn_os_WaitThread, (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    if (!ht) return;
    uint32_t s;
    while ((s = ht->state.load(std::memory_order_acquire)) != uint32_t(State::Exited)) FutexWait(&ht->state, s, -1);
}
HLE_ALIAS(nnosWaitThread, nn_os_WaitThread);

HLE_FUNC(_ZN2nn2os13DestroyThreadEPNS0_10ThreadTypeE, void, nn_os_DestroyThread, (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    if (!ht) return;
    if (ht->state.load() != uint32_t(State::Created)) {
        nn_os_WaitThread(t);
        pthread_join(ht->pthread, nullptr);
    }
    std::lock_guard<std::mutex> lk(g_threads_mutex);
    g_threads.erase(t);
}
HLE_ALIAS(nnosDestroyThread, nn_os_DestroyThread);

HLE_FUNC(_ZN2nn2os16GetCurrentThreadEv, GuestThread*, nn_os_GetCurrentThread, ()) { return CurrentThread(); }
HLE_ALIAS(nnosGetCurrentThread, nn_os_GetCurrentThread);

HLE_FUNC(_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc, void, nn_os_SetThreadName,
         (GuestThread * t, const char* name)) {
    if (HostThread* ht = Lookup(t)) {
        ht->name = name ? name : "";
        ht->name_ptr = nullptr;
        if (ht == t_current) pthread_setname_np(pthread_self(), ht->name.substr(0, 15).c_str());
    }
}

HLE_FUNC(nnosSetThreadNamePointer, void, nnosSetThreadNamePointer, (GuestThread * t, const char* name)) {
    if (HostThread* ht = Lookup(t)) {
        ht->name = name ? name : "";
        ht->name_ptr = name;
    }
}

HLE_FUNC(_ZN2nn2os20GetThreadNamePointerEPKNS0_10ThreadTypeE, const char*, nn_os_GetThreadNamePointer,
         (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    if (!ht) return "";
    return ht->name_ptr ? ht->name_ptr : ht->name.c_str();
}

HLE_FUNC(_ZN2nn2os17GetThreadPriorityEPKNS0_10ThreadTypeE, int, nn_os_GetThreadPriority, (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    return ht ? ht->priority : 16;
}
HLE_ALIAS(nnosGetThreadPriority, nn_os_GetThreadPriority);

HLE_FUNC(_ZN2nn2os20ChangeThreadPriorityEPNS0_10ThreadTypeEi, int, nn_os_ChangeThreadPriority,
         (GuestThread * t, int prio)) {
    HostThread* ht = Lookup(t);
    if (!ht) return 16;
    int old = ht->priority;
    ht->priority = prio;
    return old;
}

HLE_FUNC(_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim, void, nn_os_SetThreadCoreMask,
         (GuestThread * t, int ideal, uint64_t mask)) {
    if (HostThread* ht = Lookup(t)) ht->ideal_core = ideal;
    (void)mask;
}

HLE_FUNC(_ZN2nn2os11GetThreadIdEPKNS0_10ThreadTypeE, uint64_t, nn_os_GetThreadId, (GuestThread * t)) {
    HostThread* ht = Lookup(t);
    return ht ? ht->id : 0;
}

HLE_FUNC(_ZN2nn2os19GetCurrentStackInfoEPmS1_, void, nn_os_GetCurrentStackInfo, (uintptr_t * stack, size_t* size)) {
    CurrentThread();
    if (stack) *stack = t_current->stack_lo;
    if (size) *size = t_current->stack_hi - t_current->stack_lo;
}

namespace dsr::hle::os {
namespace {
// Diagnostics: who sleeps how long (finds the game's frame limiter). Per caller (guest return address and the
// caller's own return address, both as main+offset), every 10 s, the ones that slept the most.
struct SleepSite {
    uint64_t calls = 0;
    int64_t total_ns = 0, min_ns = INT64_MAX, max_ns = 0;
    std::string thread;
};
std::mutex g_sleep_mutex;
std::map<std::pair<uintptr_t, uintptr_t>, SleepSite> g_sleep_sites;
std::chrono::steady_clock::time_point g_sleep_report = std::chrono::steady_clock::now();

uintptr_t MainOffset(uintptr_t a) {
    const dsr::loader::Module& m = dsr::GetRuntime().main;
    return m.Contains(a) ? a - m.base() : 0;
}

void RecordSleep(uintptr_t ret, uintptr_t frame, int64_t ns) {
    // `frame` is the calling guest function's frame record (its x29): [frame + 8] is where it returns to.
    uintptr_t ret2 = 0;
    if (t_current && frame >= t_current->stack_lo && frame + 16 <= t_current->stack_hi)
        ret2 = *reinterpret_cast<const uintptr_t*>(frame + 8);
    const auto key = std::make_pair(MainOffset(ret), MainOffset(ret2));
    std::lock_guard<std::mutex> lk(g_sleep_mutex);
    SleepSite& site = g_sleep_sites[key];
    site.calls++;
    site.total_ns += ns;
    site.min_ns = std::min(site.min_ns, ns);
    site.max_ns = std::max(site.max_ns, ns);
    if (site.thread.empty() && t_current) site.thread = t_current->name;
    const auto now = std::chrono::steady_clock::now();
    if (now - g_sleep_report < std::chrono::seconds(10)) return;
    g_sleep_report = now;
    std::vector<std::pair<int64_t, std::pair<std::pair<uintptr_t, uintptr_t>, SleepSite>>> top;
    for (auto& [k, v] : g_sleep_sites) top.push_back({v.total_ns, {k, v}});
    std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < top.size() && i < 6; i++) {
        const auto& [k, v] = top[i].second;
        if (v.total_ns < 50'000'000) break;
        LOGI(kTag, "sleep: main+0x%lx (from main+0x%lx) thread '%s': %llu calls, %.1f s total, %.2f-%.2f ms",
             (unsigned long)k.first, (unsigned long)k.second, v.thread.c_str(), (unsigned long long)v.calls,
             v.total_ns / 1e9, v.min_ns / 1e6, v.max_ns / 1e6);
    }
    g_sleep_sites.clear();
}
}  // namespace
}  // namespace dsr::hle::os

HLE_FUNC(_ZN2nn2os11SleepThreadENS_8TimeSpanE, void, nn_os_SleepThread, (int64_t ns)) {
    dsr::hle::os::RecordSleep(reinterpret_cast<uintptr_t>(__builtin_return_address(0)),
                *static_cast<const uintptr_t*>(__builtin_frame_address(0)), ns);  // saved x29: the caller's frame
    if (ns <= 0) {
        sched_yield();
        return;
    }
    timespec ts{time_t(ns / 1'000'000'000), long(ns % 1'000'000'000)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
    }
}
HLE_ALIAS(nnosSleepThread, nn_os_SleepThread);

HLE_FUNC(_ZN2nn2os11YieldThreadEv, void, nn_os_YieldThread, ()) { sched_yield(); }

HLE_FUNC(_ZN2nn2os20GetCurrentCoreNumberEv, int, nn_os_GetCurrentCoreNumber, ()) {
    int c = sched_getcpu();
    return c < 0 ? 0 : c % 3;  // applications see cores 0-2
}

HLE_FUNC(_ZN2nn2os26GetThreadAvailableCoreMaskEv, uint64_t, nn_os_GetThreadAvailableCoreMask, ()) { return 0b111; }

// ---- TLS ----------------------------------------------------------------------------------------
static uint32_t AllocTls(uint32_t* out, const void* dtor) {
    std::lock_guard<std::mutex> lk(g_tls_mutex);
    for (int i = 0; i < kTlsSlots; i++) {
        if (!g_tls_slots[i].used) {
            g_tls_slots[i] = {true, dtor};
            *out = uint32_t(i);
            return kResultSuccess;
        }
    }
    return kResultOutOfResource;
}

HLE_FUNC(_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE, uint32_t, nn_os_AllocateTlsSlot,
         (uint32_t * out, const void* dtor)) {
    return AllocTls(out, dtor);
}
HLE_FUNC(_ZN2nn2os18SdkAllocateTlsSlotEPNS0_7TlsSlotEPFvmE, uint32_t, nn_os_SdkAllocateTlsSlot,
         (uint32_t * out, const void* dtor)) {
    return AllocTls(out, dtor);
}
HLE_FUNC(_ZN2nn2os11FreeTlsSlotENS0_7TlsSlotE, void, nn_os_FreeTlsSlot, (uint32_t slot)) {
    std::lock_guard<std::mutex> lk(g_tls_mutex);
    if (slot < kTlsSlots) g_tls_slots[slot] = {};
}
HLE_FUNC(_ZN2nn2os11GetTlsValueENS0_7TlsSlotE, uintptr_t, nn_os_GetTlsValue, (uint32_t slot)) {
    return slot < kTlsSlots ? t_tls_values[slot] : 0;
}
HLE_FUNC(_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm, void, nn_os_SetTlsValue, (uint32_t slot, uintptr_t v)) {
    if (slot < kTlsSlots) t_tls_values[slot] = v;
}

// ---- host process arguments ---------------------------------------------------------------------
static char* g_argv[] = {const_cast<char*>("DarkSoulsRemastered"), nullptr};
HLE_FUNC(_ZN2nn2os11GetHostArgcEv, int, nn_os_GetHostArgc, ()) { return 1; }
HLE_FUNC(_ZN2nn2os11GetHostArgvEv, char**, nn_os_GetHostArgv, ()) { return g_argv; }
