#pragma once
// Shared internals of the nn::os HLE (threads, synchronization, time).

#include <atomic>
#include <cstdint>
#include <string>

namespace dsr::hle::os {

// ---- Switch tick (19.2 MHz) ----------------------------------------------------------------------
constexpr int64_t kTickFrequency = 19'200'000;
int64_t NowNs();                        // CLOCK_MONOTONIC
int64_t NowTicks();
int64_t TicksToNs(int64_t ticks);
int64_t NsToTicks(int64_t ns);

// ---- futex primitives operating on guest memory --------------------------------------------------
// Returns false on timeout. timeout_ns < 0 waits forever.
bool FutexWait(std::atomic<uint32_t>* addr, uint32_t expected, int64_t timeout_ns);
void FutexWake(std::atomic<uint32_t>* addr, int count);

// 4-byte lock word compatible with nn::os InternalCriticalSection storage (zero = unlocked).
struct CritSec {
    static void Lock(std::atomic<uint32_t>* w);
    static bool TryLock(std::atomic<uint32_t>* w);
    static void Unlock(std::atomic<uint32_t>* w);
};

// 4-byte condition variable word compatible with nn::os InternalConditionVariable storage.
struct CondVar {
    // Atomically releases cs, waits, re-acquires cs. Returns false on timeout.
    static bool Wait(std::atomic<uint32_t>* cv, std::atomic<uint32_t>* cs, int64_t timeout_ns);
    static void Signal(std::atomic<uint32_t>* cv);
    static void Broadcast(std::atomic<uint32_t>* cv);
};

template <typename T>
std::atomic<T>* AtomicAt(void* base, size_t off) {
    return reinterpret_cast<std::atomic<T>*>(static_cast<uint8_t*>(base) + off);
}
template <typename T>
T& FieldAt(void* base, size_t off) {
    return *reinterpret_cast<T*>(static_cast<uint8_t*>(base) + off);
}

// ---- threads ------------------------------------------------------------------------------------
using GuestThread = void;  // nn::os::ThreadType (opaque, 0x1C0 bytes on SDK 4.x)

// Returns the guest ThreadType* representing the calling host thread. Threads not created via
// nn::os::CreateThread (e.g. the runtime's main game thread) get an adopted ThreadType.
GuestThread* CurrentThread();

// Called on the game thread before guest code runs; adopts it as the Switch main thread.
void AdoptMainThread(const char* name, size_t stack_size, int priority);

// ---- "global wake epoch" used by MultiWait: any event/semaphore/queue change bumps it ------------
void BumpWaitEpoch();
uint32_t WaitEpoch();
bool WaitForEpochChange(uint32_t seen, int64_t timeout_ns);

// ---- host-backed system events (nn::os::SystemEventType) -----------------------------------------
// The guest struct's first 8 bytes hold a pointer to a HostSystemEvent created by HLE services.
struct HostSystemEvent;
void InitSystemEvent(void* system_event_type, bool auto_clear, bool signaled);
void SignalSystemEvent(void* system_event_type);

// Signals a guest nn::os::EventType from host code (e.g. a decoder callback thread).
void SignalGuestEvent(void* event_type);
void ClearGuestEvent(void* event_type);

// ---- MultiWait participation --------------------------------------------------------------------
// Returns true if the object a holder refers to is currently signaled.
bool IsEventSignaled(void* event_type);

}  // namespace dsr::hle::os
