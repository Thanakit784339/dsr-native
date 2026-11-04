// nn::os synchronization objects.
//
// Mutex/Event/Semaphore/ConditionVariable operate directly on the guest structs using the SDK's own
// field layout (reverse engineered from sdk NSO, SDK 4.5), so statically initialized objects work:
//   MutexType             +0 u8 state  +1 bool recursive  +4 int lockLevel  +8 int nest
//                         +0x10 ThreadType* owner  +0x18 u32 critsec          (size 0x20)
//   EventType             +0/+8 list node  +0x10 bool signaled  +0x11 bool initial  +0x12 u8 clearMode
//                         +0x13 u8 state  +0x14 u32 broadcastCounter  +0x1C u32 critsec  +0x20 u32 cv
//   SemaphoreType         +0/+8 list node  +0x10 u8 state  +0x14 int count  +0x18 int max
//                         +0x1C u32 critsec  +0x20 u32 cv
//   ConditionVariableType +0 u8 state  +4 u32 cv
// ReaderWriterLock, MessageQueue, SystemEvent and MultiWait keep a pointer to a host object in their
// first 8 bytes (they are always initialized through an API call).

#include <climits>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "common/log.h"
#include "hle/os/os.h"
#include "hle/registry.h"

namespace dsr::hle::os {
namespace {
constexpr const char* kTag = "os.sync";

// ---- Mutex --------------------------------------------------------------------------------------
struct MutexView {
    void* p;
    uint8_t& state() { return FieldAt<uint8_t>(p, 0); }
    bool& recursive() { return FieldAt<bool>(p, 1); }
    int& lock_level() { return FieldAt<int>(p, 4); }
    int& nest() { return FieldAt<int>(p, 8); }
    std::atomic<GuestThread*>* owner() { return AtomicAt<GuestThread*>(p, 0x10); }
    std::atomic<uint32_t>* cs() { return AtomicAt<uint32_t>(p, 0x18); }
};

void MutexLock(void* m) {
    MutexView v{m};
    GuestThread* self = CurrentThread();
    if (v.owner()->load(std::memory_order_relaxed) == self) {
        if (!v.recursive()) FATAL(kTag, "non-recursive mutex %p locked twice by the same thread", m);
        v.nest()++;
        return;
    }
    CritSec::Lock(v.cs());
    v.owner()->store(self, std::memory_order_relaxed);
    v.nest() = 1;
}

bool MutexTryLock(void* m) {
    MutexView v{m};
    GuestThread* self = CurrentThread();
    if (v.owner()->load(std::memory_order_relaxed) == self) {
        if (!v.recursive()) return false;
        v.nest()++;
        return true;
    }
    if (!CritSec::TryLock(v.cs())) return false;
    v.owner()->store(self, std::memory_order_relaxed);
    v.nest() = 1;
    return true;
}

void MutexUnlock(void* m) {
    MutexView v{m};
    if (--v.nest() > 0) return;
    v.owner()->store(nullptr, std::memory_order_relaxed);
    CritSec::Unlock(v.cs());
}

// ---- Event --------------------------------------------------------------------------------------
struct EventView {
    void* p;
    bool& signaled() { return FieldAt<bool>(p, 0x10); }
    uint8_t& clear_mode() { return FieldAt<uint8_t>(p, 0x12); }  // 0 manual, 1 auto
    uint32_t& counter() { return FieldAt<uint32_t>(p, 0x14); }
    std::atomic<uint32_t>* cs() { return AtomicAt<uint32_t>(p, 0x1C); }
    std::atomic<uint32_t>* cv() { return AtomicAt<uint32_t>(p, 0x20); }
};

bool EventWait(void* e, int64_t timeout_ns) {
    EventView v{e};
    int64_t deadline = timeout_ns >= 0 ? NowNs() + timeout_ns : -1;
    CritSec::Lock(v.cs());
    uint32_t counter = v.counter();
    while (!v.signaled() && counter == v.counter()) {
        int64_t left = -1;
        if (deadline >= 0) {
            left = deadline - NowNs();
            if (left <= 0) {
                CritSec::Unlock(v.cs());
                return false;
            }
        }
        CondVar::Wait(v.cv(), v.cs(), left);
    }
    if (v.clear_mode() == 1) v.signaled() = false;
    CritSec::Unlock(v.cs());
    return true;
}

// ---- Semaphore ----------------------------------------------------------------------------------
struct SemView {
    void* p;
    int& count() { return FieldAt<int>(p, 0x14); }
    int& max() { return FieldAt<int>(p, 0x18); }
    std::atomic<uint32_t>* cs() { return AtomicAt<uint32_t>(p, 0x1C); }
    std::atomic<uint32_t>* cv() { return AtomicAt<uint32_t>(p, 0x20); }
};

bool SemAcquire(void* s, int64_t timeout_ns) {
    SemView v{s};
    int64_t deadline = timeout_ns >= 0 ? NowNs() + timeout_ns : -1;
    CritSec::Lock(v.cs());
    while (v.count() == 0) {
        int64_t left = -1;
        if (deadline >= 0) {
            left = deadline - NowNs();
            if (left <= 0) {
                CritSec::Unlock(v.cs());
                return false;
            }
        }
        CondVar::Wait(v.cv(), v.cs(), left);
    }
    v.count()--;
    CritSec::Unlock(v.cs());
    return true;
}

// ---- host-backed objects ------------------------------------------------------------------------
template <typename T>
T* HostObj(void* guest) {
    return *reinterpret_cast<T**>(guest);
}
template <typename T>
void SetHostObj(void* guest, T* obj) {
    *reinterpret_cast<T**>(guest) = obj;
}

struct RwLock {
    std::mutex m;
    std::condition_variable cv;
    int readers = 0;
    GuestThread* writer = nullptr;
    int write_nest = 0;
};

struct MessageQueue {
    std::mutex m;
    std::condition_variable not_empty, not_full;
    size_t capacity = 0;
    std::deque<uintptr_t> items;
};

struct MultiWait {
    std::vector<void*> holders;  // MultiWaitHolderType*
};

}  // namespace

struct HostSystemEvent {
    std::mutex m;
    std::condition_variable cv;
    bool signaled = false;
    bool auto_clear = false;
};

void InitSystemEvent(void* se, bool auto_clear, bool signaled) {
    auto* h = new HostSystemEvent;
    h->auto_clear = auto_clear;
    h->signaled = signaled;
    SetHostObj(se, h);
}

static HostSystemEvent* SysEvt(void* se) {
    auto* h = HostObj<HostSystemEvent>(se);
    if (!h) {
        LOGW(kTag, "SystemEvent %p used without an HLE owner; creating an unsignaled one", se);
        InitSystemEvent(se, false, false);
        h = HostObj<HostSystemEvent>(se);
    }
    return h;
}

void SignalSystemEvent(void* se) {
    HostSystemEvent* h = SysEvt(se);
    {
        std::lock_guard<std::mutex> lk(h->m);
        h->signaled = true;
    }
    h->cv.notify_all();
    BumpWaitEpoch();
}

bool IsEventSignaled(void* e) { return EventView{e}.signaled(); }

}  // namespace dsr::hle::os

using namespace dsr::hle::os;

// ===== Mutex =====================================================================================
HLE_FUNC(_ZN2nn2os15InitializeMutexEPNS0_9MutexTypeEbi, void, nn_os_InitializeMutex,
         (void* m, bool recursive, int level)) {
    MutexView v{m};
    memset(m, 0, 0x20);
    v.recursive() = recursive;
    v.lock_level() = level;
    v.state() = 1;
}
HLE_ALIAS(nnosInitializeMutex, nn_os_InitializeMutex);
HLE_FUNC(_ZN2nn2os13FinalizeMutexEPNS0_9MutexTypeE, void, nn_os_FinalizeMutex, (void* m)) { MutexView{m}.state() = 0; }
HLE_ALIAS(nnosFinalizeMutex, nn_os_FinalizeMutex);
HLE_FUNC(_ZN2nn2os9LockMutexEPNS0_9MutexTypeE, void, nn_os_LockMutex, (void* m)) { MutexLock(m); }
HLE_ALIAS(nnosLockMutex, nn_os_LockMutex);
HLE_FUNC(_ZN2nn2os12TryLockMutexEPNS0_9MutexTypeE, bool, nn_os_TryLockMutex, (void* m)) { return MutexTryLock(m); }
HLE_FUNC(_ZN2nn2os11UnlockMutexEPNS0_9MutexTypeE, void, nn_os_UnlockMutex, (void* m)) { MutexUnlock(m); }
HLE_ALIAS(nnosUnlockMutex, nn_os_UnlockMutex);
HLE_FUNC(_ZN2nn2os28IsMutexLockedByCurrentThreadEPKNS0_9MutexTypeE, bool, nn_os_IsMutexLockedByCurrentThread,
         (void* m)) {
    return MutexView{m}.owner()->load() == CurrentThread();
}

// ===== ConditionVariable =========================================================================
HLE_FUNC(_ZN2nn2os27InitializeConditionVariableEPNS0_21ConditionVariableTypeE, void, nn_os_InitializeCV,
         (void* c)) {
    FieldAt<uint32_t>(c, 4) = 0;
    FieldAt<uint8_t>(c, 0) = 1;
}
HLE_FUNC(_ZN2nn2os25FinalizeConditionVariableEPNS0_21ConditionVariableTypeE, void, nn_os_FinalizeCV, (void* c)) {
    FieldAt<uint8_t>(c, 0) = 0;
}
HLE_FUNC(_ZN2nn2os23SignalConditionVariableEPNS0_21ConditionVariableTypeE, void, nn_os_SignalCV, (void* c)) {
    CondVar::Signal(AtomicAt<uint32_t>(c, 4));
}
HLE_FUNC(_ZN2nn2os26BroadcastConditionVariableEPNS0_21ConditionVariableTypeE, void, nn_os_BroadcastCV, (void* c)) {
    CondVar::Broadcast(AtomicAt<uint32_t>(c, 4));
}

static bool CvWait(void* c, void* m, int64_t timeout_ns) {
    MutexView v{m};
    int nest = v.nest();
    GuestThread* self = v.owner()->load();
    v.nest() = 0;
    v.owner()->store(nullptr);
    bool ok = CondVar::Wait(AtomicAt<uint32_t>(c, 4), v.cs(), timeout_ns);
    v.owner()->store(self);
    v.nest() = nest;
    return ok;
}
HLE_FUNC(_ZN2nn2os21WaitConditionVariableEPNS0_21ConditionVariableTypeEPNS0_9MutexTypeE, void, nn_os_WaitCV,
         (void* c, void* m)) {
    CvWait(c, m, -1);
}
// Returns nn::os::ConditionVariableStatus: 0 = Timeout, 1 = NoTimeout
HLE_FUNC(_ZN2nn2os26TimedWaitConditionVariableEPNS0_21ConditionVariableTypeEPNS0_9MutexTypeENS_8TimeSpanE, int,
         nn_os_TimedWaitCV, (void* c, void* m, int64_t ns)) {
    return CvWait(c, m, ns < 0 ? 0 : ns) ? 1 : 0;
}

// ===== Event =====================================================================================
HLE_FUNC(_ZN2nn2os15InitializeEventEPNS0_9EventTypeEbNS0_14EventClearModeE, void, nn_os_InitializeEvent,
         (void* e, bool signaled, int clear_mode)) {
    memset(e, 0, 0x28);
    FieldAt<void*>(e, 0) = e;
    FieldAt<void*>(e, 8) = e;
    FieldAt<bool>(e, 0x10) = signaled;
    FieldAt<bool>(e, 0x11) = signaled;
    FieldAt<uint8_t>(e, 0x12) = uint8_t(clear_mode);
    FieldAt<uint8_t>(e, 0x13) = 1;
}
HLE_ALIAS(nnosInitializeEvent, nn_os_InitializeEvent);
HLE_FUNC(_ZN2nn2os13FinalizeEventEPNS0_9EventTypeE, void, nn_os_FinalizeEvent, (void* e)) {
    FieldAt<uint8_t>(e, 0x13) = 0;
}
HLE_ALIAS(nnosFinalizeEvent, nn_os_FinalizeEvent);

HLE_FUNC(_ZN2nn2os11SignalEventEPNS0_9EventTypeE, void, nn_os_SignalEvent, (void* e)) {
    EventView v{e};
    CritSec::Lock(v.cs());
    if (!v.signaled()) {
        v.signaled() = true;
        v.counter()++;
        if (v.clear_mode() == 1)
            CondVar::Signal(v.cv());
        else
            CondVar::Broadcast(v.cv());
    }
    CritSec::Unlock(v.cs());
    BumpWaitEpoch();
}
HLE_ALIAS(nnosSignalEvent, nn_os_SignalEvent);

HLE_FUNC(_ZN2nn2os9WaitEventEPNS0_9EventTypeE, void, nn_os_WaitEvent, (void* e)) { EventWait(e, -1); }
HLE_ALIAS(nnosWaitEvent, nn_os_WaitEvent);
HLE_FUNC(_ZN2nn2os12TryWaitEventEPNS0_9EventTypeE, bool, nn_os_TryWaitEvent, (void* e)) { return EventWait(e, 0); }
HLE_FUNC(_ZN2nn2os14TimedWaitEventEPNS0_9EventTypeENS_8TimeSpanE, bool, nn_os_TimedWaitEvent,
         (void* e, int64_t ns)) {
    return EventWait(e, ns < 0 ? 0 : ns);
}
HLE_FUNC(_ZN2nn2os10ClearEventEPNS0_9EventTypeE, void, nn_os_ClearEvent, (void* e)) {
    EventView v{e};
    CritSec::Lock(v.cs());
    v.signaled() = false;
    CritSec::Unlock(v.cs());
}

// ===== Semaphore =================================================================================
HLE_FUNC(_ZN2nn2os19InitializeSemaphoreEPNS0_13SemaphoreTypeEii, void, nn_os_InitializeSemaphore,
         (void* s, int initial, int max)) {
    memset(s, 0, 0x28);
    FieldAt<void*>(s, 0) = s;
    FieldAt<void*>(s, 8) = s;
    FieldAt<uint8_t>(s, 0x10) = 1;
    FieldAt<int>(s, 0x14) = initial;
    FieldAt<int>(s, 0x18) = max;
}
HLE_FUNC(_ZN2nn2os17FinalizeSemaphoreEPNS0_13SemaphoreTypeE, void, nn_os_FinalizeSemaphore, (void* s)) {
    FieldAt<uint8_t>(s, 0x10) = 0;
}
HLE_FUNC(_ZN2nn2os16AcquireSemaphoreEPNS0_13SemaphoreTypeE, void, nn_os_AcquireSemaphore, (void* s)) {
    SemAcquire(s, -1);
}
HLE_FUNC(_ZN2nn2os19TryAcquireSemaphoreEPNS0_13SemaphoreTypeE, bool, nn_os_TryAcquireSemaphore, (void* s)) {
    return SemAcquire(s, 0);
}
HLE_FUNC(_ZN2nn2os21TimedAcquireSemaphoreEPNS0_13SemaphoreTypeENS_8TimeSpanE, bool, nn_os_TimedAcquireSemaphore,
         (void* s, int64_t ns)) {
    return SemAcquire(s, ns < 0 ? 0 : ns);
}
static void SemRelease(void* s, int n) {
    SemView v{s};
    CritSec::Lock(v.cs());
    v.count() = std::min(v.count() + n, v.max());
    CondVar::Broadcast(v.cv());
    CritSec::Unlock(v.cs());
    BumpWaitEpoch();
}
HLE_FUNC(_ZN2nn2os16ReleaseSemaphoreEPNS0_13SemaphoreTypeE, void, nn_os_ReleaseSemaphore, (void* s)) {
    SemRelease(s, 1);
}
HLE_FUNC(_ZN2nn2os16ReleaseSemaphoreEPNS0_13SemaphoreTypeEi, void, nn_os_ReleaseSemaphoreN, (void* s, int n)) {
    SemRelease(s, n);
}
HLE_FUNC(_ZN2nn2os24GetCurrentSemaphoreCountEPKNS0_13SemaphoreTypeE, int, nn_os_GetCurrentSemaphoreCount,
         (void* s)) {
    return SemView{s}.count();
}

// ===== ReaderWriterLock ==========================================================================
HLE_FUNC(_ZN2nn2os26InitializeReaderWriterLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_InitializeRwLock,
         (void* l)) {
    SetHostObj(l, new RwLock);
}
HLE_FUNC(_ZN2nn2os24FinalizeReaderWriterLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_FinalizeRwLock, (void* l)) {
    delete HostObj<RwLock>(l);
    SetHostObj<RwLock>(l, nullptr);
}
static bool RwRead(void* l, bool block) {
    RwLock* r = HostObj<RwLock>(l);
    GuestThread* self = CurrentThread();
    std::unique_lock<std::mutex> lk(r->m);
    if (r->writer == self) {  // reader inside own write lock is allowed
        r->readers++;
        return true;
    }
    if (!block && r->writer) return false;
    r->cv.wait(lk, [&] { return r->writer == nullptr; });
    r->readers++;
    return true;
}
static bool RwWrite(void* l, bool block) {
    RwLock* r = HostObj<RwLock>(l);
    GuestThread* self = CurrentThread();
    std::unique_lock<std::mutex> lk(r->m);
    if (r->writer == self) {
        r->write_nest++;
        return true;
    }
    if (!block && (r->writer || r->readers)) return false;
    r->cv.wait(lk, [&] { return r->writer == nullptr && r->readers == 0; });
    r->writer = self;
    r->write_nest = 1;
    return true;
}
HLE_FUNC(_ZN2nn2os15AcquireReadLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_AcquireReadLock, (void* l)) {
    RwRead(l, true);
}
HLE_FUNC(_ZN2nn2os18TryAcquireReadLockEPNS0_20ReaderWriterLockTypeE, bool, nn_os_TryAcquireReadLock, (void* l)) {
    return RwRead(l, false);
}
HLE_FUNC(_ZN2nn2os15ReleaseReadLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_ReleaseReadLock, (void* l)) {
    RwLock* r = HostObj<RwLock>(l);
    {
        std::lock_guard<std::mutex> lk(r->m);
        r->readers--;
    }
    r->cv.notify_all();
}
HLE_FUNC(_ZN2nn2os16AcquireWriteLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_AcquireWriteLock, (void* l)) {
    RwWrite(l, true);
}
HLE_FUNC(_ZN2nn2os19TryAcquireWriteLockEPNS0_20ReaderWriterLockTypeE, bool, nn_os_TryAcquireWriteLock, (void* l)) {
    return RwWrite(l, false);
}
HLE_FUNC(_ZN2nn2os16ReleaseWriteLockEPNS0_20ReaderWriterLockTypeE, void, nn_os_ReleaseWriteLock, (void* l)) {
    RwLock* r = HostObj<RwLock>(l);
    {
        std::lock_guard<std::mutex> lk(r->m);
        if (--r->write_nest == 0) r->writer = nullptr;
    }
    r->cv.notify_all();
}
HLE_FUNC(_ZN2nn2os14IsReadLockHeldEPKNS0_20ReaderWriterLockTypeE, bool, nn_os_IsReadLockHeld, (void* l)) {
    RwLock* r = HostObj<RwLock>(l);
    std::lock_guard<std::mutex> lk(r->m);
    return r->readers > 0;
}
HLE_FUNC(_ZN2nn2os30IsWriteLockHeldByCurrentThreadEPKNS0_20ReaderWriterLockTypeE, bool,
         nn_os_IsWriteLockHeldByCurrentThread, (void* l)) {
    RwLock* r = HostObj<RwLock>(l);
    std::lock_guard<std::mutex> lk(r->m);
    return r->writer == CurrentThread();
}

// ===== MessageQueue ==============================================================================
HLE_FUNC(_ZN2nn2os22InitializeMessageQueueEPNS0_16MessageQueueTypeEPmm, void, nn_os_InitializeMessageQueue,
         (void* q, uintptr_t* buffer, size_t count)) {
    (void)buffer;
    auto* mq = new MessageQueue;
    mq->capacity = count;
    SetHostObj(q, mq);
}
HLE_FUNC(_ZN2nn2os20FinalizeMessageQueueEPNS0_16MessageQueueTypeE, void, nn_os_FinalizeMessageQueue, (void* q)) {
    delete HostObj<MessageQueue>(q);
    SetHostObj<MessageQueue>(q, nullptr);
}
static bool MqPut(void* q, uintptr_t v, bool front, bool block) {
    MessageQueue* mq = HostObj<MessageQueue>(q);
    std::unique_lock<std::mutex> lk(mq->m);
    if (!block && mq->items.size() >= mq->capacity) return false;
    mq->not_full.wait(lk, [&] { return mq->items.size() < mq->capacity; });
    if (front)
        mq->items.push_front(v);
    else
        mq->items.push_back(v);
    lk.unlock();
    mq->not_empty.notify_one();
    BumpWaitEpoch();
    return true;
}
static bool MqGet(uintptr_t* out, void* q, bool remove, bool block) {
    MessageQueue* mq = HostObj<MessageQueue>(q);
    std::unique_lock<std::mutex> lk(mq->m);
    if (!block && mq->items.empty()) return false;
    mq->not_empty.wait(lk, [&] { return !mq->items.empty(); });
    *out = mq->items.front();
    if (remove) {
        mq->items.pop_front();
        lk.unlock();
        mq->not_full.notify_one();
    }
    return true;
}
HLE_FUNC(_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm, void, nn_os_SendMessageQueue,
         (void* q, uintptr_t v)) {
    MqPut(q, v, false, true);
}
HLE_FUNC(_ZN2nn2os19TrySendMessageQueueEPNS0_16MessageQueueTypeEm, bool, nn_os_TrySendMessageQueue,
         (void* q, uintptr_t v)) {
    return MqPut(q, v, false, false);
}
HLE_FUNC(_ZN2nn2os15JamMessageQueueEPNS0_16MessageQueueTypeEm, void, nn_os_JamMessageQueue, (void* q, uintptr_t v)) {
    MqPut(q, v, true, true);
}
HLE_FUNC(_ZN2nn2os18TryJamMessageQueueEPNS0_16MessageQueueTypeEm, bool, nn_os_TryJamMessageQueue,
         (void* q, uintptr_t v)) {
    return MqPut(q, v, true, false);
}
HLE_FUNC(_ZN2nn2os19ReceiveMessageQueueEPmPNS0_16MessageQueueTypeE, void, nn_os_ReceiveMessageQueue,
         (uintptr_t * out, void* q)) {
    MqGet(out, q, true, true);
}
HLE_FUNC(_ZN2nn2os22TryReceiveMessageQueueEPmPNS0_16MessageQueueTypeE, bool, nn_os_TryReceiveMessageQueue,
         (uintptr_t * out, void* q)) {
    return MqGet(out, q, true, false);
}
HLE_FUNC(_ZN2nn2os16PeekMessageQueueEPmPKNS0_16MessageQueueTypeE, void, nn_os_PeekMessageQueue,
         (uintptr_t * out, void* q)) {
    MqGet(out, q, false, true);
}
HLE_FUNC(_ZN2nn2os19TryPeekMessageQueueEPmPKNS0_16MessageQueueTypeE, bool, nn_os_TryPeekMessageQueue,
         (uintptr_t * out, void* q)) {
    return MqGet(out, q, false, false);
}

// ===== SystemEvent ===============================================================================
static bool SysWait(void* se, int64_t timeout_ns) {
    HostSystemEvent* h = SysEvt(se);
    std::unique_lock<std::mutex> lk(h->m);
    if (timeout_ns < 0)
        h->cv.wait(lk, [&] { return h->signaled; });
    else if (!h->cv.wait_for(lk, std::chrono::nanoseconds(timeout_ns), [&] { return h->signaled; }))
        return false;
    if (h->auto_clear) h->signaled = false;
    return true;
}
HLE_FUNC(_ZN2nn2os15WaitSystemEventEPNS0_15SystemEventTypeE, void, nn_os_WaitSystemEvent, (void* se)) {
    SysWait(se, -1);
}
HLE_FUNC(_ZN2nn2os18TryWaitSystemEventEPNS0_15SystemEventTypeE, bool, nn_os_TryWaitSystemEvent, (void* se)) {
    return SysWait(se, 0);
}
HLE_FUNC(_ZN2nn2os20TimedWaitSystemEventEPNS0_15SystemEventTypeENS_8TimeSpanE, bool, nn_os_TimedWaitSystemEvent,
         (void* se, int64_t ns)) {
    return SysWait(se, ns < 0 ? 0 : ns);
}
HLE_FUNC(_ZN2nn2os16ClearSystemEventEPNS0_15SystemEventTypeE, void, nn_os_ClearSystemEvent, (void* se)) {
    HostSystemEvent* h = SysEvt(se);
    std::lock_guard<std::mutex> lk(h->m);
    h->signaled = false;
}
HLE_FUNC(_ZN2nn2os18DestroySystemEventEPNS0_15SystemEventTypeE, void, nn_os_DestroySystemEvent, (void* se)) {
    delete HostObj<HostSystemEvent>(se);
    SetHostObj<HostSystemEvent>(se, nullptr);
}

// ===== MultiWait (event holders only — the only kind this game links) ============================
HLE_FUNC(_ZN2nn2os19InitializeMultiWaitEPNS0_13MultiWaitTypeE, void, nn_os_InitializeMultiWait, (void* mw)) {
    SetHostObj(mw, new MultiWait);
}
HLE_FUNC(_ZN2nn2os17FinalizeMultiWaitEPNS0_13MultiWaitTypeE, void, nn_os_FinalizeMultiWait, (void* mw)) {
    delete HostObj<MultiWait>(mw);
    SetHostObj<MultiWait>(mw, nullptr);
}
// Holder layout (ours): +0 EventType*  +8 MultiWait* (owner)
HLE_FUNC(_ZN2nn2os25InitializeMultiWaitHolderEPNS0_19MultiWaitHolderTypeEPNS0_9EventTypeE, void,
         nn_os_InitializeMultiWaitHolder, (void* h, void* event)) {
    FieldAt<void*>(h, 0) = event;
    FieldAt<void*>(h, 8) = nullptr;
}
HLE_FUNC(_ZN2nn2os23FinalizeMultiWaitHolderEPNS0_19MultiWaitHolderTypeE, void, nn_os_FinalizeMultiWaitHolder,
         (void* h)) {
    FieldAt<void*>(h, 0) = nullptr;
}
HLE_FUNC(_ZN2nn2os19LinkMultiWaitHolderEPNS0_13MultiWaitTypeEPNS0_19MultiWaitHolderTypeE, void,
         nn_os_LinkMultiWaitHolder, (void* mw, void* h)) {
    HostObj<MultiWait>(mw)->holders.push_back(h);
    FieldAt<void*>(h, 8) = mw;
}
HLE_FUNC(_ZN2nn2os24UnlinkAllMultiWaitHolderEPNS0_13MultiWaitTypeE, void, nn_os_UnlinkAllMultiWaitHolder,
         (void* mw)) {
    MultiWait* m = HostObj<MultiWait>(mw);
    for (void* h : m->holders) FieldAt<void*>(h, 8) = nullptr;
    m->holders.clear();
}
HLE_FUNC(_ZN2nn2os7WaitAnyEPNS0_13MultiWaitTypeE, void*, nn_os_WaitAny, (void* mw)) {
    MultiWait* m = HostObj<MultiWait>(mw);
    for (;;) {
        uint32_t epoch = WaitEpoch();
        for (void* h : m->holders) {
            void* ev = FieldAt<void*>(h, 0);
            if (ev && IsEventSignaled(ev)) {
                EventView v{ev};
                if (v.clear_mode() == 1) {
                    CritSec::Lock(v.cs());
                    v.signaled() = false;
                    CritSec::Unlock(v.cs());
                }
                return h;
            }
        }
        WaitForEpochChange(epoch, 50'000'000);  // re-scan at least every 50 ms
    }
}
