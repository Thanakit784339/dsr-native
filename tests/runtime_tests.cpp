// Self-contained tests for the loader and HLE runtime (no game files needed).
// Portable tests run everywhere; ABI tests (x18 handling, thunks, setjmp) run only on arm64.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "arch/arch.h"
#include "hle/nvn.h"
#include "hle/nvn/backend.h"
#include "hle/registry.h"
#include "loader/dynamic_thunks.h"
#include "loader/nso.h"
#include "loader/sha256.h"

static int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

template <typename F>
F Hle(const char* mangled) {
    void* p = dsr::hle::FindFunction(mangled);
    if (!p) {
        fprintf(stderr, "  missing HLE function %s\n", mangled);
        exit(1);
    }
    return reinterpret_cast<F>(p);
}

// ---- loader ------------------------------------------------------------------------------------
static void TestLz4() {
    const uint8_t lit[] = {0x50, 'h', 'e', 'l', 'l', 'o'};
    uint8_t out[16] = {};
    CHECK(dsr::loader::Lz4DecompressBlock(lit, sizeof(lit), out, 5));
    CHECK(memcmp(out, "hello", 5) == 0);
    // "abc" + match(offset 3, length 6) -> "abcabcabc" (overlapping copy)
    const uint8_t match[] = {0x32, 'a', 'b', 'c', 0x03, 0x00};
    CHECK(dsr::loader::Lz4DecompressBlock(match, sizeof(match), out, 9));
    CHECK(memcmp(out, "abcabcabc", 9) == 0);
    // wrong output size must be rejected
    CHECK(!dsr::loader::Lz4DecompressBlock(match, sizeof(match), out, 8));
}

static void TestSha256() {
    uint8_t d[32];
    dsr::loader::Sha256(reinterpret_cast<const uint8_t*>("abc"), 3, d);
    const uint8_t expect[4] = {0xba, 0x78, 0x16, 0xbf};
    CHECK(memcmp(d, expect, 4) == 0);
    std::vector<uint8_t> big(1000, 'a');
    dsr::loader::Sha256(big.data(), big.size(), d);  // exercises the two-block tail path
    const uint8_t expect2[4] = {0x41, 0xed, 0xec, 0xe4};
    CHECK(memcmp(d, expect2, 4) == 0);
}

// ---- nn::mem::StandardAllocator ----------------------------------------------------------------
static void TestStandardAllocator() {
    using Ctor = void (*)(void*);
    using Init = void (*)(void*, void*, size_t);
    using Alloc = void* (*)(void*, size_t, size_t);
    using Free = void (*)(void*, void*);
    using FreeSize = size_t (*)(void*);
    auto ctor = Hle<Ctor>("_ZN2nn3mem17StandardAllocatorC1Ev");
    auto init = Hle<Init>("_ZN2nn3mem17StandardAllocator10InitializeEPvm");
    auto alloc = Hle<Alloc>("_ZN2nn3mem17StandardAllocator8AllocateEmm");
    auto free_ = Hle<Free>("_ZN2nn3mem17StandardAllocator4FreeEPv");
    auto free_size = Hle<FreeSize>("_ZNK2nn3mem17StandardAllocator16GetTotalFreeSizeEv");

    alignas(16) uint8_t obj[64];
    const size_t region_size = 8 << 20;
    auto* region = static_cast<uint8_t*>(malloc(region_size));
    ctor(obj);
    init(obj, region, region_size);
    const size_t initial_free = free_size(obj);

    std::mt19937 rng(1234);
    struct A { uint8_t* p; size_t n; uint8_t tag; };
    std::vector<A> live;
    for (int i = 0; i < 20000; i++) {
        if (live.empty() || rng() % 3) {
            size_t n = 1 + rng() % 4096;
            size_t align = size_t(1) << (4 + rng() % 9);  // 16..4096
            auto* p = static_cast<uint8_t*>(alloc(obj, n, align));
            if (!p) continue;
            CHECK((reinterpret_cast<uintptr_t>(p) & (align - 1)) == 0);
            CHECK(p >= region && p + n <= region + region_size);
            uint8_t tag = uint8_t(rng());
            memset(p, tag, n);
            live.push_back({p, n, tag});
        } else {
            size_t k = rng() % live.size();
            A a = live[k];
            for (size_t j = 0; j < a.n; j++)
                if (a.p[j] != a.tag) {
                    CHECK(!"allocation was overwritten");
                    break;
                }
            free_(obj, a.p);
            live[k] = live.back();
            live.pop_back();
        }
    }
    for (const A& a : live) free_(obj, a.p);
    CHECK(free_size(obj) == initial_free);  // everything coalesced back
    free(region);
}

// ---- nn::os sync primitives (operate on SDK struct layouts) ---------------------------------------
static void TestOsSync() {
    using MutexInit = void (*)(void*, bool, int);
    using MutexOp = void (*)(void*);
    auto minit = Hle<MutexInit>("_ZN2nn2os15InitializeMutexEPNS0_9MutexTypeEbi");
    auto lock = Hle<MutexOp>("_ZN2nn2os9LockMutexEPNS0_9MutexTypeE");
    auto unlock = Hle<MutexOp>("_ZN2nn2os11UnlockMutexEPNS0_9MutexTypeE");

    alignas(8) uint8_t mutex[0x20];
    minit(mutex, true, 0);
    lock(mutex);
    lock(mutex);  // recursive
    unlock(mutex);
    unlock(mutex);

    long counter = 0;
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; t++)
        ts.emplace_back([&] {
            for (int i = 0; i < 20000; i++) {
                lock(mutex);
                counter++;
                unlock(mutex);
            }
        });
    for (auto& t : ts) t.join();
    CHECK(counter == 80000);

    // Event: auto-clear wakes exactly one waiter per signal.
    using EventInit = void (*)(void*, bool, int);
    using EventOp = void (*)(void*);
    using EventTimed = bool (*)(void*, int64_t);
    auto einit = Hle<EventInit>("_ZN2nn2os15InitializeEventEPNS0_9EventTypeEbNS0_14EventClearModeE");
    auto signal = Hle<EventOp>("_ZN2nn2os11SignalEventEPNS0_9EventTypeE");
    auto timed = Hle<EventTimed>("_ZN2nn2os14TimedWaitEventEPNS0_9EventTypeENS_8TimeSpanE");
    alignas(8) uint8_t ev[0x28];
    einit(ev, false, 1);
    CHECK(!timed(ev, 1'000'000));
    std::thread waker([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        signal(ev);
    });
    CHECK(timed(ev, 2'000'000'000));
    CHECK(!timed(ev, 1'000'000));  // auto-cleared
    waker.join();

    // Semaphore producer/consumer.
    using SemInit = void (*)(void*, int, int);
    using SemOp = void (*)(void*);
    auto sinit = Hle<SemInit>("_ZN2nn2os19InitializeSemaphoreEPNS0_13SemaphoreTypeEii");
    auto acquire = Hle<SemOp>("_ZN2nn2os16AcquireSemaphoreEPNS0_13SemaphoreTypeE");
    auto release = Hle<SemOp>("_ZN2nn2os16ReleaseSemaphoreEPNS0_13SemaphoreTypeE");
    alignas(8) uint8_t sem[0x28];
    sinit(sem, 0, 1000);
    std::atomic<int> got{0};
    std::thread consumer([&] {
        for (int i = 0; i < 500; i++) {
            acquire(sem);
            got++;
        }
    });
    for (int i = 0; i < 500; i++) release(sem);
    consumer.join();
    CHECK(got == 500);
}

// ---- null NVN driver ---------------------------------------------------------------------------
template <typename F>
F Nvn(const char* name) {
    void* p = dsr::hle::nvn::FindNvnImpl(name);
    if (!p) {
        fprintf(stderr, "  missing NVN proc %s\n", name);
        exit(1);
    }
    return reinterpret_cast<F>(p);
}

static void TestNullNvn() {
    // Version check the game performs before anything else.
    auto get_int = Nvn<void (*)(const void*, int, int*)>("nvnDeviceGetInteger");
    int major = 0, minor = 0, ubo = 0, unknown = 1234;
    get_int(nullptr, 0, &major);
    get_int(nullptr, 1, &minor);
    get_int(nullptr, 4, &ubo);
    get_int(nullptr, 9999, &unknown);
    CHECK(major == 53 && minor > 104 && ubo == 256);
    CHECK(unknown == 1234);  // unknown queries keep the caller's default

    // Memory pool -> buffer: mapped pointer and GPU address must follow pool storage + offset.
    alignas(16) static uint8_t storage[0x4000];
    uint8_t pool_builder[64], pool[256], buf_builder[64], buf[64];
    Nvn<void (*)(void*)>("nvnMemoryPoolBuilderSetDefaults")(pool_builder);
    Nvn<void (*)(void*, void*, size_t)>("nvnMemoryPoolBuilderSetStorage")(pool_builder, storage, sizeof(storage));
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnMemoryPoolInitialize")(pool, pool_builder));
    CHECK(Nvn<void* (*)(const void*)>("nvnMemoryPoolMap")(pool) == storage);
    Nvn<void (*)(void*)>("nvnBufferBuilderSetDefaults")(buf_builder);
    Nvn<void (*)(void*, const void*, int64_t, size_t)>("nvnBufferBuilderSetStorage")(buf_builder, pool, 0x100, 0x200);
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnBufferInitialize")(buf, buf_builder));
    CHECK(Nvn<void* (*)(const void*)>("nvnBufferMap")(buf) == storage + 0x100);
    CHECK(Nvn<uint64_t (*)(const void*)>("nvnBufferGetAddress")(buf) == reinterpret_cast<uint64_t>(storage) + 0x100);

    // Texture builder -> texture: dimensions read back, storage estimate covers the data.
    uint8_t tb[128], tex[192];
    Nvn<void (*)(void*)>("nvnTextureBuilderSetDefaults")(tb);
    Nvn<void (*)(void*, int, int)>("nvnTextureBuilderSetSize2D")(tb, 1280, 720);
    Nvn<void (*)(void*, int)>("nvnTextureBuilderSetFormat")(tb, 37);  // RGBA8
    Nvn<void (*)(void*, int)>("nvnTextureBuilderSetLevels")(tb, 1);
    size_t sz = Nvn<size_t (*)(const void*)>("nvnTextureBuilderGetStorageSize")(tb);
    CHECK(sz >= size_t(1280) * 720 * 4);
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnTextureInitialize")(tex, tb));
    CHECK(Nvn<int (*)(const void*)>("nvnTextureGetWidth")(tex) == 1280);
    CHECK(Nvn<int (*)(const void*)>("nvnTextureGetHeight")(tex) == 720);

    // Prefix fallback covers command-buffer setters; creation calls report success.
    CHECK(dsr::hle::nvn::FindNvnImpl("nvnCommandBufferSetViewport") != nullptr);
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnSamplerPoolInitialize")(pool, nullptr));
}

// Command recording: what the game records must reach the backend in order, with textures resolved.
namespace {
struct RecordingBackend final : dsr::hle::nvn::Backend {
    std::vector<dsr::hle::nvn::CmdList> executed;
    int presents = 0;
    dsr::hle::nvn::TexRef last_present;
    void Execute(const dsr::hle::nvn::CmdList& c) override { executed.push_back(c); }
    void Present(const dsr::hle::nvn::TexRef& t) override {
        presents++;
        last_present = t;
    }
    const char* Name() const override { return "test"; }
};
}  // namespace

static void TestNvnCommands() {
    using namespace dsr::hle::nvn;
    RecordingBackend be;
    SetBackendForTesting(&be);

    alignas(16) static uint8_t storage[0x400000];
    uint8_t pool_builder[64], pool[256], tb[128], color[192], depth[192], cb[256], wb[64], window[128];
    Nvn<void (*)(void*)>("nvnMemoryPoolBuilderSetDefaults")(pool_builder);
    Nvn<void (*)(void*, void*, size_t)>("nvnMemoryPoolBuilderSetStorage")(pool_builder, storage, sizeof(storage));
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnMemoryPoolInitialize")(pool, pool_builder));
    auto make_tex = [&](void* t, int w, int h, int fmt, int64_t off) {
        Nvn<void (*)(void*)>("nvnTextureBuilderSetDefaults")(tb);
        Nvn<void (*)(void*, int, int)>("nvnTextureBuilderSetSize2D")(tb, w, h);
        Nvn<void (*)(void*, int)>("nvnTextureBuilderSetFormat")(tb, fmt);
        Nvn<void (*)(void*, const void*, int64_t)>("nvnTextureBuilderSetStorage")(tb, pool, off);
        CHECK(Nvn<bool (*)(void*, const void*)>("nvnTextureInitialize")(t, tb));
    };
    make_tex(color, 1280, 720, 37, 0);
    make_tex(depth, 1280, 720, 53, 0x200000);

    const void* colors[1] = {color};
    const float red[4] = {1, 0, 0, 1};
    Nvn<void (*)(void*)>("nvnCommandBufferBeginRecording")(cb);
    CHECK(Nvn<bool (*)(const void*)>("nvnCommandBufferIsRecording")(cb));
    Nvn<void (*)(void*, int, const void* const*, const void* const*, const void*, const void*)>(
        "nvnCommandBufferSetRenderTargets")(cb, 1, colors, nullptr, depth, nullptr);
    Nvn<void (*)(void*, int, const float*, int)>("nvnCommandBufferClearColor")(cb, 0, red, 0xF);
    Nvn<void (*)(void*, float, bool, int, int)>("nvnCommandBufferClearDepthStencil")(cb, 1.0f, true, 0, 0xFF);
    Nvn<void (*)(void*)>("nvnCommandBufferDrawArrays")(cb);
    // The game only submits buffers that report command memory use.
    CHECK(Nvn<size_t (*)(const void*)>("nvnCommandBufferGetCommandMemoryUsed")(cb) > 0);
    uint64_t h = Nvn<uint64_t (*)(void*)>("nvnCommandBufferEndRecording")(cb);
    CHECK(h != 0);
    CHECK(!Nvn<bool (*)(const void*)>("nvnCommandBufferIsRecording")(cb));

    // A second buffer that inlines the first one.
    uint8_t cb2[256];
    Nvn<void (*)(void*)>("nvnCommandBufferBeginRecording")(cb2);
    Nvn<void (*)(void*, int, const uint64_t*)>("nvnCommandBufferCallCommands")(cb2, 1, &h);
    uint64_t h2 = Nvn<uint64_t (*)(void*)>("nvnCommandBufferEndRecording")(cb2);

    uint64_t handles[2] = {h, h2};
    Nvn<void (*)(void*, int, const uint64_t*)>("nvnQueueSubmitCommands")(nullptr, 2, handles);
    CHECK(be.executed.size() == 2);
    if (be.executed.size() == 2) {
        const CmdList& l = be.executed[0];
        CHECK(l.size() == 4);
        CHECK(l.size() == 4 && l[0].type == CmdType::SetRenderTargets && l[0].num_rt == 1);
        if (l.size() == 4 && l[0].rts && l[0].rts->size() == 2) {
            CHECK((*l[0].rts)[0].valid && (*l[0].rts)[0].tex.width == 1280 && (*l[0].rts)[0].tex.format == 37);
            CHECK((*l[0].rts)[0].addr == reinterpret_cast<uint64_t>(storage));
            CHECK((*l[0].rts)[1].addr == reinterpret_cast<uint64_t>(storage) + 0x200000 && (*l[0].rts)[1].tex.format == 53);
        } else {
            CHECK(!"render targets not captured");
        }
        float got[4];
        if (l.size() == 4) memcpy(got, l[1].color, sizeof(got));
        CHECK(l.size() == 4 && l[1].type == CmdType::ClearColor && got[0] == 1.0f && got[1] == 0.0f && l[1].mask == 0xF);
        CHECK(l.size() == 4 && l[2].type == CmdType::ClearDepthStencil && l[2].depth == 1.0f && l[2].depth_mask);
        CHECK(l.size() == 4 && l[3].type == CmdType::Draw);
        CHECK(be.executed[1].size() == 4);  // inlined copy
    }

    // Presenting window texture 0 resolves to the color texture.
    const void* wtex[2] = {color, depth};
    Nvn<void (*)(void*)>("nvnWindowBuilderSetDefaults")(wb);
    Nvn<void (*)(void*, int, const void* const*)>("nvnWindowBuilderSetTextures")(wb, 2, wtex);
    CHECK(Nvn<bool (*)(void*, const void*)>("nvnWindowInitialize")(window, wb));
    Nvn<void (*)(void*, void*, int)>("nvnQueuePresentTexture")(nullptr, window, 0);
    CHECK(be.presents == 1 && be.last_present.valid && be.last_present.addr == reinterpret_cast<uint64_t>(storage));

    SetBackendForTesting(nullptr);
}

// ---- arm64 ABI glue ----------------------------------------------------------------------------
#if defined(__aarch64__)
extern "C" {
uint64_t test_guest_call_import(void* thunk, uint64_t arg);
uint64_t test_guest_sum7(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
double test_guest_fp(double, double);
int dsr_setjmp(void*) __attribute__((returns_twice));
}

static uint64_t g_seen_x18 = 0;
extern "C" uint64_t test_host_import(uint64_t a) {
    uint64_t v;
    asm volatile("mov %0, x18" : "=r"(v));
    g_seen_x18 = v;
    return a * 2;
}

static uint64_t ReadX18() {
    uint64_t v;
    asm volatile("mov %0, x18" : "=r"(v));
    return v;
}
// Loads a sentinel into x18 immediately before the call. This file is built with -ffixed-x18, so the
// compiler does not touch x18 between this and dsr_guest_call (which records it as the host value).
static void SetX18(uint64_t v) { asm volatile("mov x18, %0" ::"r"(v)); }

static void TestAbi() {
    using dsr::arch::GuestCall;
    constexpr uint64_t kSentinel = 0x5C5C5C5C12345678ull;

    // guest -> import thunk -> host: host must see its own x18 again, and get x18 back afterwards.
    void* thunk = dsr::loader::MakeDynamicThunk(reinterpret_cast<void*>(&test_host_import), 0);
    SetX18(kSentinel);
    uint64_t r = GuestCall<uint64_t>(reinterpret_cast<void*>(&test_guest_call_import), thunk, uint64_t(20));
    uint64_t after = ReadX18();
    CHECK(r == 41);
    CHECK(g_seen_x18 == kSentinel);
    CHECK(after == kSentinel);

    // Unimplemented stub returns 0 (so the guest computes 0 + 1).
    void* stub = dsr::loader::MakeDynamicThunk(dsr::arch::UnimplementedNamedEntry(),
                                               reinterpret_cast<uint64_t>("test::Unimplemented"));
    SetX18(kSentinel);
    r = GuestCall<uint64_t>(reinterpret_cast<void*>(&test_guest_call_import), stub, uint64_t(5));
    after = ReadX18();
    CHECK(r == 1);
    CHECK(after == kSentinel);

    // Argument shifting for 7 integer args (guest clobbers x18), and FP passthrough.
    SetX18(kSentinel);
    r = GuestCall<uint64_t>(reinterpret_cast<void*>(&test_guest_sum7), uint64_t(1), uint64_t(2), uint64_t(3),
                            uint64_t(4), uint64_t(5), uint64_t(6), uint64_t(7));
    after = ReadX18();
    CHECK(r == 28);
    CHECK(after == kSentinel);
    double d = GuestCall<double>(reinterpret_cast<void*>(&test_guest_fp), 1.5, 4.0);
    CHECK(d == 6.0);

    // musl-layout setjmp/longjmp.
    alignas(16) static uint8_t buf[312];
    volatile int stage = 0;
    int v = dsr_setjmp(buf);
    if (v == 0) {
        stage = 1;
        dsr_longjmp(buf, 7);
    }
    CHECK(stage == 1 && v == 7);
}
#endif

int main() {
    struct T { const char* name; void (*fn)(); } tests[] = {
        {"lz4", TestLz4},
        {"sha256", TestSha256},
        {"standard_allocator", TestStandardAllocator},
        {"os_sync", TestOsSync},
        {"null_nvn", TestNullNvn},
        {"nvn_commands", TestNvnCommands},
#if defined(__aarch64__)
        {"abi_arm64", TestAbi},
#endif
    };
    for (const T& t : tests) {
        int before = g_failures;
        t.fn();
        printf("%-20s %s\n", t.name, g_failures == before ? "ok" : "FAILED");
    }
#if !defined(__aarch64__)
    printf("%-20s skipped (not arm64)\n", "abi_arm64");
#endif
    printf("%d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
