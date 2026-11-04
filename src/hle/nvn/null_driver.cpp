// "Null" NVN driver: accepts every NVN call, tracks the object state the game reads back (sizes,
// mapped pointers, GPU addresses, texture dimensions), and renders nothing. It lets the game get past
// graphics initialization so the rest of the runtime can be exercised, and it is the skeleton the
// Vulkan backend will fill in.
//
// Functions not listed here still get a logging stub (see nvn.cpp); stubs of names containing
// "Initialize" or ending in "SetShaders" report success.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "common/log.h"
#include "hle/nvn.h"
#include "hle/nvn/nvn_types.h"
#include "hle/os/os.h"

namespace dsr::hle::nvn {

// ---- formats ------------------------------------------------------------------------------------
FormatInfo GetFormatInfo(int f) {
    if (f >= 1 && f <= 4) return {1, 1, 1};     // R8*
    if (f >= 5 && f <= 9) return {1, 1, 2};     // R16*
    if (f >= 10 && f <= 12) return {1, 1, 4};   // R32*
    if (f >= 13 && f <= 16) return {1, 1, 2};   // RG8*
    if (f >= 17 && f <= 21) return {1, 1, 4};   // RG16*
    if (f >= 22 && f <= 24) return {1, 1, 8};   // RG32*
    if (f >= 25 && f <= 28) return {1, 1, 4};   // RGB8* (padded)
    if (f >= 29 && f <= 33) return {1, 1, 8};   // RGB16*
    if (f >= 34 && f <= 36) return {1, 1, 16};  // RGB32*
    if (f >= 37 && f <= 40) return {1, 1, 4};   // RGBA8*
    if (f >= 41 && f <= 45) return {1, 1, 8};   // RGBA16*
    if (f >= 46 && f <= 48) return {1, 1, 16};  // RGBA32*
    if (f == 49) return {1, 1, 1};              // STENCIL8
    if (f == 50) return {1, 1, 2};              // DEPTH16
    if (f >= 51 && f <= 53) return {1, 1, 4};   // DEPTH24, DEPTH32F, D24S8
    if (f == 54) return {1, 1, 8};              // D32F_S8
    if (f == 55 || f == 56) return {1, 1, 4};   // RGBX8/RGBA8 sRGB
    if (f >= 57 && f <= 60) return {1, 1, 2};   // RGBA4, RGB5, RGB5A1, RGB565
    if (f >= 61 && f <= 64) return {1, 1, 4};   // RGB10A2, R11G11B10F, RGB9E5
    if (f == 65 || f == 66 || f == 69 || f == 70 || f == 73 || f == 74) return {4, 4, 8};  // DXT1, RGTC1
    if (f >= 65 && f <= 80) return {4, 4, 16};  // DXT3/5, RGTC2, BPTC
    return {1, 1, 8};                           // unknown: generous
}

namespace {
constexpr const char* kTag = "nvn.null";

Table<MemoryPoolState> g_pool_builders, g_pools;
Table<BufferState> g_buffer_builders, g_buffers;
Table<TextureState> g_texture_builders, g_textures;
Table<WindowState> g_window_builders, g_windows;
Table<QueueBuilderState> g_queue_builders;
std::atomic<uint64_t> g_command_handle{1};

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Conservative storage estimate: real NVN storage is block-linear (padded), so pad rows and heights.
size_t StorageSize(const TextureState& t) {
    FormatInfo fi = GetFormatInfo(t.format);
    int layers = std::max(1, t.depth);
    if (t.target == kTargetCubemap) layers = std::max(6, layers);
    size_t total = 0;
    for (int l = 0; l < std::max(1, t.levels); l++) {
        int w = std::max(1, t.width >> l), h = std::max(1, t.height >> l);
        size_t bw = (w + fi.block_w - 1) / fi.block_w, bh = (h + fi.block_h - 1) / fi.block_h;
        size_t row = AlignUp(bw * fi.bytes_per_block, 64);
        total += row * AlignUp(bh, 8) * (t.target == kTarget3D ? std::max(1, t.depth >> l) : 1);
    }
    if (t.target != kTarget3D) total *= layers;
    total *= std::max(1, t.samples);
    return AlignUp(std::max<size_t>(total, 512), 512);
}

uintptr_t PoolBase(const void* pool) { return reinterpret_cast<uintptr_t>(g_pools.Copy(pool).storage); }

// ---- device ----------------------------------------------------------------------------------------
// Returns 0 so getters that fall back to it never hand garbage to the game.
uint64_t Nop() { return 0; }
bool True() { return true; }

bool DeviceInitialize(void* device, const void* builder) {
    LOGI(kTag, "nvnDeviceInitialize(%p) — null renderer (no GPU output yet)", device);
    return true;
}

void DeviceGetInteger(const void* device, int pname, int* v) {
    switch (pname) {
        case kApiMajorVersion: *v = 53; break;
        case kApiMinorVersion: *v = 105; break;
        case 2: *v = 14; break;       // uniform buffer bindings per stage
        case 3: *v = 65536; break;    // max uniform buffer size
        case kUniformBufferAlignment: *v = 256; break;
        case 5: *v = 8; break;        // color buffer bindings
        case 6: *v = 16; break;       // vertex buffer bindings
        case 7: *v = 4; break;        // transform feedback bindings
        case 8: *v = 16; break;       // storage buffer bindings per stage
        case 9: *v = 32; break;       // texture bindings per stage
        case 10: *v = 16; break;      // counter alignment
        case 11: case 12: case 13: *v = 4; break;
        case 14: *v = 16; break;      // vertex attributes
        case kTextureDescriptorSize: *v = 32; break;
        case kSamplerDescriptorSize: *v = 32; break;
        case 17: case 18: *v = 256; break;  // reserved texture/sampler descriptors
        case kCommandBufferCommandAlignment: *v = 4; break;
        case kCommandBufferControlAlignment: *v = 8; break;
        case 21: *v = 4096; break;    // min command size
        case 22: *v = 4096; break;    // min control size
        case 23: *v = 128; break;     // scratch scale factor minimum
        case 24: *v = 256; break;     // scratch scale factor recommended
        case 25: *v = 4096; break;    // scratch alignment
        case 26: *v = 128 * 1024; break;  // scratch granularity
        case 27: *v = 16; break;      // max anisotropy
        case 36: *v = 1 << 20; break; // max texture pool size
        case 37: *v = 4096; break;    // max sampler pool size
        case 38: *v = 16; break;      // max viewports
        case 39: *v = 4096; break;    // texture object page alignment
        case 43: *v = 256 * 1024; break;  // L2 size
        case 44: *v = 15; break;      // max texture levels
        case 45: *v = 2048; break;    // max texture layers
        case kLinearTextureStrideAlignment: *v = 32; break;
        case 57: *v = 128; break;     // linear render target stride alignment
        case 58: *v = 4096; break;    // memory pool page size
        case kQueueCommandMemoryDefaultSize: *v = 0x10000; break;
        default:
            // Leave the caller's pre-filled default untouched.
            LOGD(kTag, "nvnDeviceGetInteger: unknown pname %d (left at %d)", pname, *v);
            break;
    }
}

uint64_t DeviceGetTextureHandle(const void*, int texture_id, int sampler_id) {
    return (uint64_t(uint32_t(sampler_id)) << 20) | uint32_t(texture_id);
}
uint64_t DeviceGetImageHandle(const void*, int image_id) { return uint32_t(image_id); }
uint64_t DeviceGetSeparateHandle(const void*, int id) { return uint32_t(id); }
uint64_t DeviceGetTimestampNs(const void*) { return uint64_t(os::NowNs()); }

// ---- memory pools ------------------------------------------------------------------------------------
void PoolBuilderSetDefaults(void* b) { g_pool_builders.Reset(b); }
void PoolBuilderSetFlags(void* b, int flags) { g_pool_builders.Get(b).flags = flags; }
void PoolBuilderSetStorage(void* b, void* mem, size_t size) {
    auto& s = g_pool_builders.Get(b);
    s.storage = static_cast<uint8_t*>(mem);
    s.size = size;
}
bool PoolInitialize(void* pool, const void* b) {
    g_pools.Get(pool) = g_pool_builders.Copy(b);
    return true;
}
void PoolFinalize(void* pool) { g_pools.Erase(pool); }
void* PoolMap(const void* pool) { return g_pools.Copy(pool).storage; }
uint64_t PoolGetBufferAddress(const void* pool) { return PoolBase(pool); }
size_t PoolGetSize(const void* pool) { return g_pools.Copy(pool).size; }
int PoolGetFlags(const void* pool) { return g_pools.Copy(pool).flags; }

// ---- buffers -------------------------------------------------------------------------------------------
void BufferBuilderSetDefaults(void* b) { g_buffer_builders.Reset(b); }
void BufferBuilderSetStorage(void* b, const void* pool, int64_t offset, size_t size) {
    g_buffer_builders.Get(b) = {pool, offset, size};
}
bool BufferInitialize(void* buf, const void* b) {
    g_buffers.Get(buf) = g_buffer_builders.Copy(b);
    return true;
}
void BufferFinalize(void* buf) { g_buffers.Erase(buf); }
void* BufferMap(const void* buf) {
    BufferState s = g_buffers.Copy(buf);
    uintptr_t base = PoolBase(s.pool);
    return base ? reinterpret_cast<void*>(base + s.offset) : nullptr;
}
uint64_t BufferGetAddress(const void* buf) {
    BufferState s = g_buffers.Copy(buf);
    return PoolBase(s.pool) + uint64_t(s.offset);
}
size_t BufferGetSize(const void* buf) { return g_buffers.Copy(buf).size; }

// ---- textures --------------------------------------------------------------------------------------------
void TexBuilderSetDefaults(void* b) { g_texture_builders.Reset(b); }
void TexBuilderSetFlags(void* b, int v) { g_texture_builders.Get(b).flags = v; }
void TexBuilderSetTarget(void* b, int v) { g_texture_builders.Get(b).target = v; }
void TexBuilderSetWidth(void* b, int v) { g_texture_builders.Get(b).width = v; }
void TexBuilderSetHeight(void* b, int v) { g_texture_builders.Get(b).height = v; }
void TexBuilderSetDepth(void* b, int v) { g_texture_builders.Get(b).depth = v; }
void TexBuilderSetSize1D(void* b, int w) {
    auto& t = g_texture_builders.Get(b);
    t.width = w;
    t.height = t.depth = 1;
}
void TexBuilderSetSize2D(void* b, int w, int h) {
    auto& t = g_texture_builders.Get(b);
    t.width = w;
    t.height = h;
    t.depth = 1;
}
void TexBuilderSetSize3D(void* b, int w, int h, int d) {
    auto& t = g_texture_builders.Get(b);
    t.width = w;
    t.height = h;
    t.depth = d;
}
void TexBuilderSetLevels(void* b, int v) { g_texture_builders.Get(b).levels = v; }
void TexBuilderSetFormat(void* b, int v) { g_texture_builders.Get(b).format = v; }
void TexBuilderSetSamples(void* b, int v) { g_texture_builders.Get(b).samples = v; }
void TexBuilderSetStride(void* b, int64_t v) { g_texture_builders.Get(b).stride = int(v); }
void TexBuilderSetStorage(void* b, const void* pool, int64_t offset) {
    auto& t = g_texture_builders.Get(b);
    t.pool = pool;
    t.offset = offset;
}
size_t TexBuilderGetStorageSize(const void* b) { return StorageSize(g_texture_builders.Copy(b)); }
size_t TexBuilderGetStorageAlignment(const void* b) { return 512; }
int TexBuilderGetWidth(const void* b) { return g_texture_builders.Copy(b).width; }
int TexBuilderGetHeight(const void* b) { return g_texture_builders.Copy(b).height; }
int TexBuilderGetDepth(const void* b) { return g_texture_builders.Copy(b).depth; }
int TexBuilderGetLevels(const void* b) { return g_texture_builders.Copy(b).levels; }
int TexBuilderGetFormat(const void* b) { return g_texture_builders.Copy(b).format; }
int TexBuilderGetTarget(const void* b) { return g_texture_builders.Copy(b).target; }

bool TexInitialize(void* tex, const void* b) {
    g_textures.Get(tex) = g_texture_builders.Copy(b);
    return true;
}
void TexFinalize(void* tex) { g_textures.Erase(tex); }
int TexGetWidth(const void* t) { return g_textures.Copy(t).width; }
int TexGetHeight(const void* t) { return g_textures.Copy(t).height; }
int TexGetDepth(const void* t) { return g_textures.Copy(t).depth; }
int TexGetLevels(const void* t) { return g_textures.Copy(t).levels; }
int TexGetFormat(const void* t) { return g_textures.Copy(t).format; }
int TexGetSamples(const void* t) { return g_textures.Copy(t).samples; }
int TexGetTarget(const void* t) { return g_textures.Copy(t).target; }
size_t TexGetStorageSize(const void* t) { return StorageSize(g_textures.Copy(t)); }
uint64_t TexGetTextureAddress(const void* t) {
    TextureState s = g_textures.Copy(t);
    return PoolBase(s.pool) + uint64_t(s.offset);
}

// ---- queues ------------------------------------------------------------------------------------------------
void QueueBuilderSetDefaults(void* b) { g_queue_builders.Reset(b); }
void QueueBuilderSetCommandMemorySize(void* b, size_t s) { g_queue_builders.Get(b).command_memory = s; }
void QueueBuilderSetComputeMemorySize(void* b, size_t s) { g_queue_builders.Get(b).compute_memory = s; }
size_t QueueBuilderGetQueueMemorySize(const void* b) {
    QueueBuilderState s = g_queue_builders.Copy(b);
    return AlignUp(s.command_memory + s.compute_memory + 0x10000, 0x1000);
}
bool QueueInitialize(void* q, const void* b) {
    LOGI(kTag, "nvnQueueInitialize(%p)", q);
    return true;
}

// Frame pacing for the null renderer: Dark Souls Remastered runs at 30 fps on Switch.
void QueuePresentTexture(void* queue, void* window, int index) {
    using clock = std::chrono::steady_clock;
    static thread_local clock::time_point next = clock::now();
    static std::atomic<uint64_t> frames{0};
    static clock::time_point report = clock::now();
    const auto frame = std::chrono::microseconds(33333);
    auto now = clock::now();
    if (next > now)
        std::this_thread::sleep_until(next);
    else if (now - next > frame * 4)
        next = now;  // fell behind (loading): resync instead of bursting
    next += frame;
    uint64_t n = ++frames;
    if (n == 1) LOGI(kTag, "first frame presented (window %p, texture %d)", window, index);
    if (clock::now() - report > std::chrono::seconds(10)) {
        report = clock::now();
        LOGI(kTag, "%llu frames presented", (unsigned long long)n);
    }
}

// ---- command buffers -----------------------------------------------------------------------------------------
uint64_t CmdEndRecording(void*) { return g_command_handle.fetch_add(1); }
size_t CmdZero(const void*) { return 0; }
size_t CmdPlenty(const void*) { return 16u << 20; }
bool CmdIsRecording(const void*) { return false; }

// ---- sync ------------------------------------------------------------------------------------------------------
int SyncWait(const void*, uint64_t timeout) { return 0; /* already signaled */ }

// ---- window ------------------------------------------------------------------------------------------------------
void WindowBuilderSetDefaults(void* b) { g_window_builders.Reset(b); }
void WindowBuilderSetTextures(void* b, int n, const void* const* textures) {
    auto& w = g_window_builders.Get(b);
    w.num_textures = std::min(n, 4);
    for (int i = 0; i < w.num_textures; i++) w.textures[i] = textures[i];
}
void WindowBuilderSetPresentInterval(void* b, int v) { g_window_builders.Get(b).present_interval = v; }
bool WindowInitialize(void* w, const void* b) {
    WindowState s = g_window_builders.Copy(b);
    g_windows.Get(w) = s;
    TextureState t = g_textures.Copy(s.textures[0]);
    LOGI(kTag, "nvnWindowInitialize: %d textures, %dx%d format %d", s.num_textures, t.width, t.height, t.format);
    return true;
}
void WindowFinalize(void* w) { g_windows.Erase(w); }
int WindowAcquireTexture(void* w, void* sync, int* index) {
    auto& s = g_windows.Get(w);
    int n = std::max(1, s.num_textures);
    *index = s.next;
    s.next = (s.next + 1) % n;
    return 0;  // success
}
int WindowGetNumTextures(const void* w) { return g_windows.Copy(w).num_textures; }

struct Proc {
    const char* name;
    void* fn;
};
#define P(n, f) {n, reinterpret_cast<void*>(&f)}

const Proc kProcs[] = {
    // device
    P("nvnDeviceBuilderSetDefaults", Nop), P("nvnDeviceBuilderSetFlags", Nop),
    P("nvnDeviceInitialize", DeviceInitialize), P("nvnDeviceFinalize", Nop),
    P("nvnDeviceGetInteger", DeviceGetInteger), P("nvnDeviceGetTextureHandle", DeviceGetTextureHandle),
    P("nvnDeviceGetImageHandle", DeviceGetImageHandle),
    P("nvnDeviceGetTexelFetchHandle", DeviceGetSeparateHandle),
    P("nvnDeviceGetSeparateTextureHandle", DeviceGetSeparateHandle),
    P("nvnDeviceGetSeparateSamplerHandle", DeviceGetSeparateHandle),
    P("nvnDeviceGetCurrentTimestampInNanoseconds", DeviceGetTimestampNs),
    P("nvnDeviceSetDepthMode", Nop), P("nvnDeviceSetWindowOriginMode", Nop),
    // memory pools
    P("nvnMemoryPoolBuilderSetDefaults", PoolBuilderSetDefaults), P("nvnMemoryPoolBuilderSetDevice", Nop),
    P("nvnMemoryPoolBuilderSetFlags", PoolBuilderSetFlags), P("nvnMemoryPoolBuilderSetStorage", PoolBuilderSetStorage),
    P("nvnMemoryPoolInitialize", PoolInitialize), P("nvnMemoryPoolFinalize", PoolFinalize),
    P("nvnMemoryPoolMap", PoolMap), P("nvnMemoryPoolGetBufferAddress", PoolGetBufferAddress),
    P("nvnMemoryPoolGetSize", PoolGetSize), P("nvnMemoryPoolGetFlags", PoolGetFlags),
    P("nvnMemoryPoolFlushMappedRange", Nop), P("nvnMemoryPoolInvalidateMappedRange", Nop),
    // buffers
    P("nvnBufferBuilderSetDefaults", BufferBuilderSetDefaults), P("nvnBufferBuilderSetDevice", Nop),
    P("nvnBufferBuilderSetStorage", BufferBuilderSetStorage), P("nvnBufferInitialize", BufferInitialize),
    P("nvnBufferFinalize", BufferFinalize), P("nvnBufferMap", BufferMap), P("nvnBufferGetAddress", BufferGetAddress),
    P("nvnBufferGetSize", BufferGetSize), P("nvnBufferFlushMappedRange", Nop), P("nvnBufferInvalidateMappedRange", Nop),
    // textures
    P("nvnTextureBuilderSetDefaults", TexBuilderSetDefaults), P("nvnTextureBuilderSetDevice", Nop),
    P("nvnTextureBuilderSetFlags", TexBuilderSetFlags), P("nvnTextureBuilderSetTarget", TexBuilderSetTarget),
    P("nvnTextureBuilderSetWidth", TexBuilderSetWidth), P("nvnTextureBuilderSetHeight", TexBuilderSetHeight),
    P("nvnTextureBuilderSetDepth", TexBuilderSetDepth), P("nvnTextureBuilderSetSize1D", TexBuilderSetSize1D),
    P("nvnTextureBuilderSetSize2D", TexBuilderSetSize2D), P("nvnTextureBuilderSetSize3D", TexBuilderSetSize3D),
    P("nvnTextureBuilderSetLevels", TexBuilderSetLevels), P("nvnTextureBuilderSetFormat", TexBuilderSetFormat),
    P("nvnTextureBuilderSetSamples", TexBuilderSetSamples), P("nvnTextureBuilderSetStride", TexBuilderSetStride),
    P("nvnTextureBuilderSetStorage", TexBuilderSetStorage), P("nvnTextureBuilderSetPackagedTextureData", Nop),
    P("nvnTextureBuilderSetPackagedTextureLayout", Nop), P("nvnTextureBuilderSetSwizzle", Nop),
    P("nvnTextureBuilderSetDepthStencilMode", Nop), P("nvnTextureBuilderSetGLTextureName", Nop),
    P("nvnTextureBuilderGetStorageSize", TexBuilderGetStorageSize),
    P("nvnTextureBuilderGetStorageAlignment", TexBuilderGetStorageAlignment),
    P("nvnTextureBuilderGetWidth", TexBuilderGetWidth), P("nvnTextureBuilderGetHeight", TexBuilderGetHeight),
    P("nvnTextureBuilderGetDepth", TexBuilderGetDepth), P("nvnTextureBuilderGetLevels", TexBuilderGetLevels),
    P("nvnTextureBuilderGetFormat", TexBuilderGetFormat), P("nvnTextureBuilderGetTarget", TexBuilderGetTarget),
    P("nvnTextureInitialize", TexInitialize), P("nvnTextureFinalize", TexFinalize),
    P("nvnTextureGetWidth", TexGetWidth), P("nvnTextureGetHeight", TexGetHeight),
    P("nvnTextureGetDepth", TexGetDepth), P("nvnTextureGetLevels", TexGetLevels),
    P("nvnTextureGetFormat", TexGetFormat), P("nvnTextureGetSamples", TexGetSamples),
    P("nvnTextureGetTarget", TexGetTarget), P("nvnTextureGetStorageSize", TexGetStorageSize),
    P("nvnTextureGetTextureAddress", TexGetTextureAddress), P("nvnTextureFlushTexels", Nop),
    P("nvnTextureInvalidateTexels", Nop), P("nvnTextureWriteTexels", Nop), P("nvnTextureReadTexels", Nop),
    // pools of descriptors, samplers, programs
    P("nvnTexturePoolInitialize", True), P("nvnTexturePoolFinalize", Nop), P("nvnTexturePoolRegisterTexture", Nop),
    P("nvnTexturePoolRegisterImage", Nop), P("nvnSamplerPoolInitialize", True), P("nvnSamplerPoolFinalize", Nop),
    P("nvnSamplerPoolRegisterSampler", Nop), P("nvnSamplerPoolRegisterSamplerBuilder", Nop),
    P("nvnSamplerInitialize", True), P("nvnSamplerFinalize", Nop),
    P("nvnProgramInitialize", True), P("nvnProgramFinalize", Nop), P("nvnProgramSetShaders", True),
    P("nvnEventInitialize", True), P("nvnEventFinalize", Nop),
    // queues
    P("nvnQueueBuilderSetDefaults", QueueBuilderSetDefaults), P("nvnQueueBuilderSetDevice", Nop),
    P("nvnQueueBuilderSetFlags", Nop), P("nvnQueueBuilderSetCommandMemorySize", QueueBuilderSetCommandMemorySize),
    P("nvnQueueBuilderSetComputeMemorySize", QueueBuilderSetComputeMemorySize),
    P("nvnQueueBuilderSetCommandFlushThreshold", Nop), P("nvnQueueBuilderSetQueueMemory", Nop),
    P("nvnQueueBuilderGetQueueMemorySize", QueueBuilderGetQueueMemorySize),
    P("nvnQueueInitialize", QueueInitialize), P("nvnQueueFinalize", Nop), P("nvnQueueSubmitCommands", Nop),
    P("nvnQueueFlush", Nop), P("nvnQueueFinish", Nop), P("nvnQueueFenceSync", Nop), P("nvnQueueWaitSync", Nop),
    P("nvnQueuePresentTexture", QueuePresentTexture),
    // command buffers (recording is a no-op)
    P("nvnCommandBufferInitialize", True), P("nvnCommandBufferFinalize", Nop),
    P("nvnCommandBufferAddCommandMemory", Nop), P("nvnCommandBufferAddControlMemory", Nop),
    P("nvnCommandBufferBeginRecording", Nop), P("nvnCommandBufferEndRecording", CmdEndRecording),
    P("nvnCommandBufferGetCommandMemoryUsed", CmdZero), P("nvnCommandBufferGetControlMemoryUsed", CmdZero),
    P("nvnCommandBufferGetCommandMemoryFree", CmdPlenty), P("nvnCommandBufferGetControlMemoryFree", CmdPlenty),
    P("nvnCommandBufferGetCommandMemorySize", CmdPlenty), P("nvnCommandBufferGetControlMemorySize", CmdPlenty),
    P("nvnCommandBufferIsRecording", CmdIsRecording),
    // sync
    P("nvnSyncInitialize", True), P("nvnSyncFinalize", Nop), P("nvnSyncWait", SyncWait),
    // window
    P("nvnWindowBuilderSetDefaults", WindowBuilderSetDefaults), P("nvnWindowBuilderSetDevice", Nop),
    P("nvnWindowBuilderSetNativeWindow", Nop), P("nvnWindowBuilderSetTextures", WindowBuilderSetTextures),
    P("nvnWindowBuilderSetPresentInterval", WindowBuilderSetPresentInterval),
    P("nvnWindowInitialize", WindowInitialize), P("nvnWindowFinalize", WindowFinalize),
    P("nvnWindowAcquireTexture", WindowAcquireTexture), P("nvnWindowGetNumActiveTextures", WindowGetNumTextures),
    P("nvnWindowGetNumTextures", WindowGetNumTextures), P("nvnWindowSetPresentInterval", Nop),
    P("nvnWindowSetCrop", Nop), P("nvnWindowSetNumActiveTextures", Nop),
};
#undef P

// Every other nvnCommandBuffer*/state-object setter is a no-op with no return value worth tracking.
const char* const kNopPrefixes[] = {
    "nvnCommandBuffer", "nvnBlendState", "nvnChannelMaskState", "nvnColorState", "nvnDepthStencilState",
    "nvnMultisampleState", "nvnPolygonState", "nvnVertexAttribState", "nvnVertexStreamState",
    "nvnSamplerBuilder", "nvnTextureView",
};

struct Registrar {
    Registrar() {
        for (const Proc& p : kProcs) RegisterNvnProc(p.name, p.fn);
        for (const char* prefix : kNopPrefixes) RegisterNvnPrefix(prefix, reinterpret_cast<void*>(&Nop));
    }
} g_registrar;

}  // namespace
}  // namespace dsr::hle::nvn
