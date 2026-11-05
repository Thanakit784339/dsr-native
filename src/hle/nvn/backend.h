#pragma once
// The NVN front end (null_driver.cpp + commands.cpp) records what the game asks for into host command
// lists; a Backend executes them. On Android the backend is Vulkan (vk_backend.cpp); elsewhere it is a
// null backend that only counts, so the recorder can be unit-tested on any host.

#include <cstdint>
#include <functional>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "hle/nvn/nvn_types.h"

namespace dsr::hle::nvn {

// A texture as seen at record time: its description plus the address of its storage. Images on the
// host GPU are keyed by storage address, so two NVN texture objects over the same memory share one.
struct TexRef {
    bool valid = false;
    TextureState tex;
    uint64_t addr = 0;
    // From an NVNtextureView: the subresource used (render targets render into this level/layer).
    int level = 0, layer = 0;
    int view_levels = 0, view_layers = 0;  // 0: all from level/layer on
    int view_format = 0;                    // NVN format the view reinterprets the texture as (0: its own)
};

struct CopyRegion {  // NVNcopyRegion
    int x = 0, y = 0, z = 0, width = 0, height = 0, depth = 0;
};

enum class CmdType : uint8_t {
    SetRenderTargets,
    ClearColor,         // index, color, kind, mask
    ClearDepthStencil,  // depth, depth_mask, stencil, stencil_mask
    ClearTexture,       // dst, dst_region, color, kind, mask
    CopyTexture,        // src, src_region, dst, dst_region, flags
    Draw,               // draw (see `draw`)
    BindProgram,        // obj: ShaderCode, index: stage bits
    BindVertexAttribs,  // obj: std::vector<VertexAttribObj>
    BindVertexStreams,  // obj: std::vector<VertexStreamObj>
    BindVertexBuffers,  // index: first, obj: std::vector<BufferRange>
    BindUniformBuffers, // stage, index: first, obj: std::vector<BufferRange>
    BindTextures,       // stage, index: first, obj: std::vector<BoundTexture>
    BindImages,         // stage, index: first, obj: std::vector<TexRef>
    BindBlend,          // obj: BlendStateObj
    BindColor,          // obj: ColorStateObj
    BindChannelMask,    // obj: ChannelMaskStateObj
    BindDepthStencil,   // obj: DepthStencilStateObj
    BindPolygon,        // obj: PolygonStateObj
    BindMultisample,    // obj: MultisampleStateObj
    SetViewport,        // f[0..3] = x, y, w, h
    SetScissor,         // f[0..3] = x, y, w, h
    SetDepthRange,      // f[0..1] = near, far
    SetAlphaRef,        // f[0]
    SetBlendColor,      // f[0..3]
    SetStencil,         // index: 0 ref, 1 write mask, 2 compare mask; stage: face; mask: value
    UpdateUniform,      // dst_addr; obj: std::vector<uint8_t>
    CopyBufferToTexture,// src_addr, dst, dst_region, flags, f[0] = row stride, f[1] = image stride
    FenceSync,          // src_addr: the NVNsync, signalled once the list has executed (handled by the queue)
    CopyTextureToBuffer,// src, src_region, dst_addr, flags, f[0] = row stride, f[1] = image stride
    CopyBufferToBuffer, // src_addr, dst_addr, index = size (bytes; see `size64`)
    Report,             // dst_addr; index 0: value (`mask` = the value), 1: counter (`mask` = NVNcounterType)
    CallList,           // obj: const CmdList, executed in place (nvnCommandBufferCallCommands / CopyCommands)
};

struct BufferRange {  // NVNbufferRange
    uint64_t address = 0;
    uint64_t size = 0;
};

struct BoundTexture {
    TexRef texture;
    SamplerObj sampler;
    bool valid = false;
};

struct DrawParams {
    int mode = 4;           // NVNdrawPrimitive (4 = TRIANGLES)
    int index_type = -1;    // -1: non-indexed; 0 u8, 1 u16, 2 u32
    int first = 0, count = 0;
    uint64_t index_address = 0;
    int base_vertex = 0, base_instance = 0, instances = 1;
};

enum ClearKind : uint8_t { kClearFloat = 0, kClearInt = 1, kClearUint = 2 };

// Textures and regions of copies / texture clears (rare commands: kept out of Cmd).
struct CopyInfo {
    TexRef src, dst;
    CopyRegion src_region, dst_region;
};

struct Cmd {
    CmdType type{};
    uint8_t kind = kClearFloat;
    bool depth_mask = false;
    uint16_t inline_count = 0;  // payload items stored in `inline_data` (when `obj` is null)
    int index = 0, mask = 0, stencil = 0, stencil_mask = 0, flags = 0;
    uint32_t color[4] = {};
    float depth = 0;
    int stage = 0;
    float f[4] = {};
    uint64_t src_addr = 0, dst_addr = 0;
    DrawParams draw;
    // Type-specific payload (see CmdType): small ones (state objects, a few buffer ranges or vertex streams)
    // inline, larger ones (and shader code, called lists) on the heap. Binds used to allocate twice each, and
    // the recording threads spent half their time in the allocator.
    static constexpr size_t kInlineBytes = 96;
    alignas(8) uint8_t inline_data[kInlineBytes];
    std::shared_ptr<const void> obj;
    // SetRenderTargets
    int num_rt = 0;
    std::shared_ptr<const std::vector<TexRef>> rts;  // colors[0..num_rt) then depth at [num_rt]
    std::shared_ptr<const CopyInfo> copy;            // CopyTexture, ClearTexture, CopyBufferToTexture, CopyTextureToBuffer
    uint64_t size64 = 0;  // CopyBufferToBuffer
};
// Payload items of a command: inline, or a std::vector<T> on the heap.
template <typename T>
void SetItems(Cmd& c, const T* p, size_t n) {
    static_assert(std::is_trivially_copyable_v<T>, "inline payloads are copied bytewise");
    if (n * sizeof(T) <= Cmd::kInlineBytes) {
        if (n) memcpy(c.inline_data, p, n * sizeof(T));
        c.inline_count = uint16_t(n);
        c.obj = nullptr;
    } else {
        c.obj = std::make_shared<const std::vector<T>>(p, p + n);
    }
}
template <typename T>
std::span<const T> Items(const Cmd& c) {
    if (c.obj) {
        const auto& v = *static_cast<const std::vector<T>*>(c.obj.get());
        return {v.data(), v.size()};
    }
    return {reinterpret_cast<const T*>(c.inline_data), c.inline_count};
}
template <typename T>
T One(const Cmd& c) {
    const std::span<const T> s = Items<T>(c);
    return s.empty() ? T{} : s[0];
}

using CmdList = std::vector<Cmd>;
// Calls `f` for every command of `list`, descending into CallList commands (nesting is shallow in practice).
template <typename F>
void ForEachCmd(const CmdList& list, F&& f, int depth = 0) {
    for (const Cmd& c : list) {
        if (c.type == CmdType::CallList) {
            if (c.obj && depth < 8) ForEachCmd(*static_cast<const CmdList*>(c.obj.get()), f, depth + 1);
        } else {
            f(c);
        }
    }
}

class Backend {
public:
    virtual ~Backend() = default;
    virtual void Execute(const CmdList& cmds) = 0;
    // Show the window texture (null ref: nothing rendered yet).
    virtual void Present(const TexRef& texture) = 0;
    virtual const char* Name() const = 0;
    // Diagnostics: capture a frame `delay_s` seconds from now (e.g. during a video). Thread-safe.
    virtual void RequestCapture(int /*delay_s*/) {}
    // A new picture (linear RGBA8, w x h) for the texture whose storage is at `addr` (a video frame): used as
    // that texture's level 0 from its next use on. Thread-safe; a newer frame replaces one not yet used.
    // The CPU rewrote the storage of the texture at `addr` (nvnTextureWriteTexels): re-check it at its next
    // use. Thread-safe.
    virtual void MarkTextureDirty(uint64_t /*addr*/) {}
    virtual void SetTextureFrame(uint64_t /*addr*/, std::shared_ptr<const std::vector<uint8_t>> /*rgba*/, int /*w*/,
                                 int /*h*/) {}
};

Backend& GetBackend();
void SetBackendForTesting(Backend* b);  // nullptr restores the default
Backend* CreatePlatformBackend();       // Vulkan on Android, null elsewhere

// Lookups provided by null_driver.cpp.
TexRef MakeTexRef(const void* nvn_texture);
TexRef WindowTexture(const void* nvn_window, int index);
// True if [addr, addr+size) lies inside one registered NVN memory pool (safe to read).
bool GuestRangeValid(uint64_t addr, size_t size);
// Hands a linear RGBA8 picture (w x h, tightly packed) to the backend as the new contents of a texture:
// `addr` is the texture's storage address or an NVNtexture object pointer. False if no such RGBA8 texture.
bool SubmitTextureFrame(uint64_t addr, std::shared_ptr<const std::vector<uint8_t>> rgba, int w, int h);

// Per-frame statistics, reported every 10 s by the present path.
struct GpuStats {
    uint64_t submits = 0, commands = 0, draws = 0, clears = 0, copies = 0, rt_binds = 0;
    uint64_t recordings = 0, submit_calls = 0, unknown_handles = 0;
};
GpuStats& Stats();

// The queue: command lists and presents run in order on a render thread (Android), so the game records the
// next frame while the previous one is executed. Inline (synchronous) on other hosts and in tests.
uint64_t GpuEnqueue(std::function<void()> task);  // returns the task's sequence number
void GpuWait(uint64_t seq);                         // until task `seq` (and all before it) has run
uint64_t GpuSubmitted();
// NVNsync objects: signalled once queue task `seq` has run (0: signalled now).
void SetSyncTarget(const void* sync, uint64_t seq);
// Milliseconds game threads waited for the render thread since the last call (presents, sync waits).
void TakeGameWaits(double* present_ms, double* sync_ms);

}  // namespace dsr::hle::nvn
