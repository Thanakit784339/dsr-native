// NVN command buffers and queue submission.
//
// The game records commands into NVNcommandBuffer objects (BeginRecording ... EndRecording, which
// returns an NVNcommandHandle) and submits handles to a queue. We record each command into a host
// CmdList; EndRecording files the list under a new handle; nvnQueueSubmitCommands hands the lists to
// the backend in order. Handles may be submitted many times; the oldest are dropped once there are
// more than kMaxHandles (real NVN handles die when their command memory is reused anyway).
//
// Every texture referenced by a command is captured by value (description + storage address) at
// record time, so later Finalize/re-Initialize of the NVN object does not affect recorded work.

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "common/log.h"
#include "hle/nvn.h"
#include "hle/nvn/backend.h"
#include "hle/nvn/state.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "nvn";
constexpr size_t kMaxHandles = 16384;

// The game checks GetCommandMemoryUsed before submitting and skips buffers that report 0, so report a
// plausible figure: an estimate of what real NVN would have written, reset when memory is (re)added.
constexpr size_t kBytesPerCommand = 32;
constexpr size_t kControlBytesPerRecording = 64;

struct Recorder {
    bool recording = false;
    CmdList cmds;
    size_t command_used = 0, control_used = 0;
};
Table<Recorder> g_recorders;

std::mutex g_handle_mutex;
std::unordered_map<uint64_t, std::shared_ptr<const CmdList>> g_handles;
std::deque<uint64_t> g_handle_order;
uint64_t g_next_handle = 1;

GpuStats g_stats;
std::atomic<uint64_t> g_begin_count{0}, g_end_count{0}, g_submit_calls{0}, g_unknown_handles{0};
std::unique_ptr<Backend> g_default_backend;
Backend* g_test_backend = nullptr;
std::once_flag g_backend_once;

std::shared_ptr<const CmdList> FindHandle(uint64_t h) {
    std::lock_guard<std::mutex> lk(g_handle_mutex);
    auto it = g_handles.find(h);
    return it == g_handles.end() ? nullptr : it->second;
}

void Append(void* cb, Cmd&& c) {
    Recorder& r = g_recorders.Get(cb);
    r.cmds.push_back(std::move(c));
    r.command_used += kBytesPerCommand;
}

// ---- recording ------------------------------------------------------------------------------------------
void BeginRecording(void* cb) {
    Recorder& r = g_recorders.Get(cb);
    r.recording = true;
    r.cmds.clear();
    r.control_used += kControlBytesPerRecording;
    g_begin_count++;
}
void AddCommandMemory(void* cb, const void*, int64_t, size_t) { g_recorders.Get(cb).command_used = 0; }
void AddControlMemory(void* cb, void*, size_t) { g_recorders.Get(cb).control_used = 0; }
size_t GetCommandMemoryUsed(const void* cb) { return g_recorders.Copy(cb).command_used; }
size_t GetControlMemoryUsed(const void* cb) { return g_recorders.Copy(cb).control_used; }
uint64_t EndRecording(void* cb) {
    Recorder& r = g_recorders.Get(cb);
    auto list = std::make_shared<const CmdList>(std::move(r.cmds));
    r.cmds = CmdList{};
    r.recording = false;
    g_end_count++;
    std::lock_guard<std::mutex> lk(g_handle_mutex);
    uint64_t h = g_next_handle++;
    g_handles.emplace(h, std::move(list));
    g_handle_order.push_back(h);
    while (g_handle_order.size() > kMaxHandles) {
        g_handles.erase(g_handle_order.front());
        g_handle_order.pop_front();
    }
    return h;
}
bool IsRecording(const void* cb) { return g_recorders.Copy(cb).recording; }

// Inline previously recorded command lists (CallCommands / CopyCommands behave the same for us).
void CallCommands(void* cb, int n, const uint64_t* handles) {
    Recorder& r = g_recorders.Get(cb);
    for (int i = 0; i < n; i++)
        if (auto list = FindHandle(handles[i])) r.cmds.insert(r.cmds.end(), list->begin(), list->end());
}

void SetRenderTargets(void* cb, int n, const void* const* colors, const void* const* /*color_views*/,
                      const void* depth, const void* /*depth_view*/) {
    Cmd c;
    c.type = CmdType::SetRenderTargets;
    c.num_rt = n < 0 ? 0 : (n > 8 ? 8 : n);
    auto rts = std::make_shared<std::vector<TexRef>>(c.num_rt + 1);
    for (int i = 0; i < c.num_rt; i++)
        if (colors && colors[i]) (*rts)[i] = MakeTexRef(colors[i]);
    if (depth) (*rts)[c.num_rt] = MakeTexRef(depth);
    c.rts = std::move(rts);
    Append(cb, std::move(c));
}

void ClearColorAny(void* cb, int index, const void* color, int mask, uint8_t kind) {
    Cmd c;
    c.type = CmdType::ClearColor;
    c.index = index;
    c.mask = mask;
    c.kind = kind;
    if (color) memcpy(c.color, color, sizeof(c.color));
    Append(cb, std::move(c));
}
void ClearColor(void* cb, int index, const float* color, int mask) { ClearColorAny(cb, index, color, mask, kClearFloat); }
void ClearColori(void* cb, int index, const int* color, int mask) { ClearColorAny(cb, index, color, mask, kClearInt); }
void ClearColorui(void* cb, int index, const uint32_t* color, int mask) {
    ClearColorAny(cb, index, color, mask, kClearUint);
}

void ClearDepthStencil(void* cb, float depth, bool depth_mask, int stencil, int stencil_mask) {
    Cmd c;
    c.type = CmdType::ClearDepthStencil;
    c.depth = depth;
    c.depth_mask = depth_mask;
    c.stencil = stencil;
    c.stencil_mask = stencil_mask;
    Append(cb, std::move(c));
}

void ClearTextureAny(void* cb, const void* tex, const void* /*view*/, const CopyRegion* region, const void* color,
                     int mask, uint8_t kind) {
    Cmd c;
    c.type = CmdType::ClearTexture;
    c.dst = MakeTexRef(tex);
    if (region) c.dst_region = *region;
    c.mask = mask;
    c.kind = kind;
    if (color) memcpy(c.color, color, sizeof(c.color));
    Append(cb, std::move(c));
}
void ClearTexture(void* cb, const void* t, const void* v, const CopyRegion* r, const float* color, int mask) {
    ClearTextureAny(cb, t, v, r, color, mask, kClearFloat);
}
void ClearTexturei(void* cb, const void* t, const void* v, const CopyRegion* r, const int* color, int mask) {
    ClearTextureAny(cb, t, v, r, color, mask, kClearInt);
}
void ClearTextureui(void* cb, const void* t, const void* v, const CopyRegion* r, const uint32_t* color, int mask) {
    ClearTextureAny(cb, t, v, r, color, mask, kClearUint);
}

void CopyTextureToTexture(void* cb, const void* src, const void* /*src_view*/, const CopyRegion* src_region,
                          const void* dst, const void* /*dst_view*/, const CopyRegion* dst_region, int flags) {
    Cmd c;
    c.type = CmdType::CopyTexture;
    c.src = MakeTexRef(src);
    c.dst = MakeTexRef(dst);
    if (src_region) c.src_region = *src_region;
    if (dst_region) c.dst_region = *dst_region;
    c.flags = flags;
    Append(cb, std::move(c));
}

// ---- binds ----------------------------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const void> Obj(T v) {
    return std::make_shared<const T>(std::move(v));
}
void Bind(void* cb, CmdType t, std::shared_ptr<const void> obj, int stage = 0, int index = 0) {
    Cmd c;
    c.type = t;
    c.obj = std::move(obj);
    c.stage = stage;
    c.index = index;
    Append(cb, std::move(c));
}
void BindProgram(void* cb, const void* program, int stage_bits) {
    Bind(cb, CmdType::BindProgram, GetProgramCode(program), 0, stage_bits);
}
// NVNvertexAttribState is 4 bytes and NVNvertexStreamState 8 bytes in game memory.
void BindVertexAttribState(void* cb, int n, const uint8_t* states) {
    std::vector<VertexAttribObj> v;
    for (int i = 0; i < n && states; i++) v.push_back(GetVertexAttrib(states + 4 * i));
    Bind(cb, CmdType::BindVertexAttribs, Obj(std::move(v)));
}
void BindVertexStreamState(void* cb, int n, const uint8_t* states) {
    std::vector<VertexStreamObj> v;
    for (int i = 0; i < n && states; i++) v.push_back(GetVertexStream(states + 8 * i));
    Bind(cb, CmdType::BindVertexStreams, Obj(std::move(v)));
}
void BindVertexBuffer(void* cb, int index, uint64_t addr, uint64_t size) {
    Bind(cb, CmdType::BindVertexBuffers, Obj(std::vector<BufferRange>{{addr, size}}), 0, index);
}
void BindVertexBuffers(void* cb, int first, int n, const BufferRange* ranges) {
    Bind(cb, CmdType::BindVertexBuffers, Obj(std::vector<BufferRange>(ranges, ranges + std::max(0, n))), 0, first);
}
void BindUniformBuffer(void* cb, int stage, int index, uint64_t addr, uint64_t size) {
    Bind(cb, CmdType::BindUniformBuffers, Obj(std::vector<BufferRange>{{addr, size}}), stage, index);
}
void BindUniformBuffers(void* cb, int stage, int first, int n, const BufferRange* ranges) {
    Bind(cb, CmdType::BindUniformBuffers, Obj(std::vector<BufferRange>(ranges, ranges + std::max(0, n))), stage, first);
}
// NVN texture handles: texture ID in the low 20 bits, sampler ID above (see DeviceGetTextureHandle).
BoundTexture ResolveHandle(uint64_t handle) {
    BoundTexture b;
    b.texture = TextureById(int(handle & 0xFFFFF));
    b.sampler = SamplerById(int(handle >> 20));
    b.valid = b.texture.valid;
    return b;
}
void BindTexture(void* cb, int stage, int index, uint64_t handle) {
    Bind(cb, CmdType::BindTextures, Obj(std::vector<BoundTexture>{ResolveHandle(handle)}), stage, index);
}
void BindTextures(void* cb, int stage, int first, int n, const uint64_t* handles) {
    std::vector<BoundTexture> v;
    for (int i = 0; i < n && handles; i++) v.push_back(ResolveHandle(handles[i]));
    Bind(cb, CmdType::BindTextures, Obj(std::move(v)), stage, first);
}
// Image handles (nvnDeviceGetImageHandle) are the pool ids.
void BindImage(void* cb, int stage, int index, uint64_t handle) {
    Bind(cb, CmdType::BindImages, Obj(std::vector<TexRef>{ImageById(int(handle & 0xFFFFF))}), stage, index);
}
void BindImages(void* cb, int stage, int first, int n, const uint64_t* handles) {
    std::vector<TexRef> v;
    for (int i = 0; i < n && handles; i++) v.push_back(ImageById(int(handles[i] & 0xFFFFF)));
    Bind(cb, CmdType::BindImages, Obj(std::move(v)), stage, first);
}
void BindBlendState(void* cb, const void* s) { Bind(cb, CmdType::BindBlend, Obj(GetBlendState(s))); }
void BindColorState(void* cb, const void* s) { Bind(cb, CmdType::BindColor, Obj(GetColorState(s))); }
void BindChannelMaskState(void* cb, const void* s) { Bind(cb, CmdType::BindChannelMask, Obj(GetChannelMaskState(s))); }
void BindDepthStencilState(void* cb, const void* s) { Bind(cb, CmdType::BindDepthStencil, Obj(GetDepthStencilState(s))); }
void BindPolygonState(void* cb, const void* s) { Bind(cb, CmdType::BindPolygon, Obj(GetPolygonState(s))); }
void BindMultisampleState(void* cb, const void* s) { Bind(cb, CmdType::BindMultisample, Obj(GetMultisampleState(s))); }

// ---- dynamic state ------------------------------------------------------------------------------------
void SetF(void* cb, CmdType t, float a, float b = 0, float c = 0, float d = 0) {
    Cmd cmd;
    cmd.type = t;
    cmd.f[0] = a;
    cmd.f[1] = b;
    cmd.f[2] = c;
    cmd.f[3] = d;
    Append(cb, std::move(cmd));
}
void SetViewport(void* cb, int x, int y, int w, int h) { SetF(cb, CmdType::SetViewport, float(x), float(y), float(w), float(h)); }
void SetViewports(void* cb, int first, int n, const float* v) {
    if (first == 0 && n > 0 && v) SetF(cb, CmdType::SetViewport, v[0], v[1], v[2], v[3]);
}
void SetScissor(void* cb, int x, int y, int w, int h) { SetF(cb, CmdType::SetScissor, float(x), float(y), float(w), float(h)); }
void SetScissors(void* cb, int first, int n, const int* v) {
    if (first == 0 && n > 0 && v) SetF(cb, CmdType::SetScissor, float(v[0]), float(v[1]), float(v[2]), float(v[3]));
}
void SetDepthRange(void* cb, float n, float f) { SetF(cb, CmdType::SetDepthRange, n, f); }
void SetAlphaRef(void* cb, float ref) { SetF(cb, CmdType::SetAlphaRef, ref); }
void SetBlendColor(void* cb, const float* c) { SetF(cb, CmdType::SetBlendColor, c[0], c[1], c[2], c[3]); }
void SetStencil(void* cb, int which, int face, int value) {
    Cmd c;
    c.type = CmdType::SetStencil;
    c.index = which;
    c.stage = face;
    c.mask = value;
    Append(cb, std::move(c));
}
void SetStencilRef(void* cb, int face, int v) { SetStencil(cb, 0, face, v); }
void SetStencilMask(void* cb, int face, int v) { SetStencil(cb, 1, face, v); }
void SetStencilValueMask(void* cb, int face, int v) { SetStencil(cb, 2, face, v); }

// ---- draws --------------------------------------------------------------------------------------------
void RecordDraw(void* cb, const DrawParams& p) {
    Cmd c;
    c.type = CmdType::Draw;
    c.draw = p;
    Append(cb, std::move(c));
}
void DrawArrays(void* cb, int mode, int first, int count) {
    DrawParams p;
    p.mode = mode;
    p.first = first;
    p.count = count;
    RecordDraw(cb, p);
}
void DrawArraysInstanced(void* cb, int mode, int first, int count, int base_instance, int instances) {
    DrawParams p;
    p.mode = mode;
    p.first = first;
    p.count = count;
    p.base_instance = base_instance;
    p.instances = instances;
    RecordDraw(cb, p);
}
void DrawElementsBaseVertex(void* cb, int mode, int type, int count, uint64_t index_addr, int base_vertex) {
    DrawParams p;
    p.mode = mode;
    p.index_type = type;
    p.count = count;
    p.index_address = index_addr;
    p.base_vertex = base_vertex;
    RecordDraw(cb, p);
}
void DrawElements(void* cb, int mode, int type, int count, uint64_t index_addr) {
    DrawElementsBaseVertex(cb, mode, type, count, index_addr, 0);
}
void DrawElementsInstanced(void* cb, int mode, int type, int count, uint64_t index_addr, int base_vertex,
                           int base_instance, int instances) {
    DrawParams p;
    p.mode = mode;
    p.index_type = type;
    p.count = count;
    p.index_address = index_addr;
    p.base_vertex = base_vertex;
    p.base_instance = base_instance;
    p.instances = instances;
    RecordDraw(cb, p);
}
// Not implemented yet; recorded as draws with no vertices so they are counted.
void DrawUnsupported(void* cb) { RecordDraw(cb, DrawParams{0, -1, 0, 0, 0, 0, 0, 0}); }

// ---- data updates ---------------------------------------------------------------------------------------
// The update lands in the buffer when the command executes, as on the GPU.
void UpdateUniformBuffer(void* cb, uint64_t buffer, size_t buffer_size, int64_t offset, size_t size, const void* data) {
    if (!data || !size || offset < 0 || size_t(offset) + size > buffer_size) return;
    Cmd c;
    c.type = CmdType::UpdateUniform;
    c.dst_addr = buffer + uint64_t(offset);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    c.obj = Obj(std::vector<uint8_t>(p, p + size));
    Append(cb, std::move(c));
}
std::atomic<int64_t> g_copy_row_stride{0}, g_copy_image_stride{0};
void SetCopyRowStride(void* cb, int64_t v) { g_copy_row_stride = v; }
void SetCopyImageStride(void* cb, int64_t v) { g_copy_image_stride = v; }
int64_t GetCopyRowStride(const void*) { return g_copy_row_stride; }
int64_t GetCopyImageStride(const void*) { return g_copy_image_stride; }
void CopyBufferToTexture(void* cb, uint64_t src, const void* dst, const void* /*view*/, const CopyRegion* region,
                         int flags) {
    Cmd c;
    c.type = CmdType::CopyBufferToTexture;
    c.src_addr = src;
    c.dst = MakeTexRef(dst);
    if (region) c.dst_region = *region;
    c.flags = flags;
    c.f[0] = float(g_copy_row_stride.load());
    c.f[1] = float(g_copy_image_stride.load());
    Append(cb, std::move(c));
}

// ---- queue ------------------------------------------------------------------------------------------------
void QueueSubmitCommands(void* /*queue*/, int n, const uint64_t* handles) {
    g_submit_calls++;
    for (int i = 0; i < n; i++) {
        auto list = FindHandle(handles[i]);
        if (!list) {
            g_unknown_handles++;
            continue;
        }
        g_stats.submits++;
        g_stats.commands += list->size();
        for (const Cmd& c : *list) {
            switch (c.type) {
                case CmdType::Draw: g_stats.draws++; break;
                default: break;
                case CmdType::ClearColor:
                case CmdType::ClearDepthStencil:
                case CmdType::ClearTexture: g_stats.clears++; break;
                case CmdType::CopyTexture: g_stats.copies++; break;
                case CmdType::SetRenderTargets: g_stats.rt_binds++; break;
            }
        }
        GetBackend().Execute(*list);
    }
}

struct Proc {
    const char* name;
    void* fn;
};
#define P(n, f) {n, reinterpret_cast<void*>(&f)}
const Proc kProcs[] = {
    P("nvnCommandBufferBeginRecording", BeginRecording),
    P("nvnCommandBufferEndRecording", EndRecording),
    P("nvnCommandBufferIsRecording", IsRecording),
    P("nvnCommandBufferAddCommandMemory", AddCommandMemory),
    P("nvnCommandBufferAddControlMemory", AddControlMemory),
    P("nvnCommandBufferGetCommandMemoryUsed", GetCommandMemoryUsed),
    P("nvnCommandBufferGetControlMemoryUsed", GetControlMemoryUsed),
    P("nvnCommandBufferCallCommands", CallCommands),
    P("nvnCommandBufferCopyCommands", CallCommands),
    P("nvnCommandBufferSetRenderTargets", SetRenderTargets),
    P("nvnCommandBufferClearColor", ClearColor),
    P("nvnCommandBufferClearColori", ClearColori),
    P("nvnCommandBufferClearColorui", ClearColorui),
    P("nvnCommandBufferClearDepthStencil", ClearDepthStencil),
    P("nvnCommandBufferClearTexture", ClearTexture),
    P("nvnCommandBufferClearTexturei", ClearTexturei),
    P("nvnCommandBufferClearTextureui", ClearTextureui),
    P("nvnCommandBufferCopyTextureToTexture", CopyTextureToTexture),
    P("nvnCommandBufferDrawArrays", DrawArrays),
    P("nvnCommandBufferDrawArraysInstanced", DrawArraysInstanced),
    P("nvnCommandBufferDrawElements", DrawElements),
    P("nvnCommandBufferDrawElementsBaseVertex", DrawElementsBaseVertex),
    P("nvnCommandBufferDrawElementsInstanced", DrawElementsInstanced),
    P("nvnCommandBufferDrawArraysIndirect", DrawUnsupported),
    P("nvnCommandBufferDrawElementsIndirect", DrawUnsupported),
    P("nvnCommandBufferMultiDrawArraysIndirectCount", DrawUnsupported),
    P("nvnCommandBufferMultiDrawElementsIndirectCount", DrawUnsupported),
    P("nvnCommandBufferDrawTexture", DrawUnsupported),
    P("nvnCommandBufferDrawTransformFeedback", DrawUnsupported),
    P("nvnCommandBufferBindProgram", BindProgram),
    P("nvnCommandBufferBindVertexAttribState", BindVertexAttribState),
    P("nvnCommandBufferBindVertexStreamState", BindVertexStreamState),
    P("nvnCommandBufferBindVertexBuffer", BindVertexBuffer),
    P("nvnCommandBufferBindVertexBuffers", BindVertexBuffers),
    P("nvnCommandBufferBindUniformBuffer", BindUniformBuffer),
    P("nvnCommandBufferBindUniformBuffers", BindUniformBuffers),
    P("nvnCommandBufferBindTexture", BindTexture),
    P("nvnCommandBufferBindTextures", BindTextures),
    P("nvnCommandBufferBindImage", BindImage),
    P("nvnCommandBufferBindImages", BindImages),
    P("nvnCommandBufferBindBlendState", BindBlendState),
    P("nvnCommandBufferBindColorState", BindColorState),
    P("nvnCommandBufferBindChannelMaskState", BindChannelMaskState),
    P("nvnCommandBufferBindDepthStencilState", BindDepthStencilState),
    P("nvnCommandBufferBindPolygonState", BindPolygonState),
    P("nvnCommandBufferBindMultisampleState", BindMultisampleState),
    P("nvnCommandBufferSetViewport", SetViewport),
    P("nvnCommandBufferSetViewports", SetViewports),
    P("nvnCommandBufferSetScissor", SetScissor),
    P("nvnCommandBufferSetScissors", SetScissors),
    P("nvnCommandBufferSetDepthRange", SetDepthRange),
    P("nvnCommandBufferSetAlphaRef", SetAlphaRef),
    P("nvnCommandBufferSetBlendColor", SetBlendColor),
    P("nvnCommandBufferSetStencilRef", SetStencilRef),
    P("nvnCommandBufferSetStencilMask", SetStencilMask),
    P("nvnCommandBufferSetStencilValueMask", SetStencilValueMask),
    P("nvnCommandBufferUpdateUniformBuffer", UpdateUniformBuffer),
    P("nvnCommandBufferSetCopyRowStride", SetCopyRowStride),
    P("nvnCommandBufferSetCopyImageStride", SetCopyImageStride),
    P("nvnCommandBufferGetCopyRowStride", GetCopyRowStride),
    P("nvnCommandBufferGetCopyImageStride", GetCopyImageStride),
    P("nvnCommandBufferCopyBufferToTexture", CopyBufferToTexture),
    P("nvnQueueSubmitCommands", QueueSubmitCommands),
};
#undef P

struct Registrar {
    Registrar() {
        for (const Proc& p : kProcs) RegisterNvnProc(p.name, p.fn);
    }
} g_registrar;

}  // namespace

GpuStats& Stats() {
    g_stats.recordings = g_end_count.load();
    g_stats.submit_calls = g_submit_calls.load();
    g_stats.unknown_handles = g_unknown_handles.load();
    return g_stats;
}

Backend& GetBackend() {
    if (g_test_backend) return *g_test_backend;
    std::call_once(g_backend_once, [] {
        g_default_backend.reset(CreatePlatformBackend());
        LOGI(kTag, "graphics backend: %s", g_default_backend->Name());
    });
    return *g_default_backend;
}
void SetBackendForTesting(Backend* b) { g_test_backend = b; }

// ---- null backend (non-Android hosts) -------------------------------------------------------------------
#if !defined(__ANDROID__)
namespace {
class NullBackend final : public Backend {
public:
    void Execute(const CmdList&) override {}
    void Present(const TexRef&) override {}
    const char* Name() const override { return "null"; }
};
}  // namespace
Backend* CreatePlatformBackend() { return new NullBackend; }
#endif

}  // namespace dsr::hle::nvn
