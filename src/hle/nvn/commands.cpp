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

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/log.h"
#include "common/platform.h"
#include "common/profiler.h"
#include "hle/nvn.h"
#include "hle/nvn/backend.h"
#include "hle/nvn/state.h"
#include "runtime/runtime.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "nvn";
// Recent lists kept for submission. ~25 frames: lists are normally submitted right after recording. Lists
// the game submits more than once (static command sets) are kept regardless (up to kMaxKept).
constexpr size_t kMaxHandles = 512;
constexpr size_t kMaxKept = 2048;

// The game checks GetCommandMemoryUsed before submitting and skips buffers that report 0, so report a
// plausible figure: an estimate of what real NVN would have written, reset when memory is (re)added.
constexpr size_t kBytesPerCommand = 32;
constexpr size_t kControlBytesPerRecording = 64;

struct Recorder {
    bool recording = false;
    CmdList cmds;
    size_t last_size = 0;  // commands in this buffer's previous recording (storage reserved up front)
    size_t command_used = 0, control_used = 0;
};
Table<Recorder> g_recorders;

std::mutex g_handle_mutex;
std::unordered_map<uint64_t, std::shared_ptr<const CmdList>> g_handles;
std::deque<uint64_t> g_handle_order;
std::unordered_map<uint64_t, uint32_t> g_submit_counts;  // submissions per live handle
std::deque<uint64_t> g_kept;                               // handles submitted twice or more
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

// Command list storage is recycled: a list that is no longer referenced gives its (cleared) vector back, and
// the next recording starts with that capacity (allocating and freeing several MB of commands per frame,
// page faults included, showed up in profiles).
std::mutex g_list_pool_mutex;
std::vector<CmdList> g_list_pool;
CmdList TakePooledList() {
    std::lock_guard<std::mutex> lk(g_list_pool_mutex);
    if (g_list_pool.empty()) return {};
    CmdList v = std::move(g_list_pool.back());
    g_list_pool.pop_back();
    return v;
}
void ReturnPooledList(CmdList&& v) {
    v.clear();
    if (v.capacity() < 256) return;  // small storage is not worth keeping (recordings reserve what they need)
    std::lock_guard<std::mutex> lk(g_list_pool_mutex);
    if (g_list_pool.size() < 32) g_list_pool.push_back(std::move(v));
}

// Lists are destroyed on a background thread: whichever thread drops the last reference (usually the game's
// submit thread) only queues it. Destroying thousands of commands a frame was over half of that thread's time.
std::mutex g_reap_mutex;
std::condition_variable g_reap_cv;
std::vector<CmdList*> g_reap_queue;
void ReaperLoop() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "dsr-reaper");
#endif
    std::vector<CmdList*> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_reap_mutex);
            g_reap_cv.wait(lk, [] { return !g_reap_queue.empty(); });
            batch.swap(g_reap_queue);
        }
        for (CmdList* l : batch) {
            ReturnPooledList(std::move(*l));
            delete l;
        }
        batch.clear();
    }
}
void ReapList(const CmdList* l) {
#if defined(__ANDROID__)
    static std::once_flag started;
    std::call_once(started, [] { std::thread(ReaperLoop).detach(); });
    {
        std::lock_guard<std::mutex> lk(g_reap_mutex);
        g_reap_queue.push_back(const_cast<CmdList*>(l));
    }
    g_reap_cv.notify_one();
#else
    ReturnPooledList(std::move(*const_cast<CmdList*>(l)));  // tests: synchronously
    delete l;
#endif
}

// The recorder of the command buffer last appended to by this thread (one lookup per command was a locked
// map lookup). Recorders are never erased, so the pointer stays valid.
thread_local const void* t_last_cb = nullptr;
thread_local Recorder* t_last_recorder = nullptr;
Recorder& RecorderOf(void* cb) {
    if (cb != t_last_cb) {
        t_last_recorder = &g_recorders.Get(cb);
        t_last_cb = cb;
    }
    return *t_last_recorder;
}

void Append(void* cb, Cmd&& c) {
    Recorder& r = RecorderOf(cb);
    r.cmds.push_back(std::move(c));
    r.command_used += kBytesPerCommand;
}

// ---- recording ------------------------------------------------------------------------------------------
void BeginRecording(void* cb) {
    dsr::platform::PreferFastCores("game command recording thread");  // the game's MTR_* workers
    Recorder& r = RecorderOf(cb);
    r.recording = true;
    r.cmds.clear();
    if (r.cmds.capacity() == 0) r.cmds = TakePooledList();
    if (r.cmds.capacity() < r.last_size) r.cmds.reserve(r.last_size + r.last_size / 4);  // no growth copies
    r.control_used += kControlBytesPerRecording;
    g_begin_count++;
}
void AddCommandMemory(void* cb, const void*, int64_t, size_t) { RecorderOf(cb).command_used = 0; }
void AddControlMemory(void* cb, void*, size_t) { RecorderOf(cb).control_used = 0; }
// Field reads only: Table::Copy would copy the whole recorder, every command recorded so far included.
size_t GetCommandMemoryUsed(const void* cb) { return RecorderOf(const_cast<void*>(cb)).command_used; }
size_t GetControlMemoryUsed(const void* cb) { return RecorderOf(const_cast<void*>(cb)).control_used; }
uint64_t EndRecording(void* cb) {
    Recorder& r = RecorderOf(cb);
    r.last_size = r.cmds.size();
    // A list much smaller than its (recycled) storage gets storage of its own size: the game keeps hundreds of
    // lists alive, and each holding the capacity of the biggest list ever recorded added up to gigabytes.
    if (r.cmds.capacity() > 2 * r.cmds.size() + 256) {
        CmdList exact;
        exact.reserve(r.cmds.size());
        for (Cmd& c : r.cmds) exact.push_back(std::move(c));
        ReturnPooledList(std::move(r.cmds));
        r.cmds = std::move(exact);
    }
    std::shared_ptr<const CmdList> list(new CmdList(std::move(r.cmds)), ReapList);
    r.cmds = CmdList{};
    r.recording = false;
    g_end_count++;
    std::lock_guard<std::mutex> lk(g_handle_mutex);
    uint64_t h = g_next_handle++;
    g_handles.emplace(h, std::move(list));
    g_handle_order.push_back(h);
    while (g_handle_order.size() > kMaxHandles) {
        const uint64_t old = g_handle_order.front();
        g_handle_order.pop_front();
        auto sc = g_submit_counts.find(old);
        if (sc != g_submit_counts.end() && sc->second >= 2) {
            g_kept.push_back(old);  // reused: keep
            continue;
        }
        if (sc != g_submit_counts.end()) g_submit_counts.erase(sc);
        g_handles.erase(old);
    }
    while (g_kept.size() > kMaxKept) {
        g_handles.erase(g_kept.front());
        g_submit_counts.erase(g_kept.front());
        g_kept.pop_front();
    }
    return h;
}
bool IsRecording(const void* cb) { return RecorderOf(const_cast<void*>(cb)).recording; }

// Previously recorded command lists (CallCommands / CopyCommands behave the same for us) are referenced, not
// copied: the game calls ~20 lists a frame, and copying their commands (with every shared_ptr in them) cost the
// game's render thread a large share of its time.
void CallCommands(void* cb, int n, const uint64_t* handles) {
    Recorder& r = RecorderOf(cb);
    for (int i = 0; i < n; i++)
        if (auto list = FindHandle(handles[i])) {
            Cmd c;
            c.type = CmdType::CallList;
            c.obj = std::move(list);
            r.cmds.push_back(std::move(c));
            r.command_used += kBytesPerCommand;
        }
}

// A render target's view picks the mip level and array layer that is rendered into.
static TexRef TargetRef(const void* texture, const void* view) {
    TexRef r = MakeTexRef(texture);
    if (view) {
        const TextureViewObj v = GetTextureView(view);
        if (v.has_levels) r.level = v.base_level;
        if (v.has_layers) r.layer = v.base_layer;
        if (v.format && v.format != r.tex.format) r.view_format = v.format;  // e.g. rendering through an sRGB view
    }
    return r;
}
void SetRenderTargets(void* cb, int n, const void* const* colors, const void* const* color_views,
                      const void* depth, const void* depth_view) {
    Cmd c;
    c.type = CmdType::SetRenderTargets;
    c.num_rt = n < 0 ? 0 : (n > 8 ? 8 : n);
    auto rts = std::make_shared<std::vector<TexRef>>(c.num_rt + 1);
    for (int i = 0; i < c.num_rt; i++)
        if (colors && colors[i]) (*rts)[i] = TargetRef(colors[i], color_views ? color_views[i] : nullptr);
    if (depth) (*rts)[c.num_rt] = TargetRef(depth, depth_view);
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
    auto ci = std::make_shared<CopyInfo>();
    ci->dst = MakeTexRef(tex);
    if (region) ci->dst_region = *region;
    c.copy = std::move(ci);
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
    auto ci = std::make_shared<CopyInfo>();
    ci->src = MakeTexRef(src);
    ci->dst = MakeTexRef(dst);
    if (src_region) ci->src_region = *src_region;
    if (dst_region) ci->dst_region = *dst_region;
    c.copy = std::move(ci);
    c.flags = flags;
    Append(cb, std::move(c));
}

// ---- binds ----------------------------------------------------------------------------------------------
void Append(void* cb, Cmd&& c);
// A bind: payload items inline when small (no allocation).
template <typename T>
void BindItems(void* cb, CmdType t, const T* items, size_t n, int stage = 0, int index = 0) {
    Cmd c;
    c.type = t;
    SetItems(c, items, n);
    c.stage = stage;
    c.index = index;
    Append(cb, std::move(c));
}
template <typename T>
void BindOne(void* cb, CmdType t, const T& v, int stage = 0, int index = 0) {
    BindItems(cb, t, &v, 1, stage, index);
}
void BindProgram(void* cb, const void* program, int stage_bits) {
    Cmd c;
    c.type = CmdType::BindProgram;
    c.obj = GetProgramCode(program);
    c.index = stage_bits;
    Append(cb, std::move(c));
}
// NVNvertexAttribState is 4 bytes and NVNvertexStreamState 8 bytes in game memory.
void BindVertexAttribState(void* cb, int n, const uint8_t* states) {
    VertexAttribObj v[32];
    n = states ? std::clamp(n, 0, 32) : 0;
    for (int i = 0; i < n; i++) v[i] = GetVertexAttrib(states + 4 * i);
    BindItems(cb, CmdType::BindVertexAttribs, v, size_t(n));
}
void BindVertexStreamState(void* cb, int n, const uint8_t* states) {
    VertexStreamObj v[32];
    n = states ? std::clamp(n, 0, 32) : 0;
    for (int i = 0; i < n; i++) v[i] = GetVertexStream(states + 8 * i);
    BindItems(cb, CmdType::BindVertexStreams, v, size_t(n));
}
void BindVertexBuffer(void* cb, int index, uint64_t addr, uint64_t size) {
    BindOne(cb, CmdType::BindVertexBuffers, BufferRange{addr, size}, 0, index);
}
void BindVertexBuffers(void* cb, int first, int n, const BufferRange* ranges) {
    BindItems(cb, CmdType::BindVertexBuffers, ranges, ranges ? size_t(std::max(0, n)) : 0, 0, first);
}
void BindUniformBuffer(void* cb, int stage, int index, uint64_t addr, uint64_t size) {
    BindOne(cb, CmdType::BindUniformBuffers, BufferRange{addr, size}, stage, index);
}
void BindUniformBuffers(void* cb, int stage, int first, int n, const BufferRange* ranges) {
    BindItems(cb, CmdType::BindUniformBuffers, ranges, ranges ? size_t(std::max(0, n)) : 0, stage, first);
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
    BindOne(cb, CmdType::BindTextures, ResolveHandle(handle), stage, index);
}
void BindTextures(void* cb, int stage, int first, int n, const uint64_t* handles) {
    std::vector<BoundTexture> v;
    for (int i = 0; i < n && handles; i++) v.push_back(ResolveHandle(handles[i]));
    BindItems(cb, CmdType::BindTextures, v.data(), v.size(), stage, first);
}
// Image handles (nvnDeviceGetImageHandle) are the pool ids.
void BindImage(void* cb, int stage, int index, uint64_t handle) {
    BindOne(cb, CmdType::BindImages, ImageById(int(handle & 0xFFFFF)), stage, index);
}
void BindImages(void* cb, int stage, int first, int n, const uint64_t* handles) {
    std::vector<TexRef> v;
    for (int i = 0; i < n && handles; i++) v.push_back(ImageById(int(handles[i] & 0xFFFFF)));
    BindItems(cb, CmdType::BindImages, v.data(), v.size(), stage, first);
}
void BindBlendState(void* cb, const void* s) { BindOne(cb, CmdType::BindBlend, GetBlendState(s)); }
void BindColorState(void* cb, const void* s) { BindOne(cb, CmdType::BindColor, GetColorState(s)); }
void BindChannelMaskState(void* cb, const void* s) { BindOne(cb, CmdType::BindChannelMask, GetChannelMaskState(s)); }
void BindDepthStencilState(void* cb, const void* s) { BindOne(cb, CmdType::BindDepthStencil, GetDepthStencilState(s)); }
void BindPolygonState(void* cb, const void* s) { BindOne(cb, CmdType::BindPolygon, GetPolygonState(s)); }
void BindMultisampleState(void* cb, const void* s) { BindOne(cb, CmdType::BindMultisample, GetMultisampleState(s)); }

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
    SetItems(c, p, size);
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
    auto ci = std::make_shared<CopyInfo>();
    ci->dst = MakeTexRef(dst);
    if (region) ci->dst_region = *region;
    c.copy = std::move(ci);
    c.flags = flags;
    c.f[0] = float(g_copy_row_stride.load());
    c.f[1] = float(g_copy_image_stride.load());
    Append(cb, std::move(c));
}

// GPU -> memory copies. The game reads results back this way (e.g. the luminance that drives auto exposure:
// without it the exposure stayed at its default and the picture was too dark).
void CopyTextureToBuffer(void* cb, const void* src, const void* /*view*/, const CopyRegion* region, uint64_t dst,
                         int flags) {
    Cmd c;
    c.type = CmdType::CopyTextureToBuffer;
    auto ci = std::make_shared<CopyInfo>();
    ci->src = MakeTexRef(src);
    if (region) ci->src_region = *region;
    c.copy = std::move(ci);
    c.dst_addr = dst;
    c.flags = flags;
    c.f[0] = float(g_copy_row_stride.load());
    c.f[1] = float(g_copy_image_stride.load());
    Append(cb, std::move(c));
}
void CopyBufferToBuffer(void* cb, uint64_t src, uint64_t dst, size_t size, int flags) {
    Cmd c;
    c.type = CmdType::CopyBufferToBuffer;
    c.src_addr = src;
    c.dst_addr = dst;
    c.size64 = size;
    c.flags = flags;
    Append(cb, std::move(c));
}
// Reports: written to game memory once the GPU has executed everything before them. Engines use them as
// lightweight fences ("the GPU is past this point, its results can be read").
void ReportCounter(void* cb, int counter, uint64_t dst) {
    Cmd c;
    c.type = CmdType::Report;
    c.index = 1;
    c.mask = counter;
    c.dst_addr = dst;
    Append(cb, std::move(c));
    static std::atomic<int> n{0};
    if (n++ < 4) LOGI(kTag, "nvnCommandBufferReportCounter(counter %d) -> @%llx", counter, (unsigned long long)dst);
}
void ReportValue(void* cb, uint32_t value, uint64_t dst) {
    Cmd c;
    c.type = CmdType::Report;
    c.index = 0;
    c.mask = int(value);
    c.dst_addr = dst;
    Append(cb, std::move(c));
    static std::atomic<int> n{0};
    if (n++ < 4) LOGI(kTag, "nvnCommandBufferReportValue(%u) -> @%llx", value, (unsigned long long)dst);
}

// ---- queue ------------------------------------------------------------------------------------------------
// ---- render thread --------------------------------------------------------------------------------------------
class GpuQueue {
public:
    uint64_t Enqueue(std::function<void()> task) {
        if (!Threaded()) {
            task();
            std::lock_guard<std::mutex> lk(m_);
            completed_ = ++submitted_;
            return submitted_;
        }
        std::unique_lock<std::mutex> lk(m_);
        if (!thread_.joinable()) thread_ = std::thread([this] { Run(); });
        tasks_.push_back(std::move(task));
        const uint64_t seq = ++submitted_;
        cv_.notify_all();
        return seq;
    }
    // Waits for task `seq`; false on timeout.
    bool Wait(uint64_t seq, std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max()) {
        std::unique_lock<std::mutex> lk(m_);
        auto done = [&] { return completed_ >= seq; };
        if (timeout == std::chrono::nanoseconds::max()) {
            done_cv_.wait(lk, done);
            return true;
        }
        return done_cv_.wait_for(lk, timeout, done);
    }
    uint64_t Completed() {
        std::lock_guard<std::mutex> lk(m_);
        return completed_;
    }
    uint64_t Submitted() {
        std::lock_guard<std::mutex> lk(m_);
        return submitted_;
    }

private:
    static bool Threaded() {
#if defined(__ANDROID__)
        return g_test_backend == nullptr;
#else
        return false;
#endif
    }
    // The render thread is the frame's critical path (all of the game's draws are turned into Vulkan commands
    // here): it is kept on the fastest cores (highest maximum clock; the prime core on Snapdragon 8 Gen 2).
    static void PinToFastestCores() {
#if defined(__linux__)
        if (!dsr::GetRuntime().config.pin_render_thread) return;
        long best = 0;
        std::vector<int> cpus;
        for (int cpu = 0; cpu < 16; cpu++) {
            char path[96];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            FILE* f = fopen(path, "r");
            if (!f) continue;
            long khz = 0;
            if (fscanf(f, "%ld", &khz) != 1) khz = 0;
            fclose(f);
            if (khz > best) {
                best = khz;
                cpus.clear();
            }
            if (khz == best && khz > 0) cpus.push_back(cpu);
        }
        if (cpus.empty()) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        std::string list;
        for (int c : cpus) {
            CPU_SET(c, &set);
            list += " " + std::to_string(c);
        }
        const int r = sched_setaffinity(0, sizeof(set), &set);
        LOGI(kTag, "render thread pinned to the fastest CPU(s)%s (%ld MHz): %s", list.c_str(), best / 1000,
             r == 0 ? "ok" : "refused");
#endif
    }
    void Run() {
        dsr::profiler::RegisterCurrentThread("render (Vulkan)");
        PinToFastestCores();
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !tasks_.empty(); });
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
            {
                std::lock_guard<std::mutex> lk(m_);
                completed_++;
            }
            done_cv_.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::deque<std::function<void()>> tasks_;
    uint64_t submitted_ = 0, completed_ = 0;
    std::thread thread_;
};
GpuQueue& Queue() {
    static GpuQueue* q = new GpuQueue;  // never destroyed: the render thread lives as long as the process
    return *q;
}

// NVNsync objects: the queue position after which each is signalled (absent or 0: signalled).
std::mutex g_sync_mutex;
std::unordered_map<const void*, uint64_t> g_sync_targets;

void FenceSync(void* cb, void* sync, int /*condition*/, int /*flags*/) {
    Cmd c;
    c.type = CmdType::FenceSync;
    c.src_addr = reinterpret_cast<uint64_t>(sync);
    Append(cb, std::move(c));
}
bool SyncInitialize(void* sync, void* /*device*/) {
    std::lock_guard<std::mutex> lk(g_sync_mutex);
    g_sync_targets.erase(sync);
    return true;
}
void SyncFinalize(void* sync) {
    std::lock_guard<std::mutex> lk(g_sync_mutex);
    g_sync_targets.erase(sync);
}
// nvnQueueFenceSync: signalled once everything submitted so far has run. (It was a no-op, leaving the sync
// signalled: the game reused per-frame memory, such as skinning matrices, while the render thread had yet to
// read it, so a character's depth prepass and colour pass could see different poses.)
std::atomic<uint64_t> g_queue_fences{0};
void QueueFenceSync(void* /*queue*/, void* sync, int /*condition*/, int /*flags*/) {
    if (!sync) return;
    if (g_queue_fences++ == 0) LOGI(kTag, "nvnQueueFenceSync used: queue syncs now wait for the render thread");
    SetSyncTarget(sync, Queue().Submitted());
}
// NVNsyncWaitResult: 0 already signalled, 1 condition satisfied, 2 timeout expired.
// Time game threads spend waiting for the render thread: [0] at present (frames queued ahead), [1] in
// nvnSyncWait / nvnQueueFinish. Reported with the frame times: shows whether the game or the render side sets
// the frame rate.
std::atomic<uint64_t> g_game_wait_ns[2];
struct WaitTimer {
    int which;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~WaitTimer() {
        g_game_wait_ns[which] += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - t0).count());
    }
};
int SyncWait(const void* sync, uint64_t timeout_ns) {
    uint64_t target = 0;
    {
        std::lock_guard<std::mutex> lk(g_sync_mutex);
        auto it = g_sync_targets.find(sync);
        if (it != g_sync_targets.end()) target = it->second;
    }
    static std::atomic<int> logged{0};
    if (logged < 6 && logged++ < 6)
        LOGI(kTag, "nvnSyncWait(%p): %s", sync, !target ? "no pending work" : Queue().Completed() >= target ? "already done" : "waits");
    if (!target || Queue().Completed() >= target) return 0;
    WaitTimer wt{1};
    const auto timeout = timeout_ns >= uint64_t(INT64_MAX) ? std::chrono::nanoseconds::max()
                                                           : std::chrono::nanoseconds(int64_t(timeout_ns));
    return Queue().Wait(target, timeout) ? 1 : 2;
}
void QueueFinish(void* /*queue*/) {
    WaitTimer wt{1};
    Queue().Wait(Queue().Submitted());
}

void QueueSubmitCommands(void* /*queue*/, int n, const uint64_t* handles) {
    dsr::platform::PreferFastCores("game submit thread");
    g_submit_calls++;
    for (int i = 0; i < n; i++) {
        auto list = FindHandle(handles[i]);
        if (!list) {
            g_unknown_handles++;
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(g_handle_mutex);
            g_submit_counts[handles[i]]++;
        }
        g_stats.submits++;
        std::vector<const void*> fences;
        ForEachCmd(*list, [&](const Cmd& c) {
            g_stats.commands++;
            if (c.type == CmdType::FenceSync) fences.push_back(reinterpret_cast<const void*>(c.src_addr));
            switch (c.type) {
                case CmdType::Draw: g_stats.draws++; break;
                default: break;
                case CmdType::ClearColor:
                case CmdType::ClearDepthStencil:
                case CmdType::ClearTexture: g_stats.clears++; break;
                case CmdType::CopyTexture: g_stats.copies++; break;
                case CmdType::SetRenderTargets: g_stats.rt_binds++; break;
            }
        });
        const uint64_t seq = Queue().Enqueue([list] { GetBackend().Execute(*list); });
        if (!fences.empty()) {
            std::lock_guard<std::mutex> lk(g_sync_mutex);
            for (const void* s : fences) g_sync_targets[s] = seq;
        }
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
    P("nvnCommandBufferCopyTextureToBuffer", CopyTextureToBuffer),
    P("nvnCommandBufferCopyBufferToBuffer", CopyBufferToBuffer),
    P("nvnCommandBufferReportCounter", ReportCounter),
    P("nvnCommandBufferReportValue", ReportValue),
    P("nvnQueueSubmitCommands", QueueSubmitCommands),
    P("nvnCommandBufferFenceSync", FenceSync), P("nvnQueueFinish", QueueFinish), P("nvnQueueFenceSync", QueueFenceSync),
    P("nvnSyncInitialize", SyncInitialize), P("nvnSyncFinalize", SyncFinalize), P("nvnSyncWait", SyncWait),
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

void SetSyncTarget(const void* sync, uint64_t seq) {
    std::lock_guard<std::mutex> lk(g_sync_mutex);
    if (seq && Queue().Completed() < seq)
        g_sync_targets[sync] = seq;
    else
        g_sync_targets.erase(sync);
}
uint64_t GpuEnqueue(std::function<void()> task) { return Queue().Enqueue(std::move(task)); }
void GpuWait(uint64_t seq) {
    if (Queue().Completed() >= seq) return;
    WaitTimer wt{0};
    Queue().Wait(seq);
}
void TakeGameWaits(double* present_ms, double* sync_ms) {
    *present_ms = g_game_wait_ns[0].exchange(0) / 1e6;
    *sync_ms = g_game_wait_ns[1].exchange(0) / 1e6;
}
uint64_t GpuSubmitted() { return Queue().Submitted(); }

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
