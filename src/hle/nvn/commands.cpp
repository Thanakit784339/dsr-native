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

#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "common/log.h"
#include "hle/nvn.h"
#include "hle/nvn/backend.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "nvn";
constexpr size_t kMaxHandles = 16384;

struct Recorder {
    bool recording = false;
    CmdList cmds;
};
Table<Recorder> g_recorders;

std::mutex g_handle_mutex;
std::unordered_map<uint64_t, std::shared_ptr<const CmdList>> g_handles;
std::deque<uint64_t> g_handle_order;
uint64_t g_next_handle = 1;

GpuStats g_stats;
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
}

// ---- recording ------------------------------------------------------------------------------------------
void BeginRecording(void* cb) {
    Recorder& r = g_recorders.Get(cb);
    r.recording = true;
    r.cmds.clear();
}
uint64_t EndRecording(void* cb) {
    Recorder& r = g_recorders.Get(cb);
    auto list = std::make_shared<const CmdList>(std::move(r.cmds));
    r.cmds = CmdList{};
    r.recording = false;
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

// All draw entry points: counted for now. Arguments are ignored, so one function serves every signature.
void Draw(void* cb) {
    Cmd c;
    c.type = CmdType::Draw;
    Append(cb, std::move(c));
}

// ---- queue ------------------------------------------------------------------------------------------------
void QueueSubmitCommands(void* /*queue*/, int n, const uint64_t* handles) {
    for (int i = 0; i < n; i++) {
        auto list = FindHandle(handles[i]);
        if (!list) continue;
        g_stats.submits++;
        g_stats.commands += list->size();
        for (const Cmd& c : *list) {
            switch (c.type) {
                case CmdType::Draw: g_stats.draws++; break;
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
    P("nvnCommandBufferDrawArrays", Draw),
    P("nvnCommandBufferDrawArraysInstanced", Draw),
    P("nvnCommandBufferDrawArraysIndirect", Draw),
    P("nvnCommandBufferDrawElements", Draw),
    P("nvnCommandBufferDrawElementsBaseVertex", Draw),
    P("nvnCommandBufferDrawElementsInstanced", Draw),
    P("nvnCommandBufferDrawElementsIndirect", Draw),
    P("nvnCommandBufferMultiDrawArraysIndirectCount", Draw),
    P("nvnCommandBufferMultiDrawElementsIndirectCount", Draw),
    P("nvnCommandBufferDrawTexture", Draw),
    P("nvnCommandBufferDrawTransformFeedback", Draw),
    P("nvnQueueSubmitCommands", QueueSubmitCommands),
};
#undef P

struct Registrar {
    Registrar() {
        for (const Proc& p : kProcs) RegisterNvnProc(p.name, p.fn);
    }
} g_registrar;

}  // namespace

GpuStats& Stats() { return g_stats; }

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
