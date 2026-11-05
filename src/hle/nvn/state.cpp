// NVN state-object setters and lookups. NVN state objects live in game memory and are configured with
// setter calls; we mirror each object's settings in a table keyed by its address. Values are logged the
// first time each distinct setting is seen, which is how the NVN enum mappings are confirmed on device.

#include "hle/nvn/state.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include "common/log.h"
#include "hle/nvn.h"
#include "hle/nvn/backend.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "nvn.state";

Table<BlendStateObj> g_blend;
Table<TextureViewObj> g_views;
Table<ColorStateObj> g_color;
Table<ChannelMaskStateObj> g_channel;
Table<DepthStencilStateObj> g_depth;
Table<PolygonStateObj> g_polygon;
Table<MultisampleStateObj> g_ms;
Table<SamplerObj> g_sampler_builders, g_samplers;

std::mutex g_pool_mutex;
std::unordered_map<int, TexRef> g_texture_ids;
std::unordered_map<int, TexRef> g_image_ids;  // nvnTexturePoolRegisterImage (storage images)
std::unordered_map<int, SamplerObj> g_sampler_ids;
std::unordered_map<const void*, std::shared_ptr<const ShaderCode>> g_programs;
std::atomic<int> g_depth_mode{0}, g_origin_mode{0};

// Logs each distinct (what, value) once.
void Seen(const char* what, int a, int b = 0, int c = 0, int d = 0) {
    // Called by setters on every use: repeats are filtered per thread first (no lock, no strings).
    thread_local std::unordered_set<uint64_t> recent;
    const uint64_t key = ((reinterpret_cast<uintptr_t>(what) * 0x9E3779B97F4A7C15ull ^ uint64_t(uint32_t(a))) *
                              0x100000001B3ull ^
                          uint64_t(uint32_t(b)) << 32 ^ uint64_t(uint32_t(c)) << 16 ^ uint64_t(uint32_t(d)));
    if (recent.size() > 4096 || !recent.insert(key).second) return;
    static std::mutex m;
    static std::set<std::tuple<std::string, int, int, int, int>> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen.size() < 400 && seen.insert({what, a, b, c, d}).second)
        LOGI(kTag, "%s: %d %d %d %d", what, a, b, c, d);
}

// ---- blend ----------------------------------------------------------------------------------------------
void BlendSetDefaults(void* s) { g_blend.Reset(s); }
void BlendSetTarget(void* s, int t) { g_blend.Get(s).target = t; }
void BlendSetFunc(void* s, int src_rgb, int dst_rgb, int src_a, int dst_a) {
    auto& b = g_blend.Get(s);
    b.src_rgb = src_rgb;
    b.dst_rgb = dst_rgb;
    b.src_a = src_a;
    b.dst_a = dst_a;
    Seen("blend func", src_rgb, dst_rgb, src_a, dst_a);
}
void BlendSetEquation(void* s, int rgb, int a) {
    auto& b = g_blend.Get(s);
    b.eq_rgb = rgb;
    b.eq_a = a;
    Seen("blend equation", rgb, a);
}
// ---- color ----------------------------------------------------------------------------------------------
void ColorSetDefaults(void* s) { g_color.Reset(s); }
void ColorSetBlendEnable(void* s, int index, bool enable) {
    auto& c = g_color.Get(s);
    if (index < 0 || index >= 8) return;
    if (enable)
        c.blend_enable |= 1u << index;
    else
        c.blend_enable &= ~(1u << index);
}
void ColorSetLogicOp(void* s, int op) { g_color.Get(s).logic_op = op; }
void ColorSetAlphaTest(void* s, int func) {
    g_color.Get(s).alpha_func = func;
    Seen("alpha test", func);
}
// ---- channel mask --------------------------------------------------------------------------------------
void ChannelSetDefaults(void* s) { g_channel.Reset(s); }
void ChannelSetMask(void* s, int index, bool r, bool g, bool b, bool a) {
    if (index < 0 || index >= 8) return;
    auto& c = g_channel.Get(s);
    uint32_t bits = (r ? 1u : 0) | (g ? 2u : 0) | (b ? 4u : 0) | (a ? 8u : 0);
    c.mask = (c.mask & ~(0xFu << (4 * index))) | (bits << (4 * index));
}
// ---- depth / stencil -----------------------------------------------------------------------------------
void DepthSetDefaults(void* s) { g_depth.Reset(s); }
void DepthSetTest(void* s, bool e) { g_depth.Get(s).depth_test = e; }
void DepthSetWrite(void* s, bool e) { g_depth.Get(s).depth_write = e; }
void DepthSetFunc(void* s, int f) {
    g_depth.Get(s).depth_func = f;
    Seen("depth func", f);
}
void StencilSetTest(void* s, bool e) { g_depth.Get(s).stencil_test = e; }
// face: 1 FRONT, 2 BACK, 3 FRONT_AND_BACK
void StencilSetFunc(void* s, int face, int func) {
    auto& d = g_depth.Get(s);
    if (face & 1) d.stencil_func[0] = func;
    if (face & 2) d.stencil_func[1] = func;
    Seen("stencil func", face, func);
}
void StencilSetOp(void* s, int face, int fail, int dfail, int dpass) {
    auto& d = g_depth.Get(s);
    for (int i = 0; i < 2; i++)
        if (face & (1 << i)) {
            d.stencil_fail[i] = fail;
            d.depth_fail[i] = dfail;
            d.depth_pass[i] = dpass;
        }
    Seen("stencil op", face, fail, dfail, dpass);
}
// ---- polygon -------------------------------------------------------------------------------------------
void PolygonSetDefaults(void* s) { g_polygon.Reset(s); }
void PolygonSetCull(void* s, int face) {
    g_polygon.Get(s).cull = face;
    Seen("cull face", face);
}
void PolygonSetFront(void* s, int ff) {
    g_polygon.Get(s).front_face = ff;
    Seen("front face", ff);
}
void PolygonSetMode(void* s, int m) { g_polygon.Get(s).mode = m; }
void PolygonSetOffsetEnables(void* s, int e) { g_polygon.Get(s).offset_enables = e; }
// ---- multisample ---------------------------------------------------------------------------------------
void MsSetDefaults(void* s) { g_ms.Reset(s); }
void MsSetEnable(void* s, bool e) { g_ms.Get(s).enable = e; }
void MsSetSamples(void* s, int n) { g_ms.Get(s).samples = n; }
// ---- vertex layout -------------------------------------------------------------------------------------
// NVNvertexAttribState (4 bytes) and NVNvertexStreamState (8 bytes) keep their values in game memory, as the
// real driver does: arrays of them are bound per draw, and a locked table lookup per element (contended by the
// game's three recording threads) was ~15% of those threads' time.
// Attribute word: bits 0-9 format, 10-15 stream, 16-31 relative offset.
uint32_t LoadWord(const void* s) {
    uint32_t w;
    memcpy(&w, s, 4);
    return w;
}
void StoreWord(void* s, uint32_t w) { memcpy(s, &w, 4); }
void AttribSetDefaults(void* s) { StoreWord(s, 0); }
void AttribSetFormat(void* s, int format, int64_t offset) {
    if (format < 0 || format > 0x3FF || offset < 0 || offset > 0xFFFF)
        LOGW(kTag, "vertex attribute format %d / offset %lld out of the packed range", format, (long long)offset);
    StoreWord(s, (LoadWord(s) & 0xFC00u) | (uint32_t(format) & 0x3FFu) | (uint32_t(offset & 0xFFFF) << 16));
    Seen("vertex format", format);
}
void AttribSetStream(void* s, int stream) { StoreWord(s, (LoadWord(s) & ~0xFC00u) | ((uint32_t(stream) & 63u) << 10)); }
int AttribGetFormat(const void* s) { return int(LoadWord(s) & 0x3FFu); }
int AttribGetStream(const void* s) { return int(LoadWord(s) >> 10 & 63u); }
void ViewSetDefaults(void* v) { g_views.Reset(v); }
void ViewSetLevels(void* v, int base, int num) {
    auto& o = g_views.Get(v);
    o.base_level = base;
    o.num_levels = num;
    o.has_levels = true;
}
void ViewSetLayers(void* v, int base, int num) {
    auto& o = g_views.Get(v);
    o.base_layer = base;
    o.num_layers = num;
    o.has_layers = true;
}
void ViewSetTarget(void* v, int target) { g_views.Get(v).target = target; }
void ViewSetFormat(void* v, int format) {
    g_views.Get(v).format = format;
    Seen("texture view format", format);
}
int ViewGetFormat(const void* v) { return g_views.Copy(v).format; }
// Stream state: stride, then divisor (two 32-bit words).
void StreamSetDefaults(void* s) { memset(s, 0, 8); }
void StreamSetStride(void* s, int64_t stride) { StoreWord(s, uint32_t(int32_t(stride))); }
void StreamSetDivisor(void* s, int d) { StoreWord(static_cast<uint8_t*>(s) + 4, uint32_t(d)); }
int64_t StreamGetStride(const void* s) { return int32_t(LoadWord(s)); }
int StreamGetDivisor(const void* s) { return int32_t(LoadWord(static_cast<const uint8_t*>(s) + 4)); }
// ---- samplers ------------------------------------------------------------------------------------------
void SamplerBuilderSetDefaults(void* b) { g_sampler_builders.Reset(b); }
void SamplerBuilderSetMinMag(void* b, int min, int mag) {
    auto& s = g_sampler_builders.Get(b);
    s.min_filter = min;
    s.mag_filter = mag;
    Seen("sampler filter", min, mag);
}
void SamplerBuilderSetWrap(void* b, int s_, int t, int r) {
    auto& s = g_sampler_builders.Get(b);
    s.wrap[0] = s_;
    s.wrap[1] = t;
    s.wrap[2] = r;
    Seen("sampler wrap", s_, t, r);
}
void SamplerBuilderSetLodClamp(void* b, float mn, float mx) {
    auto& s = g_sampler_builders.Get(b);
    s.lod_min = mn;
    s.lod_max = mx;
}
void SamplerBuilderSetLodBias(void* b, float bias) { g_sampler_builders.Get(b).lod_bias = bias; }
void SamplerBuilderSetCompare(void* b, int mode, int func) {
    auto& s = g_sampler_builders.Get(b);
    s.compare_mode = mode;
    s.compare_func = func;
    Seen("sampler compare", mode, func);
}
void SamplerBuilderSetAniso(void* b, float a) { g_sampler_builders.Get(b).max_anisotropy = a; }
bool SamplerInitialize(void* sampler, const void* builder) {
    g_samplers.Get(sampler) = g_sampler_builders.Copy(builder);
    return true;
}
void SamplerFinalize(void* sampler) { g_samplers.Erase(sampler); }
// ---- pools ---------------------------------------------------------------------------------------------
void TexturePoolRegisterTexture(void* pool, int id, const void* texture, const void* view) {
    TexRef r = MakeTexRef(texture);
    if (view) {  // a view registered with the texture limits what shaders see (one mip, a layer range)
        const TextureViewObj v = g_views.Copy(view);
        if (v.has_levels) {
            r.level = v.base_level;
            r.view_levels = v.num_levels;
        }
        if (v.has_layers) {
            r.layer = v.base_layer;
            r.view_layers = v.num_layers;
        }
        if (v.format && v.format != r.tex.format) r.view_format = v.format;
    }
    bool changed;
    {
        std::lock_guard<std::mutex> lk(g_pool_mutex);
        auto it = g_texture_ids.find(id);
        changed = it == g_texture_ids.end() || it->second.addr != r.addr || it->second.tex.format != r.tex.format ||
                  it->second.tex.width != r.tex.width || it->second.tex.height != r.tex.height ||
                  it->second.view_format != r.view_format;
        g_texture_ids[id] = r;
    }
    // A texture (re)registered under this id: its memory may hold a newly streamed-in texture (the game reuses
    // texture memory). Check it at its next use instead of waiting for the next scheduled check.
    if (changed && r.valid) GetBackend().MarkTextureDirty(r.addr);
}
void TexturePoolRegisterImage(void* pool, int id, const void* texture, const void* view) {
    TexRef r = MakeTexRef(texture);
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    static int logged = 0;
    if (logged++ < 8)
        LOGI(kTag, "image %d registered: %dx%dx%d format %d target %d", id, r.tex.width, r.tex.height, r.tex.depth,
             r.tex.format, r.tex.target);
    g_image_ids[id] = r;
}
void SamplerPoolRegisterSampler(void* pool, int id, const void* sampler) {
    SamplerObj s = g_samplers.Copy(sampler);
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_sampler_ids[id] = s;
}
void SamplerPoolRegisterBuilder(void* pool, int id, const void* builder) {
    SamplerObj s = g_sampler_builders.Copy(builder);
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_sampler_ids[id] = s;
}
// ---- device modes --------------------------------------------------------------------------------------
void DeviceSetDepthMode(void* dev, int mode) {
    g_depth_mode = mode;
    LOGI(kTag, "depth mode %d (%s)", mode, mode ? "near is zero" : "OpenGL -1..1");
}
void DeviceSetWindowOriginMode(void* dev, int mode) {
    g_origin_mode = mode;
    LOGI(kTag, "window origin mode %d (%s)", mode, mode ? "upper left" : "lower left");
}

struct Proc {
    const char* name;
    void* fn;
};
#define P(n, f) {n, reinterpret_cast<void*>(&f)}
const Proc kProcs[] = {
    P("nvnBlendStateSetDefaults", BlendSetDefaults), P("nvnBlendStateSetBlendTarget", BlendSetTarget),
    P("nvnBlendStateSetBlendFunc", BlendSetFunc), P("nvnBlendStateSetBlendEquation", BlendSetEquation),
    P("nvnColorStateSetDefaults", ColorSetDefaults), P("nvnColorStateSetBlendEnable", ColorSetBlendEnable),
    P("nvnColorStateSetLogicOp", ColorSetLogicOp), P("nvnColorStateSetAlphaTest", ColorSetAlphaTest),
    P("nvnChannelMaskStateSetDefaults", ChannelSetDefaults), P("nvnChannelMaskStateSetChannelMask", ChannelSetMask),
    P("nvnDepthStencilStateSetDefaults", DepthSetDefaults), P("nvnDepthStencilStateSetDepthTestEnable", DepthSetTest),
    P("nvnDepthStencilStateSetDepthWriteEnable", DepthSetWrite), P("nvnDepthStencilStateSetDepthFunc", DepthSetFunc),
    P("nvnDepthStencilStateSetStencilTestEnable", StencilSetTest), P("nvnDepthStencilStateSetStencilFunc", StencilSetFunc),
    P("nvnDepthStencilStateSetStencilOp", StencilSetOp),
    P("nvnPolygonStateSetDefaults", PolygonSetDefaults), P("nvnPolygonStateSetCullFace", PolygonSetCull),
    P("nvnPolygonStateSetFrontFace", PolygonSetFront), P("nvnPolygonStateSetPolygonMode", PolygonSetMode),
    P("nvnPolygonStateSetPolygonOffsetEnables", PolygonSetOffsetEnables),
    P("nvnMultisampleStateSetDefaults", MsSetDefaults), P("nvnMultisampleStateSetMultisampleEnable", MsSetEnable),
    P("nvnMultisampleStateSetSamples", MsSetSamples),
    P("nvnVertexAttribStateSetDefaults", AttribSetDefaults), P("nvnVertexAttribStateSetFormat", AttribSetFormat),
    P("nvnVertexAttribStateSetStreamIndex", AttribSetStream), P("nvnVertexAttribStateGetFormat", AttribGetFormat),
    P("nvnVertexAttribStateGetStreamIndex", AttribGetStream),
    P("nvnTextureViewSetDefaults", ViewSetDefaults), P("nvnTextureViewSetLevels", ViewSetLevels),
    P("nvnTextureViewSetLayers", ViewSetLayers), P("nvnTextureViewSetTarget", ViewSetTarget),
    P("nvnTextureViewSetFormat", ViewSetFormat), P("nvnTextureViewGetFormat", ViewGetFormat),
    P("nvnVertexStreamStateSetDefaults", StreamSetDefaults), P("nvnVertexStreamStateSetStride", StreamSetStride),
    P("nvnVertexStreamStateSetDivisor", StreamSetDivisor), P("nvnVertexStreamStateGetStride", StreamGetStride),
    P("nvnVertexStreamStateGetDivisor", StreamGetDivisor),
    P("nvnSamplerBuilderSetDefaults", SamplerBuilderSetDefaults), P("nvnSamplerBuilderSetMinMagFilter", SamplerBuilderSetMinMag),
    P("nvnSamplerBuilderSetWrapMode", SamplerBuilderSetWrap), P("nvnSamplerBuilderSetLodClamp", SamplerBuilderSetLodClamp),
    P("nvnSamplerBuilderSetLodBias", SamplerBuilderSetLodBias), P("nvnSamplerBuilderSetCompare", SamplerBuilderSetCompare),
    P("nvnSamplerBuilderSetMaxAnisotropy", SamplerBuilderSetAniso),
    P("nvnSamplerInitialize", SamplerInitialize), P("nvnSamplerFinalize", SamplerFinalize),
    P("nvnTexturePoolRegisterTexture", TexturePoolRegisterTexture),
    P("nvnTexturePoolRegisterImage", TexturePoolRegisterImage),
    P("nvnSamplerPoolRegisterSampler", SamplerPoolRegisterSampler),
    P("nvnSamplerPoolRegisterSamplerBuilder", SamplerPoolRegisterBuilder),
    P("nvnDeviceSetDepthMode", DeviceSetDepthMode), P("nvnDeviceSetWindowOriginMode", DeviceSetWindowOriginMode),
};
#undef P

struct Registrar {
    Registrar() {
        for (const Proc& p : kProcs) RegisterNvnProc(p.name, p.fn);
    }
} g_registrar;
}  // namespace

BlendStateObj GetBlendState(const void* o) { return g_blend.Copy(o); }
ColorStateObj GetColorState(const void* o) { return g_color.Copy(o); }
ChannelMaskStateObj GetChannelMaskState(const void* o) { return g_channel.Copy(o); }
TextureViewObj GetTextureView(const void* o) { return g_views.Copy(o); }
DepthStencilStateObj GetDepthStencilState(const void* o) { return g_depth.Copy(o); }
PolygonStateObj GetPolygonState(const void* o) { return g_polygon.Copy(o); }
MultisampleStateObj GetMultisampleState(const void* o) { return g_ms.Copy(o); }
VertexAttribObj GetVertexAttrib(const void* o) {
    const uint32_t w = LoadWord(o);
    return {int(w & 0x3FFu), int(w >> 16), int(w >> 10 & 63u)};
}
VertexStreamObj GetVertexStream(const void* o) {
    return {int(int32_t(LoadWord(o))), int(int32_t(LoadWord(static_cast<const uint8_t*>(o) + 4)))};
}

TexRef TextureById(int id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it = g_texture_ids.find(id);
    return it == g_texture_ids.end() ? TexRef{} : it->second;
}
TexRef ImageById(int id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it = g_image_ids.find(id);
    if (it != g_image_ids.end()) return it->second;
    // This game never registers images: its image handles are ids of textures registered as textures.
    auto t = g_texture_ids.find(id);
    return t == g_texture_ids.end() ? TexRef{} : t->second;
}
SamplerObj SamplerById(int id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it = g_sampler_ids.find(id);
    return it == g_sampler_ids.end() ? SamplerObj{} : it->second;
}
void SetProgramCode(const void* program, std::shared_ptr<const ShaderCode> code) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_programs[program] = std::move(code);
}
std::shared_ptr<const ShaderCode> GetProgramCode(const void* program) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it = g_programs.find(program);
    return it == g_programs.end() ? nullptr : it->second;
}
DeviceModes GetDeviceModes() { return {g_depth_mode.load(), g_origin_mode.load()}; }

}  // namespace dsr::hle::nvn
