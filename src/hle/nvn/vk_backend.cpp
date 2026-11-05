// Vulkan backend (Android).
//
// Executes recorded NVN command lists on the device GPU:
//   - render targets, clears and texture copies on real GPU images (keyed by guest storage address);
//   - draws: Maxwell shaders translated to SPIR-V (shader/translator.cpp), pipelines built from the
//     mirrored NVN state, vertex/index/uniform data copied from game memory into a streaming buffer,
//     textures deswizzled from block-linear game memory and uploaded on first use;
//   - presentation of the window texture to a swapchain on the device screen.
//
// Execution is deliberately simple: each submitted command list is recorded into one Vulkan command
// buffer, submitted and waited on. NVN syncs are therefore always already signaled, and the streaming
// buffer and descriptor pool can be reset at the start of every list. Dynamic rendering (Vulkan 1.3)
// avoids render-pass objects entirely.
#if defined(__ANDROID__)

#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <deque>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#include "common/log.h"
#include "common/platform.h"
#include "hle/nvn/backend.h"
#include "runtime/runtime.h"
#include "hle/nvn/layout.h"
#include "hle/nvn/shader/translator.h"
#include "hle/nvn/state.h"
#include "hle/video.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "vulkan";
constexpr VkDeviceSize kStreamSize = 48ull << 20;  // per slot (four slots)
constexpr VkDeviceSize kMaxUniformRange = 65536;
constexpr uint32_t kCbufBase = 3;      // NVN uniform-buffer binding b is read by shaders as c[b + 3]
constexpr uint32_t kCompilerConstants = 1;  // c[1]: literal constants from the shader's control section

bool IsDepthFormat(int f) { return f >= 49 && f <= 54; }
bool HasDepth(int f) { return f >= 50 && f <= 54; }
bool HasStencil(int f) { return f == 49 || f == 53 || f == 54; }
bool IsCompressed(int f) { return f >= 65 && f <= 80; }
bool IgnoresAlpha(int f) { return f == 55 || (f >= 109 && f <= 120); }  // RGBX formats
VkImageAspectFlags AspectOf(int f) {
    if (!IsDepthFormat(f)) return VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageAspectFlags a = 0;
    if (HasDepth(f)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (HasStencil(f)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a;
}

// ---- format tables ------------------------------------------------------------------------------------------
// NVN format -> candidate Vulkan formats for images, best first.
std::vector<VkFormat> ImageCandidates(int f) {
    switch (f) {
        case 1: return {VK_FORMAT_R8_UNORM};
        case 2: return {VK_FORMAT_R8_SNORM};
        case 3: return {VK_FORMAT_R8_UINT};
        case 4: return {VK_FORMAT_R8_SINT};
        case 5: return {VK_FORMAT_R16_SFLOAT};
        case 6: return {VK_FORMAT_R16_UNORM};
        case 7: return {VK_FORMAT_R16_SNORM};
        case 8: return {VK_FORMAT_R16_UINT};
        case 9: return {VK_FORMAT_R16_SINT};
        case 10: return {VK_FORMAT_R32_SFLOAT};
        case 11: return {VK_FORMAT_R32_UINT};
        case 12: return {VK_FORMAT_R32_SINT};
        case 13: return {VK_FORMAT_R8G8_UNORM};
        case 14: return {VK_FORMAT_R8G8_SNORM};
        case 15: return {VK_FORMAT_R8G8_UINT};
        case 16: return {VK_FORMAT_R8G8_SINT};
        case 17: return {VK_FORMAT_R16G16_SFLOAT};
        case 18: return {VK_FORMAT_R16G16_UNORM};
        case 19: return {VK_FORMAT_R16G16_SNORM};
        case 20: return {VK_FORMAT_R16G16_UINT};
        case 21: return {VK_FORMAT_R16G16_SINT};
        case 22: return {VK_FORMAT_R32G32_SFLOAT};
        case 23: return {VK_FORMAT_R32G32_UINT};
        case 24: return {VK_FORMAT_R32G32_SINT};
        case 37: return {VK_FORMAT_R8G8B8A8_UNORM};
        case 38: return {VK_FORMAT_R8G8B8A8_SNORM};
        case 39: return {VK_FORMAT_R8G8B8A8_UINT};
        case 40: return {VK_FORMAT_R8G8B8A8_SINT};
        case 41: return {VK_FORMAT_R16G16B16A16_SFLOAT};
        case 42: return {VK_FORMAT_R16G16B16A16_UNORM};
        case 43: return {VK_FORMAT_R16G16B16A16_SNORM};
        case 44: return {VK_FORMAT_R16G16B16A16_UINT};
        case 45: return {VK_FORMAT_R16G16B16A16_SINT};
        case 46: return {VK_FORMAT_R32G32B32A32_SFLOAT};
        case 47: return {VK_FORMAT_R32G32B32A32_UINT};
        case 48: return {VK_FORMAT_R32G32B32A32_SINT};
        case 49: return {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        case 50: return {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT};
        case 51: return {VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
        case 52: return {VK_FORMAT_D32_SFLOAT};
        case 53: return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        case 54: return {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
        case 55:
        case 56: return {VK_FORMAT_R8G8B8A8_SRGB};
        case 57: return {VK_FORMAT_R4G4B4A4_UNORM_PACK16};
        case 58:
        case 59: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16};
        case 60: return {VK_FORMAT_R5G6B5_UNORM_PACK16};
        case 61: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32};
        case 62: return {VK_FORMAT_A2B10G10R10_UINT_PACK32};
        case 63: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32};
        case 64: return {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
        case 65: return {VK_FORMAT_BC1_RGB_UNORM_BLOCK};
        case 66: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK};
        case 67: return {VK_FORMAT_BC2_UNORM_BLOCK};
        case 68: return {VK_FORMAT_BC3_UNORM_BLOCK};
        case 69: return {VK_FORMAT_BC1_RGB_SRGB_BLOCK};
        case 70: return {VK_FORMAT_BC1_RGBA_SRGB_BLOCK};
        case 71: return {VK_FORMAT_BC2_SRGB_BLOCK};
        case 72: return {VK_FORMAT_BC3_SRGB_BLOCK};
        case 73: return {VK_FORMAT_BC4_UNORM_BLOCK};
        case 74: return {VK_FORMAT_BC4_SNORM_BLOCK};
        case 75: return {VK_FORMAT_BC5_UNORM_BLOCK};
        case 76: return {VK_FORMAT_BC5_SNORM_BLOCK};
        case 77: return {VK_FORMAT_BC7_UNORM_BLOCK};
        case 78: return {VK_FORMAT_BC7_SRGB_BLOCK};
        case 79: return {VK_FORMAT_BC6H_SFLOAT_BLOCK};
        case 80: return {VK_FORMAT_BC6H_UFLOAT_BLOCK};
        // RGBX: RGBA storage with alpha ignored (sampled as 1, see IgnoresAlpha).
        case 109: return {VK_FORMAT_R8G8B8A8_UNORM};
        case 110: return {VK_FORMAT_R8G8B8A8_SNORM};
        case 111: return {VK_FORMAT_R8G8B8A8_UINT};
        case 112: return {VK_FORMAT_R8G8B8A8_SINT};
        case 113: return {VK_FORMAT_R16G16B16A16_SFLOAT};
        case 114: return {VK_FORMAT_R16G16B16A16_UNORM};
        case 115: return {VK_FORMAT_R16G16B16A16_SNORM};
        case 116: return {VK_FORMAT_R16G16B16A16_UINT};
        case 117: return {VK_FORMAT_R16G16B16A16_SINT};
        case 118: return {VK_FORMAT_R32G32B32A32_SFLOAT};
        case 119: return {VK_FORMAT_R32G32B32A32_UINT};
        case 120: return {VK_FORMAT_R32G32B32A32_SINT};
        default: return {};
    }
}

struct VertexFormat {
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t bytes = 0;
    bool is_int = false, is_signed = false;
};
VertexFormat VertexFormatOf(int f) {
    // components x component bytes; integer formats are fetched as integers.
    struct E { int nvn; VkFormat vk; uint32_t bytes; bool i, s; };
    static const E kTable[] = {
        {1, VK_FORMAT_R8_UNORM, 1, false, false}, {2, VK_FORMAT_R8_SNORM, 1, false, false},
        {3, VK_FORMAT_R8_UINT, 1, true, false}, {4, VK_FORMAT_R8_SINT, 1, true, true},
        {5, VK_FORMAT_R16_SFLOAT, 2, false, false}, {6, VK_FORMAT_R16_UNORM, 2, false, false},
        {7, VK_FORMAT_R16_SNORM, 2, false, false}, {8, VK_FORMAT_R16_UINT, 2, true, false},
        {9, VK_FORMAT_R16_SINT, 2, true, true}, {10, VK_FORMAT_R32_SFLOAT, 4, false, false},
        {11, VK_FORMAT_R32_UINT, 4, true, false}, {12, VK_FORMAT_R32_SINT, 4, true, true},
        {13, VK_FORMAT_R8G8_UNORM, 2, false, false}, {14, VK_FORMAT_R8G8_SNORM, 2, false, false},
        {15, VK_FORMAT_R8G8_UINT, 2, true, false}, {16, VK_FORMAT_R8G8_SINT, 2, true, true},
        {17, VK_FORMAT_R16G16_SFLOAT, 4, false, false}, {18, VK_FORMAT_R16G16_UNORM, 4, false, false},
        {19, VK_FORMAT_R16G16_SNORM, 4, false, false}, {20, VK_FORMAT_R16G16_UINT, 4, true, false},
        {21, VK_FORMAT_R16G16_SINT, 4, true, true}, {22, VK_FORMAT_R32G32_SFLOAT, 8, false, false},
        {23, VK_FORMAT_R32G32_UINT, 8, true, false}, {24, VK_FORMAT_R32G32_SINT, 8, true, true},
        {25, VK_FORMAT_R8G8B8_UNORM, 3, false, false}, {26, VK_FORMAT_R8G8B8_SNORM, 3, false, false},
        {27, VK_FORMAT_R8G8B8_UINT, 3, true, false}, {28, VK_FORMAT_R8G8B8_SINT, 3, true, true},
        {29, VK_FORMAT_R16G16B16_SFLOAT, 6, false, false}, {30, VK_FORMAT_R16G16B16_UNORM, 6, false, false},
        {31, VK_FORMAT_R16G16B16_SNORM, 6, false, false}, {32, VK_FORMAT_R16G16B16_UINT, 6, true, false},
        {33, VK_FORMAT_R16G16B16_SINT, 6, true, true}, {34, VK_FORMAT_R32G32B32_SFLOAT, 12, false, false},
        {35, VK_FORMAT_R32G32B32_UINT, 12, true, false}, {36, VK_FORMAT_R32G32B32_SINT, 12, true, true},
        {37, VK_FORMAT_R8G8B8A8_UNORM, 4, false, false}, {38, VK_FORMAT_R8G8B8A8_SNORM, 4, false, false},
        {39, VK_FORMAT_R8G8B8A8_UINT, 4, true, false}, {40, VK_FORMAT_R8G8B8A8_SINT, 4, true, true},
        {41, VK_FORMAT_R16G16B16A16_SFLOAT, 8, false, false}, {42, VK_FORMAT_R16G16B16A16_UNORM, 8, false, false},
        {43, VK_FORMAT_R16G16B16A16_SNORM, 8, false, false}, {44, VK_FORMAT_R16G16B16A16_UINT, 8, true, false},
        {45, VK_FORMAT_R16G16B16A16_SINT, 8, true, true}, {46, VK_FORMAT_R32G32B32A32_SFLOAT, 16, false, false},
        {47, VK_FORMAT_R32G32B32A32_UINT, 16, true, false}, {48, VK_FORMAT_R32G32B32A32_SINT, 16, true, true},
        {61, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, false, false},
        {62, VK_FORMAT_A2B10G10R10_UINT_PACK32, 4, true, false},
        {63, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, false, false},
    };
    for (const E& e : kTable)
        if (e.nvn == f) return {e.vk, e.bytes, e.i, e.s};
    return {};
}

// ---- NVN enum -> Vulkan (NVN uses GL-style orderings; confirmed from the state logs) --------------------------
VkCompareOp CompareOp(int f) {  // NEVER=1 .. ALWAYS=8
    static const VkCompareOp k[] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS,
                                    VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL, VK_COMPARE_OP_GREATER,
                                    VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS};
    return (f >= 1 && f <= 8) ? k[f] : VK_COMPARE_OP_ALWAYS;
}
VkBlendFactor BlendFactor(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO;
        case 2: return VK_BLEND_FACTOR_ONE;
        case 3: return VK_BLEND_FACTOR_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: return VK_BLEND_FACTOR_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 16: return VK_BLEND_FACTOR_SRC1_COLOR;
        case 17: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case 18: return VK_BLEND_FACTOR_SRC1_ALPHA;
        case 19: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp BlendOp(int e) {
    switch (e) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}
VkStencilOp StencilOp(int o) {
    switch (o) {
        case 2: return VK_STENCIL_OP_ZERO;
        case 3: return VK_STENCIL_OP_REPLACE;
        case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 6: return VK_STENCIL_OP_INVERT;
        case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}
VkSamplerAddressMode AddressMode(int w) {
    switch (w) {
        case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 2: case 3: case 4: case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
VkComponentSwizzle Swizzle(uint8_t s) {
    switch (s) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 2: return VK_COMPONENT_SWIZZLE_R;
        case 3: return VK_COMPONENT_SWIZZLE_G;
        case 4: return VK_COMPONENT_SWIZZLE_B;
        case 5: return VK_COMPONENT_SWIZZLE_A;
        default: return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
}

uint64_t Fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
// Incremental 64-bit FNV-1a for cache keys (no allocation).
struct Key64 {
    uint64_t h = 1469598103934665603ull;
    void Add(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    }
    template <typename T>
    void Add(const T& v) { Add(&v, sizeof(v)); }
};
struct VecHash {
    size_t operator()(const std::vector<uint32_t>& v) const { return size_t(Fnv(v.data(), v.size() * 4)); }
};

// ---- objects ------------------------------------------------------------------------------------------------
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    VkImageType type = VK_IMAGE_TYPE_2D;
    int nvn_format = 0, target = kTarget2D;
    int width = 0, height = 0, depth = 1, levels = 1, layers = 1;
    VkImageAspectFlags aspect = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkFormatFeatureFlags features = 0;
    bool written = false;  // has the GPU (or an upload) put anything in it
    bool uploaded = false;
    // Contents came from game memory (not rendered by the GPU): re-uploaded when the game rewrites the
    // memory. Checked via a sparse hash: after an unchanged check the next one comes twice as many command
    // lists later (up to kMaxCheckInterval); a change, or an explicit MarkTextureDirty, resets that.
    bool cpu_source = false;
    uint64_t sample_hash = 0;
    uint64_t checked_in_list = 0;
    uint64_t next_check_list = 0;
    uint32_t check_interval = 1;
    std::map<std::vector<uint32_t>, VkImageView> views;
    std::unordered_map<uint64_t, VkImageView> fast_views;  // same views by hashed key
};

struct Shader {
    bool ok = false;
    VkShaderModule module = VK_NULL_HANDLE;
    shader::ShaderInfo info;
    std::string error;
};

// Per-stage resources for one draw.
struct DrawState {
    std::shared_ptr<const ShaderCode> program[2];  // vertex, fragment
    std::vector<VertexAttribObj> attribs;
    std::vector<VertexStreamObj> streams;
    BufferRange vertex_buffers[16];
    BufferRange uniforms[2][18];
    BoundTexture textures[2][32];
    TexRef images[2][8];  // storage images (nvnCommandBufferBindImage)
    BlendStateObj blend[8];
    ColorStateObj color;
    ChannelMaskStateObj channel;
    DepthStencilStateObj depth;
    PolygonStateObj polygon;
    MultisampleStateObj multisample;
    float viewport[4] = {0, 0, 0, 0};
    bool viewport_set = false;
    float scissor[4] = {0, 0, 0, 0};
    bool scissor_set = false;
    float depth_range[2] = {0.0f, 1.0f};
    float alpha_ref = 0.0f;
    float blend_color[4] = {0, 0, 0, 0};
    uint32_t stencil_ref[2] = {0, 0}, stencil_write[2] = {0xFF, 0xFF}, stencil_compare[2] = {0xFF, 0xFF};
    std::vector<TexRef> rts;  // colors then depth at [num_rt]
    int num_rt = 0;
};

class VulkanBackend final : public Backend {
public:
    const char* Name() const override { return "vulkan"; }

    struct MsTimer {
        double& total;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        explicit MsTimer(double& t) : total(t) {}
        ~MsTimer() { total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    };

    void Execute(const CmdList& cmds) override {
        std::lock_guard<std::mutex> lk(m_);
        MsTimer timer(exec_ms_);
        if (!EnsureDevice()) return;
        ResetTransient();
        list_counter_++;
        MaybeArmCapture();
        for (const Cmd& c : cmds) {
            switch (c.type) {
                case CmdType::SetRenderTargets:
                    if (capture_active_) CaptureEndOfPass();
                    st_.num_rt = c.num_rt;
                    st_.rts = c.rts ? *c.rts : std::vector<TexRef>{};
                    LogTargets(st_.rts, st_.num_rt);
                    break;
                case CmdType::ClearColor:
                    if (c.mask == 0 || c.index < 0 || c.index >= st_.num_rt) break;
                    if (Image* img = GetImage(st_.rts[c.index], false)) {
                        Begin();
                        EndRendering();
                        ClearColorImage(*img, c.color, c.kind, st_.rts[c.index].layer, st_.rts[c.index].level);
                    }
                    break;
                case CmdType::ClearTexture:
                    if (c.mask == 0) break;
                    if (Image* img = GetImage(c.dst, false)) {
                        Begin();
                        EndRendering();
                        ClearColorImage(*img, c.color, c.kind);
                    }
                    break;
                case CmdType::ClearDepthStencil: ClearDepth(c); break;
                case CmdType::CopyTexture: {
                    Image* src = GetImage(c.src, true);  // a CPU-written source (video frames) is refreshed
                    Image* dst = GetImage(c.dst, false);
                    if (!src || !dst || src == dst) break;
                    Begin();
                    EndRendering();
                    CopyImage(*src, c.src_region, *dst, c.dst_region, c.flags);
                    break;
                }
                case CmdType::CopyBufferToTexture: CopyBufferToTexture(c); break;
                case CmdType::UpdateUniform: {
                    auto data = std::static_pointer_cast<const std::vector<uint8_t>>(c.obj);
                    if (data && GuestRangeValid(c.dst_addr, data->size())) {
                        memcpy(reinterpret_cast<void*>(c.dst_addr), data->data(), data->size());
                        stream_cache_.clear();  // cached copies may include the old bytes
                    }
                    break;
                }
                case CmdType::BindProgram: {
                    auto code = std::static_pointer_cast<const ShaderCode>(c.obj);
                    if (program_binds_logged_.insert(c.index * 16 + (code ? code->stage + 1 : 0)).second &&
                        program_binds_logged_.size() <= 12)
                        LOGI(kTag, "program bind: stage bits 0x%x, program stage %d", c.index, code ? code->stage : -1);
                    for (int s = 0; s < 2; s++)
                        if (c.index & (1 << s)) st_.program[s] = (code && code->stage == s) ? code : nullptr;
                    break;
                }
                case CmdType::BindVertexAttribs:
                    st_.attribs = *std::static_pointer_cast<const std::vector<VertexAttribObj>>(c.obj);
                    break;
                case CmdType::BindVertexStreams:
                    st_.streams = *std::static_pointer_cast<const std::vector<VertexStreamObj>>(c.obj);
                    break;
                case CmdType::BindVertexBuffers: {
                    const auto& v = *std::static_pointer_cast<const std::vector<BufferRange>>(c.obj);
                    for (size_t i = 0; i < v.size(); i++)
                        if (c.index + i < 16) st_.vertex_buffers[c.index + i] = v[i];
                    break;
                }
                case CmdType::BindUniformBuffers: {
                    if (c.stage < 0 || c.stage > 1) break;  // only vertex / fragment
                    const auto& v = *std::static_pointer_cast<const std::vector<BufferRange>>(c.obj);
                    for (size_t i = 0; i < v.size(); i++) {
                        const size_t idx = size_t(c.index) + i;
                        if (idx < 18) st_.uniforms[c.stage][idx] = v[i];
                        LogUniformBind(c.stage, int(idx), v[i]);
                    }
                    break;
                }
                case CmdType::BindTextures: {
                    if (c.stage < 0 || c.stage > 1) break;
                    const auto& v = *std::static_pointer_cast<const std::vector<BoundTexture>>(c.obj);
                    for (size_t i = 0; i < v.size(); i++)
                        if (c.index + i < 32) st_.textures[c.stage][c.index + i] = v[i];
                    break;
                }
                case CmdType::BindImages: {
                    if (c.stage < 0 || c.stage > 1) break;
                    const auto& v = *std::static_pointer_cast<const std::vector<TexRef>>(c.obj);
                    for (size_t i = 0; i < v.size(); i++) {
                        if (c.index + i >= 8) break;
                        st_.images[c.stage][c.index + i] = v[i];
                        if (image_binds_logged_ < 6 && v[i].valid) {
                            image_binds_logged_++;
                            LOGI(kTag, "image bound: stage %d slot %zu, %dx%dx%d format %d", c.stage, c.index + i,
                                 v[i].tex.width, v[i].tex.height, v[i].tex.depth, v[i].tex.format);
                        }
                    }
                    break;
                }
                case CmdType::BindBlend: {
                    const auto& b = *std::static_pointer_cast<const BlendStateObj>(c.obj);
                    if (b.target >= 0 && b.target < 8) st_.blend[b.target] = b;
                    break;
                }
                case CmdType::BindColor: st_.color = *std::static_pointer_cast<const ColorStateObj>(c.obj); break;
                case CmdType::BindChannelMask:
                    st_.channel = *std::static_pointer_cast<const ChannelMaskStateObj>(c.obj);
                    break;
                case CmdType::BindDepthStencil:
                    st_.depth = *std::static_pointer_cast<const DepthStencilStateObj>(c.obj);
                    break;
                case CmdType::BindPolygon: st_.polygon = *std::static_pointer_cast<const PolygonStateObj>(c.obj); break;
                case CmdType::BindMultisample:
                    st_.multisample = *std::static_pointer_cast<const MultisampleStateObj>(c.obj);
                    break;
                case CmdType::SetViewport:
                    memcpy(st_.viewport, c.f, sizeof(st_.viewport));
                    st_.viewport_set = true;
                    break;
                case CmdType::SetScissor:
                    memcpy(st_.scissor, c.f, sizeof(st_.scissor));
                    st_.scissor_set = true;
                    break;
                case CmdType::SetDepthRange:
                    st_.depth_range[0] = c.f[0];
                    st_.depth_range[1] = c.f[1];
                    break;
                case CmdType::SetAlphaRef: st_.alpha_ref = c.f[0]; break;
                case CmdType::SetBlendColor: memcpy(st_.blend_color, c.f, sizeof(st_.blend_color)); break;
                case CmdType::SetStencil:
                    for (int face = 0; face < 2; face++) {
                        if (!(c.stage & (1 << face))) continue;
                        if (c.index == 0) st_.stencil_ref[face] = uint32_t(c.mask);
                        if (c.index == 1) st_.stencil_write[face] = uint32_t(c.mask);
                        if (c.index == 2) st_.stencil_compare[face] = uint32_t(c.mask);
                    }
                    break;
                case CmdType::Draw: Draw(c.draw); break;
                case CmdType::FenceSync: break;  // signalled by the queue once the list has run
            }
        }
        if (capture_active_) CaptureEndOfPass();
        if (recording_) {
            EndRendering();
            SubmitAndWait(VK_NULL_HANDLE, VK_NULL_HANDLE);
            recording_ = false;
        }
        if (capture_active_) FinishCapture();
        last_list_draws_ = drawn_ - list_start_draws_;
        list_start_draws_ = drawn_;
    }

    // ======================================================================================================
    // frame capture (diagnostics): every image a frame renders, downscaled to RGBA8, saved to one file
    // ======================================================================================================
    struct CaptureItem {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory image_memory = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0, src_w = 0, src_h = 0, nvn_format = 0, draws = 0, pass = 0, flags = 0, raw_bpp = 4;
        uint64_t fs_hash = 0;
    };
    // Video frames: the newest picture per texture storage address, uploaded at that texture's next use.
    struct PendingFrame {
        std::shared_ptr<const std::vector<uint8_t>> rgba;
        int w = 0, h = 0;
    };
    std::mutex frames_mutex_;
    std::unordered_map<uint64_t, PendingFrame> pending_frames_;
    std::atomic<int> pending_count_{0};
    uint64_t frames_set_ = 0, frames_uploaded_ = 0;  // statistics (frames_set_ under frames_mutex_)

    // Textures the CPU wrote through NVN (nvnTextureWriteTexels): checked at their next use. Thread-safe.
    std::mutex dirty_mutex_;
    std::unordered_set<uint64_t> dirty_addrs_;
    std::atomic<int> dirty_count_{0};
    static constexpr uint32_t kMaxCheckInterval = 16;
    void MarkTextureDirty(uint64_t addr) override {
        std::lock_guard<std::mutex> lk(dirty_mutex_);
        if (dirty_addrs_.insert(addr).second) dirty_count_++;
    }
    bool TakeDirty(uint64_t addr) {
        if (dirty_count_.load(std::memory_order_relaxed) == 0) return false;
        std::lock_guard<std::mutex> lk(dirty_mutex_);
        if (!dirty_addrs_.erase(addr)) return false;
        dirty_count_--;
        return true;
    }

    void SetTextureFrame(uint64_t addr, std::shared_ptr<const std::vector<uint8_t>> rgba, int w, int h) override {
        std::lock_guard<std::mutex> lk(frames_mutex_);
        PendingFrame& f = pending_frames_[addr];
        if (!f.rgba) pending_count_++;
        f = {std::move(rgba), w, h};
        frames_set_++;
    }
    // Uploads the pending frame for `img` (render thread). True if there was one.
    bool TakeFrame(Image& img, const TexRef& t) {
        if (pending_count_.load(std::memory_order_relaxed) == 0) return false;
        PendingFrame f;
        {
            std::lock_guard<std::mutex> lk(frames_mutex_);
            auto it = pending_frames_.find(t.addr);
            if (it == pending_frames_.end() || !it->second.rgba) return false;
            f = std::move(it->second);
            pending_frames_.erase(it);
            pending_count_--;
        }
        const int w = std::min(f.w, img.width), h = std::min(f.h, img.height);
        const size_t row = size_t(f.w) * 4, n = row * size_t(h);
        Begin();
        EndRendering();
        if (StreamRemaining() < n + 1024) Flush();
        const VkDeviceSize off = StreamAlloc(n, 16);
        if (off == VK_WHOLE_SIZE) return false;
        memcpy(stream_ptr_ + off, f.rgba->data(), n);
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.bufferRowLength = uint32_t(f.w);
        r.bufferImageHeight = uint32_t(h);
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {uint32_t(w), uint32_t(h), 1};
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd_, stream_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        // The texture's game memory is not updated: keep the change detector from re-uploading it.
        img.uploaded = true;
        img.written = true;
        img.cpu_source = true;
        img.sample_hash = SampleHash(t);
        img.checked_in_list = list_counter_;
        frames_uploaded_++;
        return true;
    }

    void RequestCapture(int delay_s) override {
        capture_request_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count() + int64_t(delay_s) * 1000;
    }
    void MaybeArmCapture() {
        if (capture_active_) return;
        const auto now = std::chrono::steady_clock::now();
        const int64_t req = capture_request_ms_.load();
        if (req && std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() >= req) {
            capture_request_ms_ = 0;
            capture_active_ = true;
            capture_pass_ = 0;
            pass_written_.clear();
            pass_draws_ = 0;
            query_pass_.clear();
            if (query_pool_) {
                Begin();
                EndRendering();
                vkCmdResetQueryPool(cmd_, query_pool_, 0, kMaxQueries);
            }
            StartPassTimestamps();
            LOGI(kTag, "frame capture (requested): recording this frame's passes");
            return;
        }
        if (captures_done_ >= 6 || !dsr::GetRuntime().config.auto_capture) return;
        if (!capture_clock_started_) {
            if (last_list_draws_ < 100) return;  // wait for real 3D frames
            capture_clock_started_ = true;
            next_capture_ = now + std::chrono::seconds(15);
            return;
        }
        if (now < next_capture_) return;
        next_capture_ = now + std::chrono::seconds(45);
        capture_active_ = true;
        capture_pass_ = 0;
        if (!query_pool_) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_OCCLUSION;
            qi.queryCount = kMaxQueries;
            vkCreateQueryPool(device_, &qi, nullptr, &query_pool_);
        }
        if (query_pool_) {
            Begin();
            EndRendering();
            vkCmdResetQueryPool(cmd_, query_pool_, 0, kMaxQueries);
        }
        query_pass_.clear();
        StartPassTimestamps();
        pass_written_.clear();
        pass_draws_ = 0;
        LOGI(kTag, "frame capture %d: recording this frame's passes", captures_done_ + 1);
    }
    void NoteWritten(Image* img) {
        if (!capture_active_ || !img) return;
        if (std::find(pass_written_.begin(), pass_written_.end(), img) == pass_written_.end()) pass_written_.push_back(img);
    }
    // GPU timestamps at pass boundaries during a capture: where the GPU's time goes.
    void StartPassTimestamps() {
        ts_count_ = 0;
        if (!ts_pool_) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = kMaxTimestamps;
            if (vkCreateQueryPool(device_, &qi, nullptr, &ts_pool_) != VK_SUCCESS) ts_pool_ = VK_NULL_HANDLE;
        }
        if (!ts_pool_) return;
        Begin();
        EndRendering();
        vkCmdResetQueryPool(cmd_, ts_pool_, 0, kMaxTimestamps);
        WriteTimestamp();
    }
    void WriteTimestamp() {
        if (!ts_pool_ || ts_count_ >= kMaxTimestamps) return;
        Begin();
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, ts_count_);
        ts_pass_[ts_count_] = capture_pass_;
        ts_count_++;
    }
    void CaptureEndOfPass() {
        WriteTimestamp();
        if (!pass_written_.empty())
            LOGI(kTag, "capture pass %u: %u draws, last fs %016llx, last textures:%s", capture_pass_, pass_draws_,
                 (unsigned long long)pass_fs_hash_, pass_tex_desc_.empty() ? " none" : pass_tex_desc_.c_str());
        if (!pass_written_.empty() && !pass_state_desc_.empty())
            LOGI(kTag, "capture pass %u state: %s", capture_pass_, pass_state_desc_.c_str());
        pass_state_desc_.clear();
        if (!pass_written_.empty() && !pass_ubo_desc_.empty())
            LOGI(kTag, "capture pass %u fragment constants:%s", capture_pass_, pass_ubo_desc_.c_str());
        pass_ubo_desc_.clear();
        for (Image* img : pass_written_) CaptureImage(*img);
        pass_written_.clear();
        pass_draws_ = 0;
        capture_pass_++;
    }
    void CaptureImage(Image& img) {
        if (capture_items_.size() >= 120) return;
        CaptureItem it;
        it.src_w = uint32_t(img.width);
        it.src_h = uint32_t(img.height);
        it.nvn_format = uint32_t(img.nvn_format);
        it.draws = pass_draws_;
        it.pass = capture_pass_;
        it.fs_hash = pass_fs_hash_;
        const bool depth = IsDepthFormat(img.nvn_format);
        // Raw copies (full size, layer 0, mip 0): depth (as the driver stores it, float on Adreno) and float
        // colour formats, whose values an 8-bit blit would clamp (negatives, HDR).
        uint32_t raw_bpp = 0;
        switch (img.vk_format) {
            case VK_FORMAT_R16_SFLOAT: raw_bpp = 2; break;
            case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: raw_bpp = 4; break;
            case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: raw_bpp = 8; break;
            case VK_FORMAT_R32G32B32A32_SFLOAT: raw_bpp = 16; break;
            default: break;
        }
        if (depth) raw_bpp = HasDepth(img.nvn_format) ? 4 : 0;
        if (raw_bpp && img.type == VK_IMAGE_TYPE_2D && it.src_w * it.src_h <= 1280u * 720u) {
            // Array images: up to 4 layers, stacked vertically (flag 16).
            const uint32_t layers = uint32_t(std::min(4, std::max(1, img.layers)));
            it.w = it.src_w;
            it.h = it.src_h * layers;
            it.flags = (depth ? 4u : 8u) | (layers > 1 ? 16u : 0u) | (uint32_t(img.vk_format) << 8);
            it.raw_bpp = raw_bpp;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = VkDeviceSize(it.w) * it.h * raw_bpp;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vkCreateBuffer(device_, &bci, nullptr, &it.buffer) != VK_SUCCESS) return;
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device_, it.buffer, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(device_, &ai, nullptr, &it.memory);
            vkBindBufferMemory(device_, it.buffer, it.memory, 0);
            Begin();
            EndRendering();
            const VkImageLayout restore = img.layout;
            Transition(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy c{};
            c.imageSubresource = {depth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, layers};
            c.imageExtent = {it.w, it.src_h, 1};
            vkCmdCopyImageToBuffer(cmd_, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, it.buffer, 1, &c);
            if (restore != VK_IMAGE_LAYOUT_UNDEFINED) Transition(img, restore);
            capture_items_.push_back(it);
            return;
        }
        if (depth || !(img.features & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || img.type == VK_IMAGE_TYPE_3D) {
            it.flags = 1;  // not captured (depth or unblittable): metadata only
            capture_items_.push_back(it);
            return;
        }
        uint32_t w = it.src_w, h = it.src_h;
        while (w > 480) {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        w = std::max(1u, w);
        h = std::max(1u, h);
        it.w = w;
        it.h = h;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateImage(device_, &ci, nullptr, &it.image) != VK_SUCCESS) return;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, it.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device_, &ai, nullptr, &it.image_memory) != VK_SUCCESS) {
            vkDestroyImage(device_, it.image, nullptr);
            return;
        }
        vkBindImageMemory(device_, it.image, it.image_memory, 0);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = VkDeviceSize(w) * h * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(device_, &bci, nullptr, &it.buffer);
        vkGetBufferMemoryRequirements(device_, it.buffer, &req);
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device_, &ai, nullptr, &it.memory);
        vkBindBufferMemory(device_, it.buffer, it.memory, 0);

        Begin();
        EndRendering();
        const VkImageLayout restore = img.layout;
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        RawBarrier(it.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[1] = {int32_t(it.src_w), int32_t(img.type == VK_IMAGE_TYPE_1D ? 1 : it.src_h), 1};
        b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstOffsets[1] = {int32_t(w), int32_t(h), 1};
        const bool linear = img.features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        vkCmdBlitImage(cmd_, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, it.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &b, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        RawBarrier(it.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd_, it.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, it.buffer, 1, &c);
        if (restore != VK_IMAGE_LAYOUT_UNDEFINED) Transition(img, restore);
        capture_items_.push_back(it);
    }
    void FinishCapture() {
        WaitGpuIdle();  // the readback buffers are written by the submission that just went out
        capture_active_ = false;
        if (ts_pool_ && ts_count_ > 1) {
            std::vector<uint64_t> ts(ts_count_);
            if (vkGetQueryPoolResults(device_, ts_pool_, 0, ts_count_, ts.size() * 8, ts.data(), 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                const double ns = props_.limits.timestampPeriod;
                std::string line;
                double total = 0;
                for (uint32_t i = 1; i < ts_count_; i++) {
                    const double ms = double(ts[i] - ts[i - 1]) * ns / 1e6;
                    total += ms;
                    if (ms < 0.05) continue;
                    char d[48];
                    snprintf(d, sizeof(d), " p%u:%.2f", ts_pass_[i], ms);
                    line += d;
                }
                LOGI(kTag, "capture %d GPU ms per pass (total %.1f ms):%s", captures_done_ + 1, total, line.c_str());
            }
        }
        captures_done_++;
        if (query_pool_ && !query_pass_.empty()) {
            std::vector<uint64_t> res(query_pass_.size());
            if (vkGetQueryPoolResults(device_, query_pool_, 0, uint32_t(res.size()), res.size() * 8, res.data(), 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                std::map<uint32_t, std::pair<uint64_t, uint32_t>> per_pass;  // samples, draws
                for (size_t q = 0; q < res.size(); q++) {
                    per_pass[query_pass_[q]].first += res[q];
                    per_pass[query_pass_[q]].second++;
                }
                std::string line;
                for (auto& [pass, v] : per_pass) {
                    char d[64];
                    snprintf(d, sizeof(d), " p%u:%llu/%u", pass, (unsigned long long)v.first, v.second);
                    line += d;
                }
                LOGI(kTag, "capture %d samples passed per pass (samples/draws):%s", captures_done_, line.c_str());
            }
        }
        char name[64];
        time_t now = time(nullptr);
        tm t;
        localtime_r(&now, &t);
        strftime(name, sizeof(name), "dsr_frame_%Y%m%d_%H%M%S.bin", &t);
        const int fd = dsr::platform::CreateUserFile(name, "application/octet-stream");
        size_t bytes = 0;
        for (CaptureItem& it : capture_items_) {
            struct Header {
                char magic[4];
                uint32_t w, h, src_w, src_h, nvn_format, draws, pass, flags;
                uint64_t fs_hash;
            } hd{{'D', 'F', 'R', 'M'}, it.w, it.h, it.src_w, it.src_h, it.nvn_format, it.draws, it.pass, it.flags, it.fs_hash};
            void* px = nullptr;
            const size_t n = size_t(it.w) * it.h * it.raw_bpp;
            if (it.memory && n) vkMapMemory(device_, it.memory, 0, n, 0, &px);
            if (!px) {
                hd.w = hd.h = 0;
                hd.flags |= 2;
            }
            if (fd >= 0) {
                bytes += size_t(write(fd, &hd, sizeof(hd)));
                if (px) bytes += size_t(write(fd, px, n));
            }
            if (px) vkUnmapMemory(device_, it.memory);
            if (it.buffer) vkDestroyBuffer(device_, it.buffer, nullptr);
            if (it.memory) vkFreeMemory(device_, it.memory, nullptr);
            if (it.image) vkDestroyImage(device_, it.image, nullptr);
            if (it.image_memory) vkFreeMemory(device_, it.image_memory, nullptr);
        }
        if (fd >= 0) close(fd);
        LOGI(kTag, "frame capture %d: %zu images, %zu bytes -> Download/DSRNative/%s", captures_done_,
             capture_items_.size(), bytes, fd >= 0 ? name : "(could not create file)");
        capture_items_.clear();
    }

    void Present(const TexRef& tex) override {
        std::lock_guard<std::mutex> lk(m_);
        if (!EnsureDevice()) return;
        ReportDraws();
        if (!EnsureSwapchain()) return;
        uint32_t idx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, 1000000000ull, acquire_sem_, VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {
            swapchain_dirty_ = true;
            return;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            LogOnce("acquire", "vkAcquireNextImageKHR failed: %d", r);
            return;
        }
        BeginCommands();
        VkImage sc = sc_images_[idx];
        RawBarrier(sc, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);

        Image* src = tex.valid ? FindImage(tex) : nullptr;
        if (src && src->written && !IsDepthFormat(src->nvn_format)) {
            // Fit the game's frame into the screen, keeping its aspect ratio.
            int sw = src->width, sh = src->height;
            int dw = int(sc_extent_.width), dh = int(sc_extent_.height);
            int w = dw, h = int(int64_t(dw) * sh / std::max(1, sw));
            if (h > dh) {
                h = dh;
                w = int(int64_t(dh) * sw / std::max(1, sh));
            }
            int x = (dw - w) / 2, y = (dh - h) / 2;
            Transition(*src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {sw, sh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[0] = {x, y, 0};
            blit.dstOffsets[1] = {x + w, y + h, 1};
            vkCmdBlitImage(cmd_, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            if (!logged_first_image_) {
                logged_first_image_ = true;
                LOGI(kTag, "showing the game's frame: %dx%d (format %d) -> %dx%d at %d,%d", sw, sh, src->nvn_format, w,
                     h, x, y);
            }
        }
        RawBarrier(sc, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        SubmitAndWait(acquire_sem_, render_sem_);

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &render_sem_;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(queue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR)
            swapchain_dirty_ = true;
        else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
            LogOnce("present", "vkQueuePresentKHR failed: %d", r);
        if (++presented_ == 1) LOGI(kTag, "first frame on screen");
    }

private:
    // ======================================================================================================
    // device
    // ======================================================================================================
    bool EnsureDevice() {
        if (device_) return true;
        if (init_failed_) return false;
        init_failed_ = true;  // cleared on success
        uint32_t inst_version = VK_API_VERSION_1_1;
        auto enum_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        if (enum_version) enum_version(&inst_version);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "DSR Native";
        app.apiVersion = std::min<uint32_t>(inst_version, VK_API_VERSION_1_3);
        const char* inst_ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = 2;
        ici.ppEnabledExtensionNames = inst_ext;
        VkResult r = vkCreateInstance(&ici, nullptr, &instance_);
        if (r != VK_SUCCESS) {
            LOGE(kTag, "vkCreateInstance failed: %d — continuing without graphics", r);
            return false;
        }
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (!n) {
            LOGE(kTag, "no Vulkan GPU found — continuing without graphics");
            return false;
        }
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(instance_, &n, pds.data());
        phys_ = pds[0];
        vkGetPhysicalDeviceProperties(phys_, &props_);
        LOGI(kTag, "GPU: %s, Vulkan %u.%u.%u, driver 0x%x", props_.deviceName, VK_VERSION_MAJOR(props_.apiVersion),
             VK_VERSION_MINOR(props_.apiVersion), VK_VERSION_PATCH(props_.apiVersion), props_.driverVersion);
        vkGetPhysicalDeviceMemoryProperties(phys_, &mem_props_);

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
        family_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                family_ = i;
                break;
            }
        if (family_ == UINT32_MAX) {
            LOGE(kTag, "no graphics queue — continuing without graphics");
            return false;
        }

        // Features: dynamic rendering is required for draws; the rest are used when present.
        VkPhysicalDeviceVulkan13Features have13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        const bool v13 = VK_API_VERSION_MINOR(props_.apiVersion) >= 3 && VK_API_VERSION_MINOR(app.apiVersion) >= 3;
        if (v13) have.pNext = &have13;
        vkGetPhysicalDeviceFeatures2(phys_, &have);
        VkPhysicalDeviceFeatures en{};
        en.textureCompressionBC = have.features.textureCompressionBC;
        en.samplerAnisotropy = have.features.samplerAnisotropy;
        en.shaderImageGatherExtended = have.features.shaderImageGatherExtended;
        en.fillModeNonSolid = have.features.fillModeNonSolid;
        en.independentBlend = have.features.independentBlend;
        en.dualSrcBlend = have.features.dualSrcBlend;
        en.imageCubeArray = have.features.imageCubeArray;
        en.depthClamp = have.features.depthClamp;
        en.fragmentStoresAndAtomics = have.features.fragmentStoresAndAtomics;
        en.geometryShader = have.features.geometryShader;  // gl_PrimitiveID in fragment shaders
        en.shaderStorageImageWriteWithoutFormat = have.features.shaderStorageImageWriteWithoutFormat;
        features_ = en;
        VkPhysicalDeviceVulkan13Features en13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        en13.dynamicRendering = have13.dynamicRendering;
        VkPhysicalDeviceFeatures2 en2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        en2.features = en;
        if (v13) en2.pNext = &en13;
        LOGI(kTag, "features: dynamic rendering %d, BC textures %d, anisotropy %d, gather-extended %d",
             int(en13.dynamicRendering), int(en.textureCompressionBC), int(en.samplerAnisotropy),
             int(en.shaderImageGatherExtended));

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char* dev_ext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &en2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_ext;
        r = vkCreateDevice(phys_, &dci, nullptr, &device_);
        if (r != VK_SUCCESS) {
            LOGE(kTag, "vkCreateDevice failed: %d — continuing without graphics", r);
            device_ = VK_NULL_HANDLE;
            return false;
        }
        if (en13.dynamicRendering) {
            begin_rendering_ = reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(device_, "vkCmdBeginRendering"));
            end_rendering_ = reinterpret_cast<PFN_vkCmdEndRendering>(vkGetDeviceProcAddr(device_, "vkCmdEndRendering"));
        }
        if (!begin_rendering_ || !end_rendering_) LOGW(kTag, "no dynamic rendering: draws will be skipped");

        vkGetDeviceQueue(device_, family_, 0, &queue_);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family_;
        vkCreateCommandPool(device_, &pci, nullptr, &pool_);
        // Two sets of per-submission resources: the GPU executes one while the next is recorded.
        for (Slot& sl : slots_) {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool_;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            vkAllocateCommandBuffers(device_, &cai, &sl.cmd);
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(device_, &fci, nullptr, &sl.fence);
            if (!CreateStream() || !CreateDescriptorPool()) return false;
            sl.stream = stream_;
            sl.mem = stream_memory_;
            sl.ptr = stream_ptr_;
            sl.pool = desc_pool_;
        }
        UseSlot(0);
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device_, &sci, nullptr, &acquire_sem_);
        vkCreateSemaphore(device_, &sci, nullptr, &render_sem_);
        LoadPipelineCache();
        init_failed_ = false;
        LOGI(kTag, "device ready (queue family %u)", family_);
        return true;
    }

    uint32_t FindMemory(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mem_props_.memoryTypes[i].propertyFlags & want) == want) return i;
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++)
            if (bits & (1u << i)) return i;
        return 0;
    }

    // ---- streaming buffer: per-list copies of vertex/index/uniform data and upload staging -------------------
    bool CreateStream() {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = kStreamSize + kMaxUniformRange;  // slack: a uniform range may start anywhere in the stream
        bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(device_, &bi, nullptr, &stream_) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, stream_, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device_, &ai, nullptr, &stream_memory_) != VK_SUCCESS) return false;
        vkBindBufferMemory(device_, stream_, stream_memory_, 0);
        void* p = nullptr;
        if (vkMapMemory(device_, stream_memory_, 0, kStreamSize + kMaxUniformRange, 0, &p) != VK_SUCCESS) return false;
        stream_ptr_ = static_cast<uint8_t*>(p);
        // A zero-filled uniform range for constant buffers the game did not bind.
        zero_offset_ = 0;
        memset(stream_ptr_, 0, kMaxUniformRange);
        return true;
    }
    // Returns the offset of `size` bytes in the stream (aligned), flushing the current list if full.
    // Callers must not hold offsets across a call that may flush (see PrepareDraw).
    VkDeviceSize StreamAlloc(VkDeviceSize size, VkDeviceSize align, bool* flushed = nullptr) {
        if (size > kStreamSize - kMaxUniformRange) return VK_WHOLE_SIZE;
        VkDeviceSize off = (stream_off_ + align - 1) / align * align;
        if (off + size > kStreamSize) {
            Flush();
            if (flushed) *flushed = true;
            off = (stream_off_ + align - 1) / align * align;
        }
        stream_off_ = off + size;
        return off;
    }
    VkDeviceSize StreamRemaining() const { return kStreamSize - stream_off_; }

    // Compiled pipelines are kept on disk so a second run does not stall on them.
    std::string PipelineCachePath() const {
        const std::string& dir = dsr::GetRuntime().config.cache_dir;
        return dir.empty() ? std::string() : dir + "/vk_pipelines.bin";
    }
    void LoadPipelineCache() {
        std::vector<uint8_t> data;
        const std::string path = PipelineCachePath();
        if (FILE* f = path.empty() ? nullptr : fopen(path.c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n > 0 && n < (256l << 20)) {
                data.resize(size_t(n));
                if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
            fclose(f);
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_);
        }
        LOGI(kTag, "pipeline cache: %zu bytes loaded", data.size());
    }
    void SavePipelineCache() {
        const std::string path = PipelineCachePath();
        if (!pipeline_cache_ || path.empty()) return;
        size_t n = 0;
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &n, nullptr) != VK_SUCCESS || !n) return;
        std::vector<uint8_t> data(n);
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &n, data.data()) != VK_SUCCESS) return;
        const std::string tmp = path + ".tmp";
        if (FILE* f = fopen(tmp.c_str(), "wb")) {
            const bool ok = fwrite(data.data(), 1, n, f) == n;
            fclose(f);
            if (ok) rename(tmp.c_str(), path.c_str());
        }
    }

    bool CreateDescriptorPool() {
        LOGI(kTag, "uniform buffers: up to %u with dynamic offsets per pipeline layout, alignment %llu, range %u",
             props_.limits.maxDescriptorSetUniformBuffersDynamic,
             (unsigned long long)props_.limits.minUniformBufferOffsetAlignment, props_.limits.maxUniformBufferRange);
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16384}};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets = 4096;
        ci.poolSizeCount = 4;
        ci.pPoolSizes = sizes;
        return vkCreateDescriptorPool(device_, &ci, nullptr, &desc_pool_) == VK_SUCCESS;
    }

    void ResetTransient() {
        stream_off_ = kMaxUniformRange;  // [0, 64K) is the zero buffer
        vkResetDescriptorPool(device_, desc_pool_, 0);
        desc_cache_.clear();
        stream_cache_.clear();
    }
    void Begin() {
        if (recording_) return;
        BeginCommands();
        recording_ = true;
    }
    // Submit what is recorded so far and start over (streaming buffer / descriptors exhausted).
    void Flush() {
        if (!recording_) return;
        EndRendering();
        SubmitAndWait(VK_NULL_HANDLE, VK_NULL_HANDLE);
        recording_ = false;
        ResetTransient();
        Begin();
        flushes_++;
    }

    // ======================================================================================================
    // swapchain
    // ======================================================================================================
    void DestroySwapchain(bool and_surface) {
        if (device_) vkDeviceWaitIdle(device_);
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
        sc_images_.clear();
        if (and_surface && surface_) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
    }

    bool EnsureSwapchain() {
        void* w = dsr::video::GetNativeWindow();
        if (w != window_) {
            if (window_) LOGI(kTag, "window changed (%p -> %p)", window_, w);
            DestroySwapchain(true);
            window_ = w;
        }
        if (!window_) return false;
        if (!surface_) {
            VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
            sci.window = static_cast<ANativeWindow*>(window_);
            VkResult r = vkCreateAndroidSurfaceKHR(instance_, &sci, nullptr, &surface_);
            if (r != VK_SUCCESS) {
                LogOnce("surface", "vkCreateAndroidSurfaceKHR failed: %d", r);
                surface_ = VK_NULL_HANDLE;
                return false;
            }
        }
        if (swapchain_ && !swapchain_dirty_) return true;

        VkSurfaceCapabilitiesKHR caps;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps) != VK_SUCCESS) return false;
        uint32_t nf = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &nf, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(nf);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &nf, formats.data());
        if (formats.empty()) return false;
        VkSurfaceFormatKHR fmt = formats[0];
        for (const auto& f : formats)
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM) {
                fmt = f;
                break;
            }
        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX) {
            auto* aw = static_cast<ANativeWindow*>(window_);
            extent = {uint32_t(ANativeWindow_getWidth(aw)), uint32_t(ANativeWindow_getHeight(aw))};
        }
        // Identity transform: the compositor rotates for the panel (the AYN Thor reports 90 degrees);
        // such swapchains report SUBOPTIMAL every frame, which is expected.
        VkSurfaceTransformFlagBitsKHR transform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                                      ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                                      : caps.currentTransform;
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        for (VkCompositeAlphaFlagBitsKHR a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & a) {
                alpha = a;
                break;
            }
        uint32_t count = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = count;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = transform;
        ci.compositeAlpha = alpha;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = swapchain_;
        VkSwapchainKHR sc = VK_NULL_HANDLE;
        VkResult r = vkCreateSwapchainKHR(device_, &ci, nullptr, &sc);
        DestroySwapchain(false);
        if (r != VK_SUCCESS) {
            LogOnce("swapchain", "vkCreateSwapchainKHR failed: %d", r);
            return false;
        }
        swapchain_ = sc;
        swapchain_dirty_ = false;
        sc_extent_ = extent;
        uint32_t ni = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &ni, nullptr);
        sc_images_.resize(ni);
        vkGetSwapchainImagesKHR(device_, swapchain_, &ni, sc_images_.data());
        if (++swapchains_created_ <= 8)
            LOGI(kTag, "swapchain %ux%u, format %d, %u images", extent.width, extent.height, fmt.format, ni);
        return true;
    }

    // ======================================================================================================
    // images
    // ======================================================================================================
    Image* FindImage(const TexRef& t) {
        auto it = images_.find(t.addr);
        if (it == images_.end()) return nullptr;
        Image& img = it->second;
        if (img.nvn_format != t.tex.format || img.width != t.tex.width || img.height != t.tex.height) {
            const uint64_t key = t.addr ^ (uint64_t(t.tex.format) << 48) ^ (uint64_t(img.nvn_format) << 56);
            if (mismatch_logged_.size() < 24 && mismatch_logged_.insert(key).second)
                LOGW(kTag, "memory @%llx used as fmt %d %dx%d but holds an image of fmt %d %dx%d%s (replaced)",
                     (unsigned long long)t.addr, t.tex.format, t.tex.width, t.tex.height, img.nvn_format, img.width,
                     img.height, img.written ? " that was drawn into" : "");
            return nullptr;
        }
        return &img;
    }

    VkFormat PickFormat(int nvn_format, VkFormatFeatureFlags* features) {
        auto cached = format_cache_.find(nvn_format);
        if (cached != format_cache_.end()) {
            *features = cached->second.second;
            return cached->second.first;
        }
        VkFormatFeatureFlags need = IsDepthFormat(nvn_format) ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                              : (VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
        VkFormat chosen = VK_FORMAT_UNDEFINED;
        VkFormatFeatureFlags feats = 0;
        for (VkFormat f : ImageCandidates(nvn_format)) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(phys_, f, &p);
            if ((p.optimalTilingFeatures & need) == need) {
                chosen = f;
                feats = p.optimalTilingFeatures;
                break;
            }
        }
        if (chosen == VK_FORMAT_UNDEFINED) LOGW(kTag, "NVN format %d has no usable Vulkan format yet", nvn_format);
        format_cache_[nvn_format] = {chosen, feats};
        *features = feats;
        return chosen;
    }

    // Image for a texture, created on first use. For sampling (`upload`), a texture the GPU never wrote is
    // filled from game memory (deswizzled) the first time.
    Image* GetImage(const TexRef& t, bool upload) {
        if (!t.valid || t.tex.width <= 0 || t.tex.height <= 0) return nullptr;
        Image* img = FindImage(t);
        if (!img) img = CreateImage(t);
        if (!img) return nullptr;
        if (upload && TakeFrame(*img, t)) return img;
        if (upload && !img->written && !img->uploaded) {
            Upload(*img, t);
        } else if (upload && img->cpu_source && img->checked_in_list != list_counter_ &&
                   (list_counter_ >= img->next_check_list || TakeDirty(t.addr))) {
            // Most game textures never change after loading; hashing every one every frame cost ~15 ms a
            // frame (a 1 MB texture is thousands of cache misses). So checks back off while nothing changes.
            img->checked_in_list = list_counter_;
            const bool big = size_t(img->width) * img->height >= 512 * 512;
            if (big) big_checks_++;
            if (SampleHash(t) == img->sample_hash) {
                img->check_interval = std::min<uint32_t>(img->check_interval * 2, kMaxCheckInterval);
                img->next_check_list = list_counter_ + img->check_interval;
            } else {
                if (big) big_changes_++;
                Upload(*img, t);
                if (++reuploads_ <= 8 || reuploads_ % 300 == 0)
                    LOGI(kTag, "texture %dx%d format %d changed in game memory: re-uploaded (%llu so far)", img->width,
                         img->height, img->nvn_format, (unsigned long long)reuploads_);
            }
        }
        return img;
    }

    Image* CreateImage(const TexRef& t) {
        VkFormatFeatureFlags feats = 0;
        VkFormat vf = PickFormat(t.tex.format, &feats);
        if (vf == VK_FORMAT_UNDEFINED) {
            if (unknown_formats_.insert(t.tex.format).second)
                LOGW(kTag, "first texture with unsupported NVN format %d: %dx%dx%d, target %d, %d levels", t.tex.format,
                     t.tex.width, t.tex.height, t.tex.depth, t.tex.target, t.tex.levels);
            return nullptr;
        }
        auto old = images_.find(t.addr);
        if (old != images_.end()) {
            if (recording_) Flush();  // the old image may be referenced by recorded work
            vkQueueWaitIdle(queue_);
            DestroyImage(old->second);
            images_.erase(old);
        }
        const TextureLayout layout = ComputeLayout(t.tex);
        Image img;
        img.nvn_format = t.tex.format;
        img.target = t.tex.target;
        img.vk_format = vf;
        img.width = t.tex.width;
        img.height = t.tex.height;
        img.levels = std::max(1, std::min(t.tex.levels, int(layout.levels.size())));
        img.layers = std::max(1, layout.layers);
        img.depth = t.tex.target == kTarget3D ? std::max(1, t.tex.depth) : 1;
        img.aspect = AspectOf(t.tex.format);
        img.features = feats;
        img.type = t.tex.target == kTarget3D ? VK_IMAGE_TYPE_3D
                   : (t.tex.target == kTarget1D || t.tex.target == kTarget1DArray) ? VK_IMAGE_TYPE_1D
                                                                                    : VK_IMAGE_TYPE_2D;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = img.type;
        ci.format = vf;
        ci.extent = {uint32_t(img.width), uint32_t(img.type == VK_IMAGE_TYPE_1D ? 1 : img.height), uint32_t(img.depth)};
        ci.mipLevels = uint32_t(img.levels);
        ci.arrayLayers = uint32_t(img.layers);
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        if ((t.tex.target == kTargetCubemap || t.tex.target == kTargetCubemapArray) && img.layers % 6 == 0 &&
            img.width == img.height)
            ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (feats & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (IsDepthFormat(t.tex.format))
            ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        else if (feats & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
            ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device_, &ci, nullptr, &img.image) != VK_SUCCESS) {
            LogOnce("image", "vkCreateImage failed for %dx%d format %d", img.width, img.height, img.nvn_format);
            return nullptr;
        }
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, img.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device_, &ai, nullptr, &img.memory) != VK_SUCCESS) {
            vkDestroyImage(device_, img.image, nullptr);
            LogOnce("memory", "out of GPU memory for %dx%d format %d", img.width, img.height, img.nvn_format);
            return nullptr;
        }
        vkBindImageMemory(device_, img.image, img.memory, 0);
        if (images_created_++ < 48)
            LOGI(kTag, "image %dx%d%s NVN format %d (target %d, %d levels, %d layers) -> Vulkan format %d", img.width,
                 img.height, img.depth > 1 ? "x3D" : "", img.nvn_format, img.target, img.levels, img.layers, vf);
        return &(images_[t.addr] = img);
    }

    void DestroyImage(Image& img) {
        for (auto& [k, v] : img.views) vkDestroyImageView(device_, v, nullptr);
        if (img.image) vkDestroyImage(device_, img.image, nullptr);
        if (img.memory) vkFreeMemory(device_, img.memory, nullptr);
        img = Image{};
    }

    // Cheap change detector for game-memory textures: 64 bytes at 64 spots plus both ends.
    uint64_t SampleHash(const TexRef& t) {
        const size_t total = ComputeLayout(t.tex).total;
        if (!total || !GuestRangeValid(t.addr, total)) return 0;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(t.addr);
        uint64_t h = Fnv(p, std::min<size_t>(256, total));
        h = Fnv(p + total - std::min<size_t>(256, total), std::min<size_t>(256, total), h);
        // Large textures (video frames: mostly dark, letterboxed) need many spots or changes go unnoticed.
        const size_t spots = total >= (1u << 20) ? 1024 : 64, len = total >= (1u << 20) ? 32 : 64;
        for (size_t i = 1; i < spots; i++) {
            size_t off = total / spots * i;
            h = Fnv(p + off, std::min<size_t>(len, total - off), h);
        }
        return h;
    }

    // Fills a sampled texture from game memory: deswizzle each level/layer into the stream, then copy.
    void Upload(Image& img, const TexRef& t) {
        img.uploaded = true;
        img.cpu_source = true;
        img.sample_hash = SampleHash(t);
        img.checked_in_list = list_counter_;
        img.check_interval = 1;
        img.next_check_list = list_counter_ + 1;
        const TextureLayout layout = ComputeLayout(t.tex);
        if (!GuestRangeValid(t.addr, layout.total)) {
            LogOnce("upload-range", "texture %dx%d format %d: storage outside known memory, not uploaded",
                    t.tex.width, t.tex.height, t.tex.format);
            return;
        }
        const FormatInfo fi = GetFormatInfo(t.tex.format);
        size_t needed = 0;
        for (int layer = 0; layer < img.layers; layer++)
            for (int lv = 0; lv < img.levels; lv++) {
                const LevelLayout& l = layout.levels[size_t(lv)];
                needed += size_t(l.row_bytes) * l.rows * l.depth + 16;
            }
        if (needed > kStreamSize - kMaxUniformRange) {
            LogOnce("upload-size", "texture too large to upload in one go (%zu bytes)", needed);
            return;
        }
        Begin();
        EndRendering();
        if (StreamRemaining() < needed + 1024) Flush();
        std::vector<VkBufferImageCopy> regions;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(t.addr);
        for (int layer = 0; layer < img.layers; layer++) {
            for (int lv = 0; lv < img.levels; lv++) {
                const LevelLayout& l = layout.levels[size_t(lv)];
                const size_t n = size_t(l.row_bytes) * l.rows * l.depth;
                VkDeviceSize off = StreamAlloc(n, 16);
                const uint8_t* src = base + size_t(layer) * layout.layer_stride + l.offset;
                if (layout.linear)
                    memcpy(stream_ptr_ + off, src, n);
                else
                    DeswizzleLevel(stream_ptr_ + off, src, l);
                VkBufferImageCopy r{};
                r.bufferOffset = off;
                r.bufferRowLength = uint32_t(l.row_bytes / fi.bytes_per_block * fi.block_w);
                r.bufferImageHeight = uint32_t(l.rows * fi.block_h);
                r.imageSubresource = {img.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT ? img.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT
                                                                                 : img.aspect,
                                      uint32_t(lv), uint32_t(layer), 1};
                r.imageExtent = {uint32_t(l.width), uint32_t(img.type == VK_IMAGE_TYPE_1D ? 1 : l.height),
                                 uint32_t(l.depth)};
                regions.push_back(r);
            }
        }
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd_, stream_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()),
                               regions.data());
        img.written = true;
        uploads_++;
        if (uploads_ <= 24)
            LOGI(kTag, "uploaded texture %dx%d format %d (%d levels, %d layers, %zu bytes)", img.width, img.height,
                 img.nvn_format, img.levels, img.layers, needed);
    }

    VkImageView View(Image& img, VkImageViewType type, uint32_t base_layer, uint32_t layers, uint32_t levels,
                     VkImageAspectFlags aspect, const uint8_t swz[4], uint32_t base_level = 0) {
        Key64 k;
        const uint32_t parts[10] = {uint32_t(type), base_layer, layers, levels, aspect, swz[0], swz[1], swz[2], swz[3], base_level};
        k.Add(parts, sizeof(parts));
        auto fast = img.fast_views.find(k.h);
        if (fast != img.fast_views.end()) return fast->second;
        std::vector<uint32_t> key(parts, parts + 10);
        auto it = img.views.find(key);
        if (it != img.views.end()) return it->second;
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = img.image;
        ci.viewType = type;
        ci.format = img.vk_format;
        ci.components = {Swizzle(swz[0]), Swizzle(swz[1]), Swizzle(swz[2]), Swizzle(swz[3])};
        ci.subresourceRange = {aspect, base_level, levels, base_layer, layers};
        VkImageView v = VK_NULL_HANDLE;
        if (vkCreateImageView(device_, &ci, nullptr, &v) != VK_SUCCESS) {
            LogOnce("view", "vkCreateImageView failed (format %d, type %d)", img.nvn_format, int(type));
            v = VK_NULL_HANDLE;
        }
        img.views[key] = v;
        img.fast_views[k.h] = v;
        return v;
    }
    // The level/layer a render target view selects (clamped to the image).
    static uint32_t SubLevel(const Image& img, int level) { return uint32_t(std::clamp(level, 0, img.levels - 1)); }
    static uint32_t SubLayer(const Image& img, int layer) {
        return uint32_t(std::clamp(layer, 0, (img.type == VK_IMAGE_TYPE_3D ? img.depth : img.layers) - 1));
    }
    VkImageView AttachmentView(Image& img, int layer = 0, int level = 0) {
        static const uint8_t kIdentity[4] = {2, 3, 4, 5};
        return View(img, img.type == VK_IMAGE_TYPE_1D ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_2D,
                    img.type == VK_IMAGE_TYPE_3D ? 0 : SubLayer(img, layer), 1, 1, img.aspect, kIdentity, SubLevel(img, level));
    }
    // View matching what the shader declared for the slot.
    VkImageView SampledView(Image& img, shader::TexDim dim, bool array, const uint8_t swz[4], const TexRef* ref = nullptr) {
        VkImageViewType type;
        uint32_t layers = 1;
        switch (dim) {
            case shader::TexDim::D1: type = array ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D; break;
            case shader::TexDim::D3: type = VK_IMAGE_VIEW_TYPE_3D; break;
            case shader::TexDim::Cube:
                type = array ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
                layers = array ? uint32_t(img.layers / 6 * 6) : 6;
                break;
            default: type = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; break;
        }
        // A texture view registered with the texture picks the first level/layer (and counts).
        const uint32_t base_level = ref ? SubLevel(img, ref->level) : 0;
        uint32_t levels = uint32_t(img.levels) - base_level;
        if (ref && ref->view_levels > 0) levels = std::min<uint32_t>(levels, uint32_t(ref->view_levels));
        uint32_t base_layer = 0;
        if (ref && img.type != VK_IMAGE_TYPE_3D && dim != shader::TexDim::Cube) base_layer = SubLayer(img, ref->layer);
        if (array && dim != shader::TexDim::Cube) {
            layers = uint32_t(img.layers) - base_layer;
            if (ref && ref->view_layers > 0) layers = std::min<uint32_t>(layers, uint32_t(ref->view_layers));
        }
        const bool type_ok = (dim == shader::TexDim::D3) == (img.type == VK_IMAGE_TYPE_3D) &&
                             (dim == shader::TexDim::D1) == (img.type == VK_IMAGE_TYPE_1D) &&
                             (dim != shader::TexDim::Cube || img.layers >= 6) && base_layer + layers <= uint32_t(img.layers);
        if (!type_ok) return VK_NULL_HANDLE;
        VkImageAspectFlags aspect = img.aspect;
        if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        return View(img, type, base_layer, layers, levels, aspect, swz, base_level);
    }

    // Formats the GPU cannot filter linearly (32-bit float on Adreno, unlike the Switch): sampling them with a
    // linear sampler is undefined in Vulkan and reads zeros here. Such textures get a nearest sampler.
    bool Filterable(const Image& img) {
        const bool ok = img.features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if (!ok && unfilterable_logged_.insert(img.nvn_format).second)
            LOGI(kTag, "NVN format %d (Vulkan %d) cannot be filtered linearly on this GPU: sampled with nearest filtering",
                 img.nvn_format, img.vk_format);
        return ok;
    }
    VkSampler Sampler(const SamplerObj& s, bool shadow, bool filterable = true) {
        Key64 fk;
        fk.Add(&s, sizeof(SamplerObj));
        fk.Add(shadow);
        fk.Add(filterable);
        auto fast = samplers_fast_.find(fk.h);
        if (fast != samplers_fast_.end()) return fast->second;
        std::vector<uint32_t> key(sizeof(SamplerObj) / 4 + 2);
        memcpy(key.data(), &s, sizeof(SamplerObj));
        key[key.size() - 2] = shadow;
        key.back() = filterable;
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = s.mag_filter == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.minFilter = (s.min_filter == 0 || s.min_filter == 2 || s.min_filter == 4) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.mipmapMode = (s.min_filter == 4 || s.min_filter == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = AddressMode(s.wrap[0]);
        ci.addressModeV = AddressMode(s.wrap[1]);
        ci.addressModeW = AddressMode(s.wrap[2]);
        if (!filterable) {
            ci.magFilter = VK_FILTER_NEAREST;
            ci.minFilter = VK_FILTER_NEAREST;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        }
        ci.mipLodBias = s.lod_bias;
        ci.minLod = std::max(0.0f, s.lod_min);
        ci.maxLod = (s.min_filter <= 1) ? 0.25f : std::max(ci.minLod, std::min(s.lod_max, 16.0f));
        if (filterable && features_.samplerAnisotropy && s.max_anisotropy > 1.0f) {
            ci.anisotropyEnable = VK_TRUE;
            ci.maxAnisotropy = std::min(s.max_anisotropy, props_.limits.maxSamplerAnisotropy);
        }
        if (shadow) {
            ci.compareEnable = VK_TRUE;
            ci.compareOp = s.compare_mode ? CompareOp(s.compare_func) : VK_COMPARE_OP_LESS_OR_EQUAL;
        }
        ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        VkSampler smp = VK_NULL_HANDLE;
        vkCreateSampler(device_, &ci, nullptr, &smp);
        samplers_[key] = smp;
        samplers_fast_[fk.h] = smp;
        return smp;
    }

    // ======================================================================================================
    // command buffer helpers
    // ======================================================================================================
    // Everything recorded state-skipping relies on starts over with a new command buffer.
    void InvalidateBoundState() {
        bound_pipeline_ = VK_NULL_HANDLE;
        bound_layout_ = VK_NULL_HANDLE;
        bound_sets_[0] = bound_sets_[1] = VK_NULL_HANDLE;
        dyn_valid_ = false;
        for (auto& b : bound_vb_) b = {VK_NULL_HANDLE, 0};
        bound_ib_ = {VK_NULL_HANDLE, 0};
        bound_ib_type_ = VK_INDEX_TYPE_MAX_ENUM;
        push_valid_ = false;
    }
    void BeginCommands() {
        InvalidateBoundState();
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
    }

    // Submits the current slot's commands and moves to the other slot. The CPU waits only if that slot's
    // previous submission is still running (so CPU recording and GPU execution overlap by one submission).
    void SubmitAndWait(VkSemaphore wait, VkSemaphore signal) {
        vkEndCommandBuffer(cmd_);
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        if (wait) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &wait;
            si.pWaitDstStageMask = &stage;
        }
        if (signal) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &signal;
        }
        VkResult r = vkQueueSubmit(queue_, 1, &si, fence_);
        if (r != VK_SUCCESS) {
            LogOnce("submit", "vkQueueSubmit failed: %d", r);
            return;
        }
        slots_[slot_].pending = true;
        NextSlot();
    }
    void UseSlot(int i) {
        slot_ = i;
        cmd_ = slots_[i].cmd;
        fence_ = slots_[i].fence;
        desc_pool_ = slots_[i].pool;
        stream_ = slots_[i].stream;
        stream_memory_ = slots_[i].mem;
        stream_ptr_ = slots_[i].ptr;
    }
    void NextSlot() {
        const int n = (slot_ + 1) % kSlots;
        Slot& sl = slots_[n];
        if (sl.pending) {
            MsTimer wait_timer(wait_ms_);
            VkResult r = vkWaitForFences(device_, 1, &sl.fence, VK_TRUE, 5000000000ull);
            if (r != VK_SUCCESS) LogOnce("fence", "GPU did not finish within 5 s (%d)", r);
            vkResetFences(device_, 1, &sl.fence);
            sl.pending = false;
        }
        UseSlot(n);
        ResetTransient();
        InvalidateBoundState();
    }
    // Waits for all submitted work (CPU readback of GPU results).
    void WaitGpuIdle() {
        MsTimer wait_timer(wait_ms_);
        vkQueueWaitIdle(queue_);
        for (Slot& sl : slots_)
            if (sl.pending) {
                vkResetFences(device_, 1, &sl.fence);
                sl.pending = false;
            }
    }

    void RawBarrier(VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                    uint32_t levels = 1, uint32_t layers = 1) {
        barriers_++;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {aspect, 0, levels, 0, layers};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
    }
    void Transition(Image& img, VkImageLayout to) {
        if (img.layout == to) return;
        RawBarrier(img.image, img.aspect, img.layout, to, uint32_t(img.levels), uint32_t(img.layers));
        img.layout = to;
    }

    // ======================================================================================================
    // clears / copies
    // ======================================================================================================
    void ClearColorImage(Image& img, const uint32_t color[4], uint8_t kind, int layer = 0, int level = 0) {
        if (IsDepthFormat(img.nvn_format)) return;
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue v;
        memcpy(v.uint32, color, sizeof(v.uint32));  // float/int/uint share the bits
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, SubLevel(img, level), 1,
                                      img.type == VK_IMAGE_TYPE_3D ? 0 : SubLayer(img, layer), 1};
        vkCmdClearColorImage(cmd_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        img.written = true;
        img.cpu_source = false;
        if (kind == kClearFloat && clear_colors_logged_ < 16) {
            float f[4];
            memcpy(f, color, sizeof(f));
            uint64_t key = (uint64_t(img.nvn_format) << 48) ^ ((uint64_t(color[0]) * 31 + color[1]) * 31 + color[2]);
            if (logged_clears_.insert(key).second) {
                clear_colors_logged_++;
                LOGI(kTag, "clear %dx%d (format %d) to (%.3f, %.3f, %.3f, %.3f)", img.width, img.height,
                     img.nvn_format, f[0], f[1], f[2], f[3]);
            }
        }
    }

    void ClearDepth(const Cmd& c) {
        if (st_.rts.size() <= size_t(st_.num_rt)) return;
        Image* img = GetImage(st_.rts[size_t(st_.num_rt)], false);
        if (!img) return;
        depth_gen_++;
        VkImageAspectFlags a = 0;
        if (c.depth_mask && HasDepth(img->nvn_format)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (c.stencil_mask && HasStencil(img->nvn_format)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
        a &= img->aspect;
        if (!a) return;
        Begin();
        EndRendering();
        Transition(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearDepthStencilValue v{std::clamp(c.depth, 0.0f, 1.0f), uint32_t(c.stencil)};
        const TexRef& ref = st_.rts[size_t(st_.num_rt)];
        VkImageSubresourceRange range{a, SubLevel(*img, ref.level), 1, SubLayer(*img, ref.layer), 1};
        vkCmdClearDepthStencilImage(cmd_, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        img->written = true;
        img->cpu_source = false;
    }

    void CopyImage(Image& src, const CopyRegion& sr, Image& dst, const CopyRegion& dr, int flags) {
        if (IsDepthFormat(src.nvn_format) != IsDepthFormat(dst.nvn_format)) return;
        auto clampw = [](int v, int def) { return v > 0 ? v : def; };
        int sw = clampw(sr.width, src.width), sh = clampw(sr.height, src.height);
        int dw = clampw(dr.width, dst.width), dh = clampw(dr.height, dst.height);
        Transition(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Transition(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const bool can_blit = (src.features & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                              (dst.features & VK_FORMAT_FEATURE_BLIT_DST_BIT) &&
                              (!IsDepthFormat(src.nvn_format) || src.vk_format == dst.vk_format);
        if (can_blit) {
            VkImageBlit b{};
            b.srcSubresource = {src.aspect, 0, 0, 1};
            b.srcOffsets[0] = {sr.x, sr.y, 0};
            b.srcOffsets[1] = {sr.x + sw, sr.y + sh, 1};
            b.dstSubresource = {dst.aspect, 0, 0, 1};
            b.dstOffsets[0] = {dr.x, dr.y, 0};
            b.dstOffsets[1] = {dr.x + dw, dr.y + dh, 1};
            if (flags & 0x8) std::swap(b.dstOffsets[0].y, b.dstOffsets[1].y);  // MIRROR_Y
            if (flags & 0x4) std::swap(b.dstOffsets[0].x, b.dstOffsets[1].x);  // MIRROR_X
            VkFilter filter = ((flags & 0x1) && !IsDepthFormat(src.nvn_format) &&
                               (src.features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
                                  ? VK_FILTER_LINEAR
                                  : VK_FILTER_NEAREST;
            vkCmdBlitImage(cmd_, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, filter);
        } else if (src.vk_format == dst.vk_format && sw == dw && sh == dh) {
            VkImageCopy c{};
            c.srcSubresource = {src.aspect, 0, 0, 1};
            c.srcOffset = {sr.x, sr.y, 0};
            c.dstSubresource = {dst.aspect, 0, 0, 1};
            c.dstOffset = {dr.x, dr.y, 0};
            c.extent = {uint32_t(sw), uint32_t(sh), 1};
            vkCmdCopyImage(cmd_, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        } else {
            LogOnce("copy", "cannot copy format %d -> %d yet", src.nvn_format, dst.nvn_format);
            return;
        }
        dst.written = dst.written || src.written;
        dst.cpu_source = false;
        if (copies_logged_ < 16) {
            copies_logged_++;
            LOGI(kTag, "copy %dx%d (format %d) -> %dx%d (format %d), flags 0x%x", sw, sh, src.nvn_format, dw, dh,
                 dst.nvn_format, flags);
        }
    }

    // Linear buffer data (row stride from SetCopyRowStride, 0 = tight) into a texture region.
    void CopyBufferToTexture(const Cmd& c) {
        Image* img = GetImage(c.dst, false);
        if (!img || IsDepthFormat(img->nvn_format)) return;
        const FormatInfo fi = GetFormatInfo(img->nvn_format);
        const CopyRegion& r = c.dst_region;
        const int w = r.width > 0 ? r.width : img->width, h = r.height > 0 ? r.height : img->height;
        const int d = std::max(1, r.depth);
        const size_t row = size_t((w + fi.block_w - 1) / fi.block_w) * size_t(fi.bytes_per_block);
        const size_t rows = size_t((h + fi.block_h - 1) / fi.block_h);
        const size_t stride = c.f[0] > 0 ? size_t(c.f[0]) : row;
        const size_t image_stride = c.f[1] > 0 ? size_t(c.f[1]) : stride * rows;
        const size_t src_size = image_stride * size_t(d - 1) + stride * (rows - 1) + row;
        if (!GuestRangeValid(c.src_addr, src_size)) {
            LogOnce("b2t-range", "buffer-to-texture copy source outside known memory");
            return;
        }
        Begin();
        EndRendering();
        if (StreamRemaining() < row * rows * size_t(d) + 1024) Flush();
        VkDeviceSize off = StreamAlloc(row * rows * size_t(d), 16);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(c.src_addr);
        for (int z = 0; z < d; z++)
            for (size_t y = 0; y < rows; y++)
                memcpy(stream_ptr_ + off + (size_t(z) * rows + y) * row, src + size_t(z) * image_stride + y * stride, row);
        VkBufferImageCopy region{};
        region.bufferOffset = off;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {r.x, r.y, img->type == VK_IMAGE_TYPE_3D ? r.z : 0};
        region.imageExtent = {uint32_t(w), uint32_t(h), uint32_t(img->type == VK_IMAGE_TYPE_3D ? d : 1)};
        if (img->type != VK_IMAGE_TYPE_3D) region.imageSubresource.baseArrayLayer = uint32_t(std::max(0, r.z));
        Transition(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd_, stream_, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        img->written = true;
        img->uploaded = true;
        img->cpu_source = false;
        if (b2t_logged_++ < 8)
            LOGI(kTag, "buffer -> texture %dx%d at %d,%d (format %d)", w, h, r.x, r.y, img->nvn_format);
    }

    // ======================================================================================================
    // rendering
    // ======================================================================================================
    // `line`: the caller (diagnostics: which code path breaks render passes, counted per 10 s).
    void EndRendering(int line = __builtin_LINE()) {
        if (rendering_) {
            end_rendering_(cmd_);
            rendering_ = false;
            pass_breaks_[line]++;
        }
    }

    // Starts rendering into the current render targets (or keeps the current pass if they match).
    // Layer and level of render target i of the current set (8: depth).
    std::pair<int, int> RtSub(size_t i) const {
        const size_t idx = i == 8 ? size_t(st_.num_rt) : i;
        if (idx >= st_.rts.size() || (i != 8 && int(i) >= st_.num_rt)) return {0, 0};
        return {st_.rts[idx].layer, st_.rts[idx].level};
    }
    bool EnsureRendering(const std::vector<Image*>& colors, Image* depth, VkExtent2D* extent) {
        std::vector<uint64_t> key;
        for (size_t i = 0; i < colors.size(); i++) {
            key.push_back(reinterpret_cast<uint64_t>(colors[i] ? colors[i]->image : VK_NULL_HANDLE));
            key.push_back(uint64_t(RtSub(i).first) << 32 | uint32_t(RtSub(i).second));
        }
        key.push_back(reinterpret_cast<uint64_t>(depth ? depth->image : VK_NULL_HANDLE));
        key.push_back(uint64_t(RtSub(8).first) << 32 | uint32_t(RtSub(8).second));
        uint32_t w = UINT32_MAX, h = UINT32_MAX;
        for (size_t i = 0; i < colors.size(); i++)
            if (Image* c = colors[i]) {
                const uint32_t lv = SubLevel(*c, RtSub(i).second);
                w = std::min(w, std::max(1u, uint32_t(c->width) >> lv));
                h = std::min(h, std::max(1u, uint32_t(c->height) >> lv));
            }
        if (depth) {
            const uint32_t lv = SubLevel(*depth, RtSub(8).second);
            w = std::min(w, std::max(1u, uint32_t(depth->width) >> lv));
            h = std::min(h, std::max(1u, uint32_t(depth->height) >> lv));
        }
        if (w == UINT32_MAX) {
            if (!no_attachment_extent_.width) return false;
            w = no_attachment_extent_.width;
            h = no_attachment_extent_.height;
            key.push_back(uint64_t(w) << 32 | h);
        }
        *extent = {w, h};
        if (rendering_ && key == rendering_key_) return true;
        EndRendering();
        std::vector<VkRenderingAttachmentInfo> atts;
        for (size_t ci = 0; ci < colors.size(); ci++) {
            Image* c = colors[ci];
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            if (c) {
                Transition(*c, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                a.imageView = AttachmentView(*c, RtSub(ci).first, RtSub(ci).second);
                a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                a.loadOp = c->written ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
                a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                NoteWritten(c);
                c->written = true;
                c->cpu_source = false;
            }
            atts.push_back(a);
        }
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (depth) {
            Transition(*depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            da.imageView = AttachmentView(*depth, RtSub(8).first, RtSub(8).second);
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            da.loadOp = depth->written ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            da.clearValue.depthStencil = {1.0f, 0};
            NoteWritten(depth);
            depth->written = true;
            depth->cpu_source = false;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {w, h}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = uint32_t(atts.size());
        ri.pColorAttachments = atts.data();
        if (depth && HasDepth(depth->nvn_format)) ri.pDepthAttachment = &da;
        if (depth && HasStencil(depth->nvn_format)) ri.pStencilAttachment = &da;
        begin_rendering_(cmd_, &ri);
        render_passes_++;
        rendering_ = true;
        rendering_key_ = key;
        return true;
    }

    // ======================================================================================================
    // shaders, layouts, pipelines
    // ======================================================================================================
    Shader* GetShader(const ShaderCode& code, const shader::TranslateOptions& opt) {
        const uint32_t parts[6] = {uint32_t(code.hash), uint32_t(code.hash >> 32), opt.int_attrib_mask,
                                   opt.signed_attrib_mask, uint32_t(opt.gl_depth_range), uint32_t(opt.alpha_func)};
        Key64 fk;
        fk.Add(parts, sizeof(parts));
        auto fast = shaders_fast_.find(fk.h);
        if (fast != shaders_fast_.end()) return fast->second;
        std::vector<uint32_t> key(parts, parts + 6);
        auto it = shaders_.find(key);
        if (it != shaders_.end()) {
            shaders_fast_[fk.h] = &it->second;
            return &it->second;
        }
        Shader& s = shaders_[key];
        shader::TranslateResult r = shader::Translate(code.code.data(), code.code.size(), opt);
        if (!r.ok) {
            s.error = r.error;
            if (shader_failures_logged_++ < 24)
                LOGW(kTag, "shader %016llx not translated: %s", (unsigned long long)code.hash, r.error.c_str());
            return &s;
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = r.spirv.size() * 4;
        ci.pCode = r.spirv.data();
        if (vkCreateShaderModule(device_, &ci, nullptr, &s.module) != VK_SUCCESS) {
            s.error = "vkCreateShaderModule failed";
            LogOnce("module", "vkCreateShaderModule failed for shader %016llx", (unsigned long long)code.hash);
            return &s;
        }
        s.info = r.info;
        s.ok = true;
        if (++shaders_translated_ <= 8 || shaders_translated_ % 100 == 0)
            LOGI(kTag, "translated %s shader %016llx (%u instructions); %d so far",
                 r.info.stage == shader::Stage::Vertex ? "vertex" : "pixel", (unsigned long long)code.hash,
                 r.info.instruction_count, shaders_translated_);
        return &s;
    }

    // Uniform buffers of a stage are bound with dynamic offsets (so descriptor sets do not depend on where this
    // draw's uniform data landed in the stream, and are shared between draws) when the device allows that many.
    bool DynamicUniforms(const shader::ShaderInfo& info) const {
        return uint32_t(__builtin_popcount(info.cbuf_mask)) <= props_.limits.maxDescriptorSetUniformBuffersDynamic / 2;
    }
    VkDescriptorSetLayout SetLayout(VkShaderStageFlags stage, const shader::ShaderInfo& info) {
        const uint32_t stage_key = uint32_t(stage) | (DynamicUniforms(info) ? 0x80000000u : 0);
        const uint64_t fast_key = (uint64_t(info.cbuf_mask) << 40) ^ (uint64_t(info.texture_mask) << 20) ^
                                  (uint64_t(info.shadow_mask) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(info.image_mask) << 58) ^
                                  uint64_t(stage_key);
        auto fast = set_layouts_fast_.find(fast_key);
        if (fast != set_layouts_fast_.end() && fast->second.first.size() == 5 && fast->second.first[0] == stage_key &&
            fast->second.first[1] == info.cbuf_mask && fast->second.first[2] == info.texture_mask &&
            fast->second.first[3] == info.shadow_mask && fast->second.first[4] == info.image_mask)
            return fast->second.second;
        std::vector<uint32_t> key = {stage_key, info.cbuf_mask, info.texture_mask, info.shadow_mask, info.image_mask};
        auto it = set_layouts_.find(key);
        if (it != set_layouts_.end()) set_layouts_fast_[fast_key] = {key, it->second};
        if (it != set_layouts_.end()) return it->second;
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i < 18; i++)
            if (info.cbuf_mask >> i & 1)
                b.push_back({i, DynamicUniforms(info) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                             1, stage, nullptr});
        for (uint32_t i = 0; i < 32; i++) {
            if (info.texture_mask >> i & 1) b.push_back({32 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stage, nullptr});
            if (info.shadow_mask >> i & 1) b.push_back({64 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stage, nullptr});
        }
        for (uint32_t i = 0; i < 8; i++)
            if (info.image_mask >> i & 1) b.push_back({96 + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stage, nullptr});
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = uint32_t(b.size());
        ci.pBindings = b.data();
        VkDescriptorSetLayout l = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(device_, &ci, nullptr, &l);
        set_layouts_[key] = l;
        return l;
    }

    VkPipelineLayout PipelineLayout(VkDescriptorSetLayout vs, VkDescriptorSetLayout fs) {
        if (vs == last_pl_vs_ && fs == last_pl_fs_ && last_pl_) return last_pl_;
        last_pl_vs_ = vs;
        last_pl_fs_ = fs;
        last_pl_ = PipelineLayoutLookup(vs, fs);
        return last_pl_;
    }
    VkPipelineLayout PipelineLayoutLookup(VkDescriptorSetLayout vs, VkDescriptorSetLayout fs) {
        std::vector<uint32_t> key = {uint32_t(reinterpret_cast<uintptr_t>(vs)), uint32_t(uint64_t(reinterpret_cast<uintptr_t>(vs)) >> 32),
                                     uint32_t(reinterpret_cast<uintptr_t>(fs)), uint32_t(uint64_t(reinterpret_cast<uintptr_t>(fs)) >> 32)};
        auto it = pipeline_layouts_.find(key);
        if (it != pipeline_layouts_.end()) return it->second;
        VkDescriptorSetLayout sets[2] = {vs, fs};
        VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        VkPipelineLayout l = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device_, &ci, nullptr, &l);
        pipeline_layouts_[key] = l;
        return l;
    }

    struct VertexInput {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
        uint32_t extent[16] = {};  // bytes past the element start each stream's attributes read
    };

    // ======================================================================================================
    // draws
    // ======================================================================================================
    // 1x1 opaque white image for the dimension a shader expects (6 layers for cubes).
    // A copy of a depth buffer, refreshed when depth has been written since the last copy.
    Image* DepthSnapshot(Image& depth, const TexRef& ref) {
        TexRef snap_ref = ref;
        snap_ref.addr = ref.addr ^ (1ull << 62);
        auto it = images_.find(snap_ref.addr);
        Image* snap = nullptr;
        if (it != images_.end() && it->second.width == depth.width && it->second.height == depth.height &&
            it->second.vk_format == depth.vk_format)
            snap = &it->second;
        else
            snap = CreateImage(snap_ref);
        if (!snap) return nullptr;
        auto& gen = snapshot_gen_[snap_ref.addr];
        if (gen != depth_gen_ + 1) {
            EndRendering();
            CopyImage(depth, CopyRegion{}, *snap, CopyRegion{}, 0);
            snap->written = true;
            gen = depth_gen_ + 1;
            Transition(depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }
        if (snap->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            EndRendering();
            Transition(*snap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        return snap;
    }

    Image* Fallback(shader::TexDim dim, bool array) {
        const int key = int(dim) * 2 + (array ? 1 : 0);
        auto it = fallbacks_.find(key);
        if (it != fallbacks_.end()) return it->second;
        TexRef t;
        t.valid = true;
        t.addr = 0x100 + uint64_t(key);  // never a real guest address
        t.tex.format = 37;
        t.tex.width = t.tex.height = 1;
        t.tex.levels = 1;
        t.tex.target = dim == shader::TexDim::D1 ? (array ? kTarget1DArray : kTarget1D)
                       : dim == shader::TexDim::D3 ? kTarget3D
                       : dim == shader::TexDim::Cube ? kTargetCubemap
                       : (array ? kTarget2DArray : kTarget2D);
        t.tex.depth = 1;
        Image* img = CreateImage(t);
        if (img) {
            Begin();
            EndRendering();
            Transition(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue white;
            white.float32[0] = white.float32[1] = white.float32[2] = white.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, uint32_t(img->layers)};
            vkCmdClearColorImage(cmd_, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            img->written = img->uploaded = true;
        }
        fallbacks_[key] = img;
        return img;
    }

    // This game does not bind textures through NVN: it writes texture handles (from
    // nvnDeviceGetTextureHandle) into its own uniform buffer at binding 10, which shaders read as
    // c[13][0x20 + 8 * slot]. Read the handle from there and resolve it like a bind would.
    BoundTexture HandleFromUniforms(int stage, uint32_t slot) {
        BoundTexture bt;
        const BufferRange& r = st_.uniforms[stage][13 - kCbufBase];
        const uint64_t off = 0x20 + 8ull * slot;
        if (!r.address || off + 4 > r.size || !GuestRangeValid(r.address + off, 4)) return bt;
        uint32_t handle;
        memcpy(&handle, reinterpret_cast<const void*>(r.address + off), 4);
        bt.texture = TextureById(int(handle & 0xFFFFF));
        bt.sampler = SamplerById(int(handle >> 20));
        bt.valid = bt.texture.valid;
        if (!bt.valid && handles_logged_++ < 8)
            LOGW(kTag, "texture handle 0x%08x (stage %d slot %u) does not name a registered texture", handle, stage, slot);
        return bt;
    }

    void Skip(const char* reason) {
        auto& n = skipped_[reason];
        if (n++ == 0) LOGW(kTag, "draw skipped: %s", reason);
    }

    // Draw-path profile: time spent per phase (reported with the 10 s stats).
    static constexpr const char* kPhaseNames[7] = {"setup", "targets", "textures", "uniforms", "vertices",
                                                   "pipeline", "descriptors+record"};
    double phase_ms_[7] = {};
    std::chrono::steady_clock::time_point phase_t_;
    void Phase(int finished) {
        auto now = std::chrono::steady_clock::now();
        if (finished >= 0) phase_ms_[finished] += std::chrono::duration<double, std::milli>(now - phase_t_).count();
        phase_t_ = now;
    }
    struct PhaseEnd {
        VulkanBackend* self;
        ~PhaseEnd() { self->Phase(6); }
    };

    void Draw(const DrawParams& p) {
        Phase(-1);
        PhaseEnd phase_end{this};
        no_attachment_extent_ = {0, 0};
        if (p.count <= 0 || p.instances <= 0) return;
        if (!begin_rendering_) return Skip("no dynamic rendering");
        if (!st_.program[0]) return Skip("no vertex program bound");
        const DeviceModes modes = GetDeviceModes();

        // ---- vertex layout and shader options --------------------------------------------------------------
        shader::TranslateOptions vopt, fopt;
        vopt.gl_depth_range = modes.depth_mode == 0;
        for (size_t i = 0; i < st_.attribs.size() && i < 32; i++) {
            VertexFormat vf = VertexFormatOf(st_.attribs[i].format);
            if (vf.is_int) vopt.int_attrib_mask |= 1u << i;
            if (vf.is_signed) vopt.signed_attrib_mask |= 1u << i;
        }
        fopt.alpha_func = st_.color.alpha_func;
        Shader* vs = GetShader(*st_.program[0], vopt);
        // No pixel program: a depth-only pass (shadow maps, depth prepass). Draw with the vertex stage only.
        Shader* fs = st_.program[1] ? GetShader(*st_.program[1], fopt) : &no_fragment_;
        if (!vs->ok || (st_.program[1] && !fs->ok)) return Skip("shader not translated");

        VertexInput vin;
        uint32_t used_streams = 0;
        for (uint32_t loc = 0; loc < 32; loc++) {
            if (!(vs->info.input_mask >> loc & 1)) continue;
            if (loc >= st_.attribs.size()) return Skip("shader reads an attribute that is not set up");
            const VertexAttribObj& a = st_.attribs[loc];
            VertexFormat vf = VertexFormatOf(a.format);
            if (vf.format == VK_FORMAT_UNDEFINED) {
                LogOnce(("vfmt" + std::to_string(a.format)).c_str(), "unknown vertex format %d", a.format);
                return Skip("unknown vertex format");
            }
            if (a.stream < 0 || a.stream >= 16) return Skip("bad vertex stream");
            if (!VertexFormatSupported(vf.format)) return Skip("vertex format not supported by the GPU");
            vin.attributes.push_back({loc, uint32_t(a.stream), vf.format, uint32_t(a.offset)});
            vin.extent[a.stream] = std::max(vin.extent[a.stream], uint32_t(a.offset) + vf.bytes);
            used_streams |= 1u << a.stream;
        }
        for (uint32_t s = 0; s < 16; s++) {
            if (!(used_streams >> s & 1)) continue;
            const VertexStreamObj vs_ = s < st_.streams.size() ? st_.streams[s] : VertexStreamObj{};
            vin.bindings.push_back({s, uint32_t(vs_.stride),
                                    vs_.divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
        }

        Phase(0);
        // ---- render targets ----------------------------------------------------------------------------------
        std::vector<Image*> colors(size_t(std::max(0, st_.num_rt)), nullptr);
        for (int i = 0; i < st_.num_rt; i++) {
            Image* img = GetImage(st_.rts[size_t(i)], false);
            if (img && (img->features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) && !IsDepthFormat(img->nvn_format))
                colors[size_t(i)] = img;
        }
        Image* depth = st_.rts.size() > size_t(st_.num_rt) ? GetImage(st_.rts[size_t(st_.num_rt)], false) : nullptr;
        if (depth && !IsDepthFormat(depth->nvn_format)) depth = nullptr;

        Begin();
        // Keep enough streaming space for this draw so nothing allocated below is lost to a flush.
        if (StreamRemaining() < (16ull << 20)) Flush();

        Phase(1);
        // ---- textures (uploads and layout changes happen outside rendering) ---------------------------------
        struct TexWrite {
            uint32_t set, binding;
            VkDescriptorImageInfo info;
        };
        std::vector<TexWrite> tex_writes;
        for (int stage = 0; stage < 2; stage++) {
            const shader::ShaderInfo& info = stage == 0 ? vs->info : fs->info;
            for (uint32_t slot = 0; slot < 32; slot++) {
                const bool plain = info.texture_mask >> slot & 1, shadow = info.shadow_mask >> slot & 1;
                if (!plain && !shadow) continue;
                BoundTexture bt = st_.textures[stage][slot];
                if (!bt.valid) bt = HandleFromUniforms(stage, slot);
                Image* img = bt.valid ? GetImage(bt.texture, true) : nullptr;
                if (!img) {
                    // Missing or unsupported texture: draw with a white stand-in rather than dropping the
                    // draw (one skipped pass can blank the whole frame).
                    Skip(bt.valid ? "texture format not supported (white stand-in used)"
                                  : "texture handle not found (white stand-in used)");
                    img = Fallback(info.texture_dim[slot], info.texture_array[slot]);
                    bt.texture.tex.swizzle[0] = 2;
                    bt.texture.tex.swizzle[1] = 3;
                    bt.texture.tex.swizzle[2] = 4;
                    bt.texture.tex.swizzle[3] = 5;
                    if (!img) return Skip("no stand-in texture");
                }
                for (Image* c : colors)
                    if (c == img) return Skip("texture is also a render target");
                if (depth == img) {
                    // Sampling the bound depth buffer (soft particles, fog): read a snapshot copy instead.
                    img = DepthSnapshot(*depth, st_.rts[size_t(st_.num_rt)]);
                    if (!img) return Skip("texture is also the depth target");
                }
                if (img->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    if (rendering_) {
                        char k[96];
                        snprintf(k, sizeof(k), "fmt %d %dx%d from layout %d%s", img->nvn_format, img->width,
                                 img->height, int(img->layout), img == depth ? " (depth target)" : "");
                        sampled_breaks_[k]++;
                    }
                    EndRendering();
                    Transition(*img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
                uint8_t swz[4];
                memcpy(swz, bt.texture.tex.swizzle, 4);
                if (IgnoresAlpha(img->nvn_format))
                    for (uint8_t& c : swz)
                        if (c == 5) c = 1;  // A -> ONE
                if (capture_active_) {
                    if (stage == 0 && slot == 0 && tex_desc_draw_ != drawn_) {
                        pass_tex_desc_.clear();
                        tex_desc_draw_ = drawn_;
                    }
                    char d[160];
                    snprintf(d, sizeof(d), " [%s%u: fmt %d %dx%d @%llx%s%s%s]", stage ? "fs" : "vs", slot, img->nvn_format,
                             img->width, img->height, (unsigned long long)bt.texture.addr, img->written ? " drawn" : "",
                             img->uploaded ? " uploaded" : "", shadow ? " compare" : "");
                    if (tex_desc_draw_ != drawn_) {
                        pass_tex_desc_.clear();
                        tex_desc_draw_ = drawn_;
                    }
                    pass_tex_desc_ += d;
                }
                VkImageView view = SampledView(*img, info.texture_dim[slot], info.texture_array[slot], swz,
                                               bt.valid ? &bt.texture : nullptr);
                if (!view) return Skip("texture view does not match the shader");
                if (plain)
                    tex_writes.push_back({uint32_t(stage), 32 + slot,
                                          {Sampler(bt.sampler, false, Filterable(*img)), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
                if (shadow)
                    tex_writes.push_back({uint32_t(stage), 64 + slot,
                                          {Sampler(bt.sampler, true, Filterable(*img)), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
            }
        }

        // Storage images the fragment shader writes (SUST). Kept in GENERAL layout while bound.
        for (int stage = 0; stage < 2; stage++) {
            const shader::ShaderInfo& info = stage == 0 ? vs->info : fs->info;
            for (uint32_t slot = 0; slot < 8; slot++) {
                if (!(info.image_mask >> slot & 1)) continue;
                if (!features_.fragmentStoresAndAtomics || !features_.shaderStorageImageWriteWithoutFormat)
                    return Skip("storage image writes not supported by the GPU");
                const TexRef& ref = st_.images[stage][slot];
                Image* img = ref.valid ? GetImage(ref, false) : nullptr;
                if (!img) return Skip("storage image not bound");
                if (!(img->features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) return Skip("storage image format not supported");
                bool conflict = false;
                for (Image* c : colors) conflict |= c == img;
                if (conflict || depth == img) {
                    // The game leaves the image bound as a colour target while the shader writes it directly
                    // (the shader has no colour outputs). Draw without attachments, sized like the image.
                    if (stage != 1 || fs->info.output_mask != 0) return Skip("storage image is also a render target");
                    colors.clear();
                    if (depth == img) depth = nullptr;
                    no_attachment_extent_ = {uint32_t(img->width), uint32_t(img->height)};
                }
                VkImageViewType type = info.image_dim[slot] == shader::TexDim::D3 ? VK_IMAGE_VIEW_TYPE_3D
                                       : info.image_dim[slot] == shader::TexDim::D1
                                           ? (info.image_array[slot] ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                                           : (info.image_array[slot] ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
                const bool type_ok = (type == VK_IMAGE_VIEW_TYPE_3D) == (img->type == VK_IMAGE_TYPE_3D) &&
                                     (info.image_dim[slot] == shader::TexDim::D1) == (img->type == VK_IMAGE_TYPE_1D);
                if (!type_ok) return Skip("storage image dimension does not match the shader");
                if (img->layout != VK_IMAGE_LAYOUT_GENERAL) {
                    EndRendering();
                    Transition(*img, VK_IMAGE_LAYOUT_GENERAL);
                }
                img->written = true;
                NoteWritten(img);
                static const uint8_t kIdentity[4] = {2, 3, 4, 5};
                const uint32_t layers = info.image_array[slot] ? uint32_t(img->layers) : 1;
                VkImageView view = View(*img, type, 0, layers, 1, img->aspect, kIdentity);
                if (!view) return Skip("storage image view failed");
                tex_writes.push_back({uint32_t(stage), 96 + slot, {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL}});
            }
        }
        Phase(2);
        // ---- uniform data ------------------------------------------------------------------------------------
        struct UboWrite {
            uint32_t set, binding;
            VkDescriptorBufferInfo info;
            bool dynamic;
            uint32_t dyn_offset;
        };
        const bool dyn_stage[2] = {DynamicUniforms(vs->info), DynamicUniforms(fs->info)};
        std::vector<UboWrite> ubo_writes;
        const VkDeviceSize ubo_align = std::max<VkDeviceSize>(16, props_.limits.minUniformBufferOffsetAlignment);
        for (int stage = 0; stage < 2; stage++) {
            const shader::ShaderInfo& info = stage == 0 ? vs->info : fs->info;
            if (!st_.program[stage]) continue;
            const ShaderCode& code = *st_.program[stage];
            for (uint32_t n = 0; n < 18; n++) {
                if (!(info.cbuf_mask >> n & 1)) continue;
                const uint8_t* src = nullptr;
                size_t size = 0;
                if (n == kCompilerConstants) {
                    ShaderConstants(code, &src, &size);
                } else if (n >= kCbufBase) {
                    const BufferRange& r = st_.uniforms[stage][n - kCbufBase];
                    size = size_t(std::min<uint64_t>(r.size, kMaxUniformRange));
                    // Only what the shader can read.
                    if (info.cbuf_bytes[n]) size = std::min<size_t>(size, info.cbuf_bytes[n]);
                    if (r.address && size && GuestRangeValid(r.address, size))
                        src = reinterpret_cast<const uint8_t*>(r.address);
                    else
                        size = 0;
                }
                VkDescriptorBufferInfo bi{stream_, zero_offset_, kMaxUniformRange};
                if (src && size) {
                    // Pad to 16 bytes from guest memory when possible (the shader reads whole vec4s).
                    size_t padded = (size + 15) & ~size_t(15);
                    if (padded != size && n != kCompilerConstants && GuestRangeValid(uint64_t(uintptr_t(src)), padded)) size = padded;
                    VkDeviceSize off;
                    if (size == padded) {
                        off = UploadCached(src, size, ubo_align);
                    } else {
                        off = StreamAlloc(padded, ubo_align);
                        if (off != VK_WHOLE_SIZE) {
                            memset(stream_ptr_ + off, 0, padded);
                            memcpy(stream_ptr_ + off, src, size);
                        }
                    }
                    if (off != VK_WHOLE_SIZE) bi = {stream_, off, padded};
                } else {
                    MissingUniform(stage, n);
                }
                if (dyn_stage[stage]) {
                    // Fixed descriptor (whole range from 0); the offset goes in at bind time.
                    ubo_writes.push_back({uint32_t(stage), n, {stream_, 0, kMaxUniformRange}, true, uint32_t(bi.offset)});
                } else {
                    ubo_writes.push_back({uint32_t(stage), n, bi, false, 0});
                }
                if (capture_active_ && stage == 1 && src && size && n >= kCbufBase) {
                    if (ubo_desc_draw_ != drawn_) {
                        pass_ubo_desc_.clear();
                        ubo_desc_draw_ = drawn_;
                    }
                    char d[64];
                    snprintf(d, sizeof(d), " c%u(%zu B):", n, size);
                    pass_ubo_desc_ += d;
                    const size_t nf = std::min<size_t>(size / 4, 32);
                    for (size_t k = 0; k < nf; k++) {
                        float f;
                        memcpy(&f, src + k * 4, 4);
                        snprintf(d, sizeof(d), " %g", f);
                        pass_ubo_desc_ += d;
                    }
                }
            }
        }

        Phase(3);
        // ---- vertex + index data -------------------------------------------------------------------------------
        // Fast path: the game's own index and vertex buffers are uploaded as they are (each once per list) and
        // the GPU applies the base vertex. Needs u16/u32 indices (or none), a topology Vulkan has, and bound
        // vertex buffers of a reasonable size. Otherwise indices are scanned and rewritten (slow path).
        std::vector<uint32_t> expanded;  // converted index list (slow path)
        const uint64_t geo_resets_before = geo_resets_;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        bool use_indices = false, fast = false;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
        uint32_t index_count = 0, first_vertex = 0;
        int32_t vertex_offset = 0;
        VkDeviceSize vb_offsets[16] = {};
        VkBuffer vb_bufs[16] = {};
        VkDeviceSize index_off = 0;
        VkBuffer index_buf = stream_;
        {
            static const int8_t kTopology[14] = {
                VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, -1, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                -1, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY};
            fast = p.mode >= 0 && p.mode < 14 && kTopology[p.mode] >= 0 && (p.index_type < 0 || p.index_type == 1 || p.index_type == 2);
            const uint32_t isz = p.index_type == 1 ? 2 : 4;
            if (fast && p.index_type >= 0 && !GuestRangeValid(p.index_address, size_t(p.count) * isz)) fast = false;
            // Vertex range the draw reads: from the (cached) index scan, or first..first+count.
            int64_t vmin = p.first, vmax = int64_t(p.first) + p.count - 1;
            if (fast && p.index_type >= 0) {
                uint32_t lo, hi;
                IndexRange(p.index_address, uint32_t(p.count), isz, &lo, &hi);
                vmin = int64_t(lo) + p.base_vertex;
                vmax = int64_t(hi) + p.base_vertex;
                if (vmin < 0 || vmax - vmin > (8 << 20)) fast = false;
            }
            for (const VkVertexInputBindingDescription& b : vin.bindings) {
                if (!fast) break;
                const BufferRange& r = st_.vertex_buffers[b.binding];
                const VertexStreamObj so = b.binding < st_.streams.size() ? st_.streams[b.binding] : VertexStreamObj{};
                const int64_t first = so.divisor ? p.base_instance : vmin;
                const int64_t n = so.divisor ? (int64_t(p.instances) + so.divisor - 1) / so.divisor : vmax - vmin + 1;
                const uint64_t start = uint64_t(first) * b.stride;
                const uint64_t bytes = uint64_t(std::max<int64_t>(0, n - 1)) * b.stride + vin.extent[b.binding];
                if (!r.address || start + bytes > r.size || !GuestRangeValid(r.address + start, size_t(bytes))) {
                    fast = false;
                    break;
                }
            }
            if (fast) {
                topology = VkPrimitiveTopology(kTopology[p.mode]);
                for (const VkVertexInputBindingDescription& b : vin.bindings) {
                    const BufferRange& r = st_.vertex_buffers[b.binding];
                    const VertexStreamObj so = b.binding < st_.streams.size() ? st_.streams[b.binding] : VertexStreamObj{};
                    const int64_t first = so.divisor ? p.base_instance : vmin;
                    const int64_t n = so.divisor ? (int64_t(p.instances) + so.divisor - 1) / so.divisor : vmax - vmin + 1;
                    const uint64_t start = uint64_t(first) * b.stride;
                    const uint64_t bytes = uint64_t(std::max<int64_t>(0, n - 1)) * b.stride + vin.extent[b.binding];
                    const uint8_t* src = reinterpret_cast<const uint8_t*>(r.address + start);
                    VkDeviceSize off = GeoCached(src, size_t(bytes));
                    vb_bufs[b.binding] = geo_buf_;
                    if (off == VK_WHOLE_SIZE) {
                        off = UploadCached(src, size_t(bytes), 16);
                        vb_bufs[b.binding] = stream_;
                    }
                    if (off == VK_WHOLE_SIZE) return Skip("vertex data too large");
                    // Instanced streams start at base_instance: the draw then uses firstInstance 0.
                    vb_offsets[b.binding] = off;
                }
                vertex_offset = int32_t(int64_t(p.base_vertex) - vmin);  // indexed: index + base - vmin
                first_vertex = uint32_t(int64_t(p.first) - vmin);        // non-indexed: 0
                if (p.index_type >= 0) {
                    use_indices = true;
                    index_type = p.index_type == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                    index_count = uint32_t(p.count);
                    const uint8_t* isrc = reinterpret_cast<const uint8_t*>(p.index_address);
                    index_off = GeoCached(isrc, size_t(p.count) * isz);
                    index_buf = geo_buf_;
                    if (index_off == VK_WHOLE_SIZE) {
                        index_off = UploadCached(isrc, size_t(p.count) * isz, 4);
                        index_buf = stream_;
                    }
                    if (index_off == VK_WHOLE_SIZE) return Skip("index data too large");
                }
                fast_draws_++;
            }
        }
        if (!fast) {
            int topology_mode = p.mode;
            int64_t min_vertex = 0, max_vertex = 0;
            bool indexed = p.index_type >= 0;
            if (indexed) {
                const uint32_t isz = p.index_type == 0 ? 1 : p.index_type == 1 ? 2 : 4;
                if (!GuestRangeValid(p.index_address, size_t(p.count) * isz)) return Skip("index data outside known memory");
                const uint8_t* ib = reinterpret_cast<const uint8_t*>(p.index_address);
                expanded.resize(size_t(p.count));
                min_vertex = INT64_MAX;
                max_vertex = INT64_MIN;
                for (int i = 0; i < p.count; i++) {
                    uint32_t v = isz == 1 ? ib[i] : isz == 2 ? reinterpret_cast<const uint16_t*>(ib)[i]
                                                             : reinterpret_cast<const uint32_t*>(ib)[i];
                    expanded[size_t(i)] = v;
                    int64_t vv = int64_t(v) + p.base_vertex;
                    min_vertex = std::min(min_vertex, vv);
                    max_vertex = std::max(max_vertex, vv);
                }
                if (min_vertex < 0) return Skip("negative vertex index");
                for (uint32_t& v : expanded) v = uint32_t(int64_t(v) + p.base_vertex - min_vertex);
            } else {
                min_vertex = p.first;
                max_vertex = int64_t(p.first) + p.count - 1;
                expanded.resize(size_t(p.count));
                for (int i = 0; i < p.count; i++) expanded[size_t(i)] = uint32_t(i);
            }
            if (max_vertex - min_vertex > (8 << 20)) return Skip("vertex range too large");
            // Primitive conversions Vulkan lacks.
            use_indices = indexed;
            switch (topology_mode) {
                case 0: topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
                case 1: topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
                case 2: {  // line loop -> strip with the first vertex repeated
                    topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
                    expanded.push_back(expanded.front());
                    use_indices = true;
                    break;
                }
                case 3: topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
                case 4: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
                case 5: case 8: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
                case 6: case 9: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
                case 7: {  // quads -> triangles
                    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                    std::vector<uint32_t> tris;
                    for (size_t q = 0; q + 3 < expanded.size(); q += 4) {
                        const uint32_t* v = &expanded[q];
                        tris.insert(tris.end(), {v[0], v[1], v[2], v[0], v[2], v[3]});
                    }
                    expanded.swap(tris);
                    use_indices = true;
                    break;
                }
                case 10: topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; break;
                case 11: topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY; break;
                case 12: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; break;
                case 13: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; break;
                default: return Skip("unsupported primitive type");
            }
            if (expanded.empty()) return;
            index_count = uint32_t(expanded.size());

            for (const VkVertexInputBindingDescription& b : vin.bindings) {
                const BufferRange& r = st_.vertex_buffers[b.binding];
                const VertexStreamObj so = b.binding < st_.streams.size() ? st_.streams[b.binding] : VertexStreamObj{};
                int64_t first = so.divisor ? p.base_instance : min_vertex;
                int64_t n = so.divisor ? (int64_t(p.instances) + so.divisor - 1) / so.divisor : max_vertex - min_vertex + 1;
                const uint64_t start = uint64_t(first) * b.stride;
                const uint64_t bytes = uint64_t(std::max<int64_t>(0, n - 1)) * b.stride + vin.extent[b.binding];
                if (!r.address) return Skip("vertex buffer not bound");
                if (start + bytes > r.size || !GuestRangeValid(r.address + start, size_t(bytes)))
                    return Skip("vertex data outside the bound buffer");
                VkDeviceSize off = UploadCached(reinterpret_cast<const uint8_t*>(r.address + start), size_t(bytes), 16);
                if (off == VK_WHOLE_SIZE) return Skip("vertex data too large");
                vb_offsets[b.binding] = off;
                vb_bufs[b.binding] = stream_;
            }
            if (use_indices) {
                index_off = StreamAlloc(expanded.size() * 4, 4);
                memcpy(stream_ptr_ + index_off, expanded.data(), expanded.size() * 4);
            }
        }
        // The geometry cache was emptied during this draw (a new command buffer, other offsets now stale): redo it.
        if (geo_resets_ != geo_resets_before) return Draw(p);
        Phase(4);
        // ---- pipeline ----------------------------------------------------------------------------------------
        VkExtent2D extent;
        // Formats the pipeline renders to (dynamic rendering).
        std::vector<VkFormat> color_formats;
        for (Image* c : colors) color_formats.push_back(c ? c->vk_format : VK_FORMAT_UNDEFINED);
        VkFormat depth_format = depth ? depth->vk_format : VK_FORMAT_UNDEFINED;
        VkDescriptorSetLayout vsl = SetLayout(VK_SHADER_STAGE_VERTEX_BIT, vs->info);
        VkDescriptorSetLayout fsl = SetLayout(VK_SHADER_STAGE_FRAGMENT_BIT, fs->info);
        VkPipelineLayout layout = PipelineLayout(vsl, fsl);
        VkPipeline pipeline = GetPipeline(vs, fs, vin, topology, color_formats, depth ? depth->nvn_format : 0,
                                          depth_format, layout);
        if (!pipeline) {
            if (pipeline_pending_) return;  // compiling in the background: drawn once it is ready
            return Skip("pipeline creation failed");
        }

        Phase(5);
        // ---- descriptors -------------------------------------------------------------------------------------
        VkDescriptorSet sets[2] = {};
        VkDescriptorSetLayout layouts[2] = {vsl, fsl};
        // Identical bindings (same layouts, buffers, offsets, images and samplers) reuse one pair of sets within
        // the list; the pool is reset with the list.
        Key64 dk;
        dk.Add(vsl);
        dk.Add(fsl);
        uint32_t dyn_offsets[36];
        uint32_t dyn_count = 0;
        for (const UboWrite& u : ubo_writes) {
            dk.Add(u.set);
            dk.Add(u.binding);
            dk.Add(u.info);  // for dynamic ones: buffer, 0, range (the same for every draw of the list)
            if (u.dynamic && dyn_count < 36) dyn_offsets[dyn_count++] = u.dyn_offset;
        }
        for (const TexWrite& t : tex_writes) {
            dk.Add(t.set);
            dk.Add(t.binding);
            dk.Add(t.info.sampler);
            dk.Add(t.info.imageView);
            dk.Add(t.info.imageLayout);
        }
        auto cached_sets = desc_cache_.find(dk.h);
        const bool reuse = cached_sets != desc_cache_.end();
        if (reuse) {
            sets[0] = cached_sets->second.first;
            sets[1] = cached_sets->second.second;
            desc_reused_++;
        }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = desc_pool_;
        dai.descriptorSetCount = 2;
        dai.pSetLayouts = layouts;
        if (!reuse && vkAllocateDescriptorSets(device_, &dai, sets) != VK_SUCCESS) {
            // Pool exhausted: everything recorded so far must run first. Retry this draw from the top.
            Flush();
            if (vkAllocateDescriptorSets(device_, &dai, sets) != VK_SUCCESS) return Skip("descriptor allocation failed");
            return Draw(p);
        }
        std::vector<VkWriteDescriptorSet> writes;
        if (!reuse) for (const UboWrite& u : ubo_writes) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[u.set];
            w.dstBinding = u.binding;
            w.descriptorCount = 1;
            w.descriptorType = u.dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo = &u.info;
            writes.push_back(w);
        }
        if (!reuse) for (const TexWrite& t : tex_writes) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[t.set];
            w.dstBinding = t.binding;
            w.descriptorCount = 1;
            w.descriptorType = t.binding >= 96 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &t.info;
            writes.push_back(w);
        }
        if (!reuse) {
            if (!writes.empty()) vkUpdateDescriptorSets(device_, uint32_t(writes.size()), writes.data(), 0, nullptr);
            desc_cache_[dk.h] = {sets[0], sets[1]};
        }

        // ---- record --------------------------------------------------------------------------------------------
        if (!EnsureRendering(colors, depth, &extent)) return Skip("no render target");
        if (pipeline != bound_pipeline_) {
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            bound_pipeline_ = pipeline;
        }
        if (layout != bound_layout_ || sets[0] != bound_sets_[0] || sets[1] != bound_sets_[1] ||
            dyn_count != bound_dyn_count_ || memcmp(dyn_offsets, bound_dyn_, dyn_count * 4) != 0) {
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, dyn_count, dyn_offsets);
            if (layout != bound_layout_) push_valid_ = false;
            bound_layout_ = layout;
            bound_sets_[0] = sets[0];
            bound_sets_[1] = sets[1];
            bound_dyn_count_ = dyn_count;
            memcpy(bound_dyn_, dyn_offsets, dyn_count * 4);
        }
        for (const VkVertexInputBindingDescription& b : vin.bindings) {
            auto& bound = bound_vb_[b.binding];
            if (bound.first == vb_bufs[b.binding] && bound.second == vb_offsets[b.binding]) continue;
            vkCmdBindVertexBuffers(cmd_, b.binding, 1, &vb_bufs[b.binding], &vb_offsets[b.binding]);
            bound = {vb_bufs[b.binding], vb_offsets[b.binding]};
        }
        float push[4] = {st_.alpha_ref, 0, 0, 0};
        if (!push_valid_ || memcmp(push, bound_push_, sizeof(push)) != 0) {
            vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);
            memcpy(bound_push_, push, sizeof(push));
            push_valid_ = true;
        }
        SetDynamicState(extent, modes);
        const bool query = capture_active_ && query_pool_ && query_pass_.size() < kMaxQueries;
        if (query) vkCmdBeginQuery(cmd_, query_pool_, uint32_t(query_pass_.size()), 0);
        if (fast) {
            if (use_indices) {
                if (bound_ib_.first != index_buf || bound_ib_.second != index_off || bound_ib_type_ != index_type) {
                    vkCmdBindIndexBuffer(cmd_, index_buf, index_off, index_type);
                    bound_ib_ = {index_buf, index_off};
                    bound_ib_type_ = index_type;
                }
                vkCmdDrawIndexed(cmd_, index_count, uint32_t(p.instances), 0, vertex_offset, uint32_t(p.base_instance));
            } else {
                vkCmdDraw(cmd_, uint32_t(p.count), uint32_t(p.instances), first_vertex, uint32_t(p.base_instance));
            }
        } else if (use_indices) {
            vkCmdBindIndexBuffer(cmd_, stream_, index_off, VK_INDEX_TYPE_UINT32);
            bound_ib_ = {stream_, index_off};
            bound_ib_type_ = VK_INDEX_TYPE_UINT32;
            vkCmdDrawIndexed(cmd_, index_count, uint32_t(p.instances), 0, 0, uint32_t(p.base_instance));
        } else {
            vkCmdDraw(cmd_, uint32_t(p.count), uint32_t(p.instances), 0, uint32_t(p.base_instance));
        }
        if (query) {
            vkCmdEndQuery(cmd_, query_pool_, uint32_t(query_pass_.size()));
            query_pass_.push_back(capture_pass_);
        }
        if (capture_active_) {
            char d[320];
            snprintf(d, sizeof(d),
                     "vs %016llx | %zu colour targets | alpha test func %d ref %g | blend enable 0x%x | channel mask 0x%08x"
                     " | cull %d front %d | depth test %d write %d | stencil %d",
                     (unsigned long long)(st_.program[0] ? st_.program[0]->hash : 0), colors.size(), st_.color.alpha_func,
                     st_.alpha_ref, st_.color.blend_enable, st_.channel.mask, st_.polygon.cull, st_.polygon.front_face,
                     int(st_.depth.depth_test), int(st_.depth.depth_write), int(st_.depth.stencil_test));
            pass_state_desc_ = d;
        }
        drawn_++;
        pass_draws_++;
        if (st_.program[1]) pass_fs_hash_ = st_.program[1]->hash;
        if (depth && st_.depth.depth_test && st_.depth.depth_write) depth_gen_++;
        if (drawn_ == 1) LOGI(kTag, "first draw recorded");
    }

    void SetDynamicState(VkExtent2D extent, const DeviceModes& modes) {
        // Y direction. In NVN's upper-left origin mode (Direct3D style), clip-space +Y is the top of the target
        // (row 0); in Vulkan it is the bottom. A negative-height viewport makes the two agree, so images land in
        // memory the way the game expects, and front faces keep their winding as seen on screen. Without it,
        // every rasterised image is mirrored and every full-screen pass mirrors it again (the parity decides
        // what ends up upside down), and back-face culling is inverted.
        // In the OpenGL-style lower-left mode, row 0 is the bottom and clip +Y the highest row: as in Vulkan.
        const bool flip = modes.origin_mode == 1;
        float vx = 0, vy = 0, vw = float(extent.width), vh = float(extent.height);
        // A negative height (the game flipping Y itself) is kept: Vulkan accepts it the same way.
        if (st_.viewport_set && st_.viewport[2] > 0 && st_.viewport[3] != 0) {
            vx = st_.viewport[0];
            vy = st_.viewport[1];
            vw = st_.viewport[2];
            vh = st_.viewport[3];
        }
        if (st_.viewport_set && viewports_logged_ < 12) {
            char key[96];
            snprintf(key, sizeof(key), "%g,%g,%g,%g", st_.viewport[0], st_.viewport[1], st_.viewport[2], st_.viewport[3]);
            if (viewport_keys_.insert(key).second) {
                viewports_logged_++;
                LOGI(kTag, "viewport (%s) on a %ux%u target", key, extent.width, extent.height);
            }
        }
        VkViewport vp;
        vp.x = vx;
        vp.width = vw;
        if (flip) {
            vp.y = vy + vh;
            vp.height = -vh;
        } else {
            vp.y = vy;
            vp.height = vh;
        }
        vp.minDepth = std::clamp(st_.depth_range[0], 0.0f, 1.0f);
        vp.maxDepth = std::clamp(st_.depth_range[1], 0.0f, 1.0f);
        VkRect2D sc{{0, 0}, extent};
        if (st_.scissor_set && st_.scissor[2] > 0 && st_.scissor[3] > 0) {
            int x = int(st_.scissor[0]), y = int(st_.scissor[1]), w = int(st_.scissor[2]), h = int(st_.scissor[3]);
            x = std::max(0, x);
            y = std::max(0, y);
            w = std::max(0, std::min(w, int(extent.width) - x));
            h = std::max(0, std::min(h, int(extent.height) - y));
            sc = {{x, y}, {uint32_t(w), uint32_t(h)}};
        }
        // Dynamic state lasts for the whole command buffer: record only what changed since the last draw.
        const uint32_t stencil[6] = {st_.stencil_ref[0], st_.stencil_ref[1], st_.stencil_write[0], st_.stencil_write[1],
                                     st_.stencil_compare[0], st_.stencil_compare[1]};
        const bool all = !dyn_valid_;
        if (all || memcmp(&vp, &dyn_.vp, sizeof(vp)) != 0) vkCmdSetViewport(cmd_, 0, 1, &vp);
        if (all || memcmp(&sc, &dyn_.sc, sizeof(sc)) != 0) vkCmdSetScissor(cmd_, 0, 1, &sc);
        if (all || memcmp(st_.blend_color, dyn_.blend, sizeof(dyn_.blend)) != 0) vkCmdSetBlendConstants(cmd_, st_.blend_color);
        if (all || stencil[0] != dyn_.stencil[0]) vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_BIT, stencil[0]);
        if (all || stencil[1] != dyn_.stencil[1]) vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_BACK_BIT, stencil[1]);
        if (all || stencil[2] != dyn_.stencil[2]) vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_FRONT_BIT, stencil[2]);
        if (all || stencil[3] != dyn_.stencil[3]) vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_BACK_BIT, stencil[3]);
        if (all || stencil[4] != dyn_.stencil[4]) vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_FRONT_BIT, stencil[4]);
        if (all || stencil[5] != dyn_.stencil[5]) vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_BACK_BIT, stencil[5]);
        dyn_.vp = vp;
        dyn_.sc = sc;
        memcpy(dyn_.blend, st_.blend_color, sizeof(dyn_.blend));
        memcpy(dyn_.stencil, stencil, sizeof(stencil));
        dyn_valid_ = true;
    }

    bool VertexFormatSupported(VkFormat f) {
        auto it = vertex_format_ok_.find(f);
        if (it != vertex_format_ok_.end()) return it->second;
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(phys_, f, &p);
        bool ok = p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
        if (!ok) LOGW(kTag, "vertex format %d not supported by this GPU", int(f));
        return vertex_format_ok_[f] = ok;
    }

    // Everything a pipeline is made from, copied so a worker thread can build it.
    struct PipelineJob {
        VkDevice device = VK_NULL_HANDLE;
        VkPipelineCache cache = VK_NULL_HANDLE;
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        std::vector<VkPipelineColorBlendAttachmentState> blend;
        std::vector<VkFormat> color_formats;
        VkFormat depth_format = VK_FORMAT_UNDEFINED, stencil_format = VK_FORMAT_UNDEFINED;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        // Result.
        std::atomic<bool> done{false};
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult result = VK_SUCCESS;
        double ms = 0;
    };
    static void BuildPipeline(PipelineJob& j) {
        const auto t0 = std::chrono::steady_clock::now();
        VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                     {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = j.vs;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = j.fs;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = uint32_t(j.bindings.size());
        vi.pVertexBindingDescriptions = j.bindings.data();
        vi.vertexAttributeDescriptionCount = uint32_t(j.attributes.size());
        vi.pVertexAttributeDescriptions = j.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = j.topology;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = uint32_t(j.blend.size());
        cb.pAttachments = j.blend.data();
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(sizeof(dyn) / sizeof(dyn[0]));
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = uint32_t(j.color_formats.size());
        rinfo.pColorAttachmentFormats = j.color_formats.data();
        rinfo.depthAttachmentFormat = j.depth_format;
        rinfo.stencilAttachmentFormat = j.stencil_format;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.pNext = &rinfo;
        ci.stageCount = j.fs != VK_NULL_HANDLE ? 2 : 1;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vp;
        ci.pRasterizationState = &j.rs;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &j.ds;
        ci.pColorBlendState = &cb;
        ci.pDynamicState = &dy;
        ci.layout = j.layout;
        j.result = vkCreateGraphicsPipelines(j.device, j.cache, 1, &ci, nullptr, &j.pipeline);
        if (j.result != VK_SUCCESS) j.pipeline = VK_NULL_HANDLE;
        j.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        j.done.store(true, std::memory_order_release);
    }

    // Background compilers: a few threads taking jobs in order.
    std::mutex compile_mutex_;
    std::condition_variable compile_cv_;
    std::deque<std::shared_ptr<PipelineJob>> compile_queue_;
    std::vector<std::thread> compilers_;
    std::unordered_map<uint64_t, std::pair<std::vector<uint32_t>, std::shared_ptr<PipelineJob>>> compiling_;
    uint64_t pipeline_waits_ = 0;  // draws skipped while their pipeline compiled (reported)
    bool pipeline_pending_ = false;  // the last GetPipeline returned null because it is still compiling
    void StartCompilers() {
        if (!compilers_.empty()) return;
        for (int i = 0; i < 2; i++)
            compilers_.emplace_back([this] {
                for (;;) {
                    std::shared_ptr<PipelineJob> j;
                    {
                        std::unique_lock<std::mutex> lk(compile_mutex_);
                        compile_cv_.wait(lk, [&] { return !compile_queue_.empty(); });
                        j = std::move(compile_queue_.front());
                        compile_queue_.pop_front();
                    }
                    BuildPipeline(*j);
                }
            });
        for (auto& t : compilers_) t.detach();
    }
    // Takes a finished background compile into the caches. VK_NULL_HANDLE while it is still running.
    VkPipeline FinishCompile(uint64_t h, std::pair<std::vector<uint32_t>, std::shared_ptr<PipelineJob>>& e) {
        PipelineJob& j = *e.second;
        if (!j.done.load(std::memory_order_acquire)) return VK_NULL_HANDLE;
        compile_ms_ += j.ms;
        if (j.result != VK_SUCCESS) LogOnce("pipeline", "vkCreateGraphicsPipelines failed: %d", j.result);
        const VkPipeline p = j.pipeline;
        pipelines_[e.first] = p;
        if (p) pipelines_fast_[h] = p;
        compiling_.erase(h);
        return p;
    }

    VkPipeline GetPipeline(Shader* vs, Shader* fs, const VertexInput& vin, VkPrimitiveTopology topology,
                           const std::vector<VkFormat>& color_formats, int depth_nvn_format, VkFormat depth_format,
                           VkPipelineLayout layout) {
        // Key: everything that shapes the pipeline. Hashed first (no allocation); the full key only on a miss.
        Key64 fk;
        fk.Add(vs->module);
        fk.Add(fs->module);
        fk.Add(vin.bindings.data(), vin.bindings.size() * sizeof(VkVertexInputBindingDescription));
        fk.Add(uint32_t(0xFFFF));
        fk.Add(vin.attributes.data(), vin.attributes.size() * sizeof(VkVertexInputAttributeDescription));
        fk.Add(topology);
        fk.Add(color_formats.data(), color_formats.size() * sizeof(VkFormat));
        fk.Add(depth_format);
        fk.Add(&st_.polygon, sizeof(st_.polygon));
        fk.Add(&st_.depth, sizeof(st_.depth));
        fk.Add(&st_.color, sizeof(st_.color));
        fk.Add(&st_.channel, sizeof(st_.channel));
        for (size_t i = 0; i < color_formats.size(); i++) fk.Add(&st_.blend[i], sizeof(BlendStateObj));
        fk.Add(layout);
        pipeline_pending_ = false;
        auto fast = pipelines_fast_.find(fk.h);
        if (fast != pipelines_fast_.end()) return fast->second;
        auto running = compiling_.find(fk.h);
        if (running != compiling_.end()) {
            VkPipeline p = FinishCompile(fk.h, running->second);
            pipeline_pending_ = !p;
            if (!p) pipeline_waits_++;
            return p;
        }
        std::vector<uint32_t> key;
        auto add = [&](const void* p, size_t n) {
            const size_t at = key.size();
            key.resize(at + (n + 3) / 4, 0);
            memcpy(&key[at], p, n);
        };
        uint64_t mods[2] = {reinterpret_cast<uint64_t>(vs->module), reinterpret_cast<uint64_t>(fs->module)};
        add(mods, sizeof(mods));
        add(vin.bindings.data(), vin.bindings.size() * sizeof(VkVertexInputBindingDescription));
        key.push_back(0xFFFF);
        add(vin.attributes.data(), vin.attributes.size() * sizeof(VkVertexInputAttributeDescription));
        key.push_back(uint32_t(topology));
        add(color_formats.data(), color_formats.size() * sizeof(VkFormat));
        key.push_back(uint32_t(depth_format));
        add(&st_.polygon, sizeof(st_.polygon));
        add(&st_.depth, sizeof(st_.depth));
        add(&st_.color, sizeof(st_.color));
        add(&st_.channel, sizeof(st_.channel));
        for (size_t i = 0; i < color_formats.size(); i++) add(&st_.blend[i], sizeof(BlendStateObj));
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) {
            pipelines_fast_[fk.h] = it->second;
            return it->second;
        }

        auto job = std::make_shared<PipelineJob>();
        PipelineJob& j = *job;
        j.device = device_;
        j.cache = pipeline_cache_;
        j.vs = vs->module;
        j.fs = fs->module;
        const bool has_fragment = fs->module != VK_NULL_HANDLE;
        j.bindings = vin.bindings;
        j.attributes = vin.attributes;
        j.topology = topology;
        VkPipelineRasterizationStateCreateInfo& rs = j.rs;
        rs.polygonMode = (st_.polygon.mode == 0 && features_.fillModeNonSolid) ? VK_POLYGON_MODE_POINT
                         : (st_.polygon.mode == 1 && features_.fillModeNonSolid) ? VK_POLYGON_MODE_LINE
                                                                                 : VK_POLYGON_MODE_FILL;
        rs.cullMode = st_.polygon.cull == 1 ? VK_CULL_MODE_FRONT_BIT
                      : st_.polygon.cull == 2 ? VK_CULL_MODE_BACK_BIT
                      : st_.polygon.cull == 3 ? VK_CULL_MODE_FRONT_AND_BACK
                                              : VK_CULL_MODE_NONE;
        rs.frontFace = st_.polygon.front_face == 0 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineDepthStencilStateCreateInfo& ds = j.ds;
        const bool has_depth = depth_format != VK_FORMAT_UNDEFINED && HasDepth(depth_nvn_format);
        const bool has_stencil = depth_format != VK_FORMAT_UNDEFINED && HasStencil(depth_nvn_format);
        ds.depthTestEnable = has_depth && st_.depth.depth_test;
        ds.depthWriteEnable = has_depth && st_.depth.depth_test && st_.depth.depth_write;
        ds.depthCompareOp = CompareOp(st_.depth.depth_func);
        ds.stencilTestEnable = has_stencil && st_.depth.stencil_test;
        for (int face = 0; face < 2; face++) {
            VkStencilOpState& so = face ? ds.back : ds.front;
            so.failOp = StencilOp(st_.depth.stencil_fail[face]);
            so.passOp = StencilOp(st_.depth.depth_pass[face]);
            so.depthFailOp = StencilOp(st_.depth.depth_fail[face]);
            so.compareOp = CompareOp(st_.depth.stencil_func[face]);
            so.compareMask = 0xFF;
            so.writeMask = 0xFF;
        }
        j.blend.resize(color_formats.size());
        for (size_t i = 0; i < j.blend.size(); i++) {
            VkPipelineColorBlendAttachmentState& b = j.blend[i];
            const BlendStateObj& bs = st_.blend[features_.independentBlend ? i : 0];
            b.blendEnable = (st_.color.blend_enable >> (features_.independentBlend ? i : 0)) & 1;
            b.srcColorBlendFactor = BlendFactor(bs.src_rgb);
            b.dstColorBlendFactor = BlendFactor(bs.dst_rgb);
            b.colorBlendOp = BlendOp(bs.eq_rgb);
            b.srcAlphaBlendFactor = BlendFactor(bs.src_a);
            b.dstAlphaBlendFactor = BlendFactor(bs.dst_a);
            b.alphaBlendOp = BlendOp(bs.eq_a);
            b.colorWriteMask = has_fragment ? ((st_.channel.mask >> (4 * i)) & 0xF) : 0;
        }
        j.color_formats = color_formats;
        j.depth_format = has_depth ? depth_format : VK_FORMAT_UNDEFINED;
        j.stencil_format = has_stencil ? depth_format : VK_FORMAT_UNDEFINED;
        j.layout = layout;
        new_pipelines_++;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (dsr::GetRuntime().config.async_pipelines) {
            // Compile in the background. A pipeline the driver has cached comes back within a few ms: wait that
            // long, so it is used this frame. A genuinely new one skips its draws until ready (no frame hitch).
            StartCompilers();
            {
                std::lock_guard<std::mutex> lk(compile_mutex_);
                compile_queue_.push_back(job);
            }
            compile_cv_.notify_one();
            auto& entry = compiling_[fk.h];
            entry = {std::move(key), job};
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(2500);
            while (!job->done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            pipeline = FinishCompile(fk.h, entry);
            if (!pipeline) {
                pipeline_waits_++;
                pipeline_pending_ = true;
                return VK_NULL_HANDLE;
            }
        } else {
            BuildPipeline(j);
            compile_ms_ += j.ms;
            pipeline = j.pipeline;
            if (j.result != VK_SUCCESS) LogOnce("pipeline", "vkCreateGraphicsPipelines failed: %d", j.result);
            pipelines_[key] = pipeline;
            if (pipeline) pipelines_fast_[fk.h] = pipeline;
        }
        if (pipeline && pipelines_.size() <= 8)
            LOGI(kTag, "pipeline %zu: %zu attributes, %zu targets, depth format %d, topology %d, blend %s",
                 pipelines_.size(), vin.attributes.size(), color_formats.size(), int(depth_format), int(topology),
                 (st_.color.blend_enable & 1) ? "on" : "off");
        return pipeline;
    }

    // c[1]: the literal constants NVN's compiler stored with the shader. The control section header gives
    // its size (word 5) and the constants' size (word 11); the constants sit at the end of the section.
    void ShaderConstants(const ShaderCode& code, const uint8_t** data, size_t* size) {
        *data = code.constants.empty() ? nullptr : code.constants.data();
        *size = code.constants.size();
    }

    // Smallest and largest index of an index buffer. Scanned once and remembered across frames: index
    // buffers are almost always static. The key includes a sparse checksum so a rewritten buffer is rescanned.
    void IndexRange(uint64_t addr, uint32_t count, uint32_t isz, uint32_t* lo, uint32_t* hi) {
        const uint8_t* ib = reinterpret_cast<const uint8_t*>(addr);
        auto at = [&](uint32_t i) -> uint32_t {
            return isz == 2 ? reinterpret_cast<const uint16_t*>(ib)[i] : reinterpret_cast<const uint32_t*>(ib)[i];
        };
        uint64_t key = (addr * 0x9E3779B97F4A7C15ull) ^ (uint64_t(count) << 32) ^ isz;
        const uint32_t step = std::max<uint32_t>(1, count / 16);
        for (uint32_t i = 0; i < count; i += step) key = (key ^ at(i)) * 0x100000001B3ull;
        key = (key ^ at(count - 1)) * 0x100000001B3ull;
        auto it = index_ranges_.find(key);
        if (it != index_ranges_.end()) {
            *lo = it->second.first;
            *hi = it->second.second;
            return;
        }
        uint32_t mn = UINT32_MAX, mx = 0;
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t v = at(i);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        if (index_ranges_.size() > 200000) index_ranges_.clear();
        index_ranges_[key] = {mn, mx};
        *lo = mn;
        *hi = mx;
    }

    // Copies guest data into the stream once per list: the same range bound by several draws (UBOs reused
    // across passes, a mesh drawn into depth, shadow and color) is uploaded once. Guest memory only changes
    // during a list through UpdateUniform, which clears the cache.
    // ---- geometry cache: vertex and index data kept on the GPU across frames -------------------------------
    // Most geometry (the level, characters) never changes after loading, yet was copied into the stream every
    // frame (~30 MB a frame in Undead Burg). Ranges now live in one persistent buffer and are copied again only
    // when their content changed: small ranges (where per-frame data like particles and UI lives) are hashed in
    // full, large ones at 256 spots plus both ends. When the buffer fills up it is emptied (after the GPU is
    // done with it) and refilled.
    static constexpr VkDeviceSize kGeoSize = 192ull << 20;
    VkBuffer geo_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory geo_mem_ = VK_NULL_HANDLE;
    uint8_t* geo_ptr_ = nullptr;
    VkDeviceSize geo_off_ = 0;
    bool geo_failed_ = false;
    struct GeoEntry {
        uint64_t hash;
        VkDeviceSize off;
        uint32_t changes;  // content changes seen; from 2 on the range is streamed instead (volatile)
    };
    std::unordered_map<uint64_t, GeoEntry> geo_;
    uint64_t geo_hits_ = 0, geo_misses_ = 0, geo_bytes_ = 0, geo_resets_ = 0, geo_volatile_ = 0;

    bool EnsureGeo() {
        if (geo_buf_) return true;
        if (geo_failed_) return false;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = kGeoSize;
        bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        VkMemoryRequirements req;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        void* p = nullptr;
        if (vkCreateBuffer(device_, &bi, nullptr, &geo_buf_) != VK_SUCCESS) goto fail;
        vkGetBufferMemoryRequirements(device_, geo_buf_, &req);
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device_, &ai, nullptr, &geo_mem_) != VK_SUCCESS) goto fail;
        vkBindBufferMemory(device_, geo_buf_, geo_mem_, 0);
        if (vkMapMemory(device_, geo_mem_, 0, kGeoSize, 0, &p) != VK_SUCCESS) goto fail;
        geo_ptr_ = static_cast<uint8_t*>(p);
        LOGI(kTag, "geometry cache: %llu MB", (unsigned long long)(kGeoSize >> 20));
        return true;
    fail:
        LOGW(kTag, "geometry cache unavailable: vertex data is streamed every frame");
        if (geo_buf_) vkDestroyBuffer(device_, geo_buf_, nullptr);
        if (geo_mem_) vkFreeMemory(device_, geo_mem_, nullptr);
        geo_buf_ = VK_NULL_HANDLE;
        geo_mem_ = VK_NULL_HANDLE;
        geo_failed_ = true;
        return false;
    }
    static uint64_t GeoHash(const uint8_t* p, size_t n) {
        auto mix = [](uint64_t h, uint64_t v) { return (h ^ v) * 0x9E3779B97F4A7C15ull; };
        uint64_t h = 0xcbf29ce484222325ull ^ n;
        uint64_t v;
        if (n <= 16384) {  // small: the whole range
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                memcpy(&v, p + i, 8);
                h = mix(h, v);
            }
            for (; i < n; i++) h = mix(h, p[i]);
            return h;
        }
        for (size_t i = 0; i < 64; i += 8) {  // both ends
            memcpy(&v, p + i, 8);
            h = mix(h, v);
            memcpy(&v, p + n - 64 + i, 8);
            h = mix(h, v);
        }
        const size_t step = (n - 8) / 256;
        for (size_t k = 0; k < 256; k++) {
            memcpy(&v, p + k * step, 8);
            h = mix(h, v);
        }
        return h;
    }
    // Offset of the range in geo_buf_ (content checked), or VK_WHOLE_SIZE if it cannot be cached.
    VkDeviceSize GeoCached(const uint8_t* src, size_t size) {
        if (size == 0 || size > (16u << 20) || !EnsureGeo()) return VK_WHOLE_SIZE;
        const uint64_t key = (uint64_t(reinterpret_cast<uintptr_t>(src)) * 0x9E3779B97F4A7C15ull) ^ uint64_t(size);
        auto it = geo_.find(key);
        if (it != geo_.end() && it->second.changes >= 2) {
            geo_volatile_++;
            return VK_WHOLE_SIZE;  // rewritten every frame or so: streamed with the list
        }
        const uint64_t h = GeoHash(src, size);
        if (it != geo_.end() && it->second.hash == h) {
            geo_hits_++;
            return it->second.off;
        }
        const uint32_t changes = it != geo_.end() ? it->second.changes + 1 : 0;
        if (changes >= 2) {
            it->second.changes = changes;
            geo_volatile_++;
            return VK_WHOLE_SIZE;
        }
        const VkDeviceSize need = (size + 63) & ~VkDeviceSize(63);
        if (geo_off_ + need > kGeoSize) {
            // Full: start over once the GPU no longer reads any of it.
            Flush();
            WaitGpuIdle();
            geo_.clear();
            geo_off_ = 0;
            geo_resets_++;
            InvalidateBoundState();
            it = geo_.end();
        }
        const VkDeviceSize off = geo_off_;
        geo_off_ += need;
        memcpy(geo_ptr_ + off, src, size);
        geo_misses_++;
        geo_bytes_ += size;
        // A changed range gets new space (the old copy may still be read by frames in flight).
        geo_[key] = {h, off, changes};
        return off;
    }

    VkDeviceSize UploadCached(const uint8_t* src, size_t size, VkDeviceSize align) {
        const uint64_t key = (uint64_t(reinterpret_cast<uintptr_t>(src)) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(size) << 1) ^ align;
        auto it = stream_cache_.find(key);
        if (it != stream_cache_.end()) {
            upload_hits_++;
            return it->second;
        }
        bool flushed = false;
        VkDeviceSize off = StreamAlloc(size, align, &flushed);
        if (off == VK_WHOLE_SIZE) return off;
        memcpy(stream_ptr_ + off, src, size);
        upload_bytes_ += size;
        stream_cache_[key] = off;
        return off;
    }

    void MissingUniform(int stage, uint32_t n) {
        auto key = std::to_string(stage) + ":" + std::to_string(n);
        if (missing_ubo_logged_.insert(key).second && missing_ubo_logged_.size() <= 16)
            LOGW(kTag, "%s shader reads c[%u] but nothing is bound there (zeros used)", stage ? "pixel" : "vertex", n);
    }

    void LogUniformBind(int stage, int index, const BufferRange& r) {
        auto key = std::to_string(stage) + ":" + std::to_string(index);
        if (ubo_binds_logged_.insert(key).second && ubo_binds_logged_.size() <= 24)
            LOGI(kTag, "uniform buffer bound: stage %d index %d (%llu bytes)", stage, index, (unsigned long long)r.size);
    }

    void ReportDraws() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_report_ < std::chrono::seconds(10)) return;
        last_report_ = now;
        std::string skipped;
        for (auto& [reason, n] : skipped_) skipped += " | " + std::to_string(n) + " " + reason;
        LOGI(kTag, "draws: %llu drawn, %zu pipelines, %d shaders, %d uploads, %d flushes%s", (unsigned long long)drawn_,
             pipelines_.size(), shaders_translated_, uploads_, flushes_, skipped.c_str());
        {
            long rss_kb = 0;
            if (FILE* f = fopen("/proc/self/statm", "r")) {
                long pages = 0, resident = 0;
                if (fscanf(f, "%ld %ld", &pages, &resident) == 2) rss_kb = resident * (sysconf(_SC_PAGESIZE) / 1024);
                fclose(f);
            }
            uint64_t gpu = 0;
            for (auto& [a, img] : images_) {
                VkMemoryRequirements req{};
                if (img.image) vkGetImageMemoryRequirements(device_, img.image, &req);
                gpu += req.size;
            }
            LOGI(kTag, "memory: process %ld MB resident, %zu GPU images using %llu MB, %zu index ranges cached", rss_kb / 1024,
                 images_.size(), (unsigned long long)(gpu >> 20), index_ranges_.size());
        }
        if (big_checks_) {
            LOGI(kTag, "game-written textures (>= 512x512): %llu checks, %llu changed and re-uploaded",
                 (unsigned long long)big_checks_, (unsigned long long)big_changes_);
            big_checks_ = big_changes_ = 0;
        }
        {
            std::lock_guard<std::mutex> lk(frames_mutex_);
            if (frames_set_) {
                LOGI(kTag, "video frames: %llu decoded into textures, %llu uploaded (newer ones replace unused ones)",
                     (unsigned long long)frames_set_, (unsigned long long)frames_uploaded_);
                frames_set_ = frames_uploaded_ = 0;
            }
        }
        LOGI(kTag, "last 10 s: %llu render passes started, %llu image barriers", (unsigned long long)render_passes_,
             (unsigned long long)barriers_);
        render_passes_ = barriers_ = 0;
        if (!pass_breaks_.empty()) {
            std::vector<std::pair<uint64_t, int>> v;
            for (auto& [line, n] : pass_breaks_) v.push_back({n, line});
            std::sort(v.rbegin(), v.rend());
            std::string top;
            for (size_t i = 0; i < v.size() && i < 8; i++)
                top += " line " + std::to_string(v[i].second) + ": " + std::to_string(v[i].first) + ";";
            LOGI(kTag, "render passes ended by (source line: count):%s", top.c_str());
            pass_breaks_.clear();
        }
        if (!sampled_breaks_.empty()) {
            std::vector<std::pair<uint64_t, std::string>> v;
            for (auto& [k, n] : sampled_breaks_) v.push_back({n, k});
            std::sort(v.rbegin(), v.rend());
            std::string top;
            for (size_t i = 0; i < v.size() && i < 8; i++) top += " [" + v[i].second + "]: " + std::to_string(v[i].first) + ";";
            LOGI(kTag, "textures that broke a render pass to become readable:%s", top.c_str());
            sampled_breaks_.clear();
        }
        LOGI(kTag, "last 10 s: %.0f ms executing lists (%.0f ms of it waiting for the GPU), %.1f MB streamed, %llu cached uploads",
             exec_ms_, wait_ms_, upload_bytes_ / 1048576.0, (unsigned long long)upload_hits_);
        if (geo_buf_) {
            LOGI(kTag, "geometry cache: %llu reused, %llu copied (%.1f MB), %llu volatile (streamed), %.0f of %llu MB used, "
                 "%llu resets", (unsigned long long)geo_hits_, (unsigned long long)geo_misses_, geo_bytes_ / 1048576.0,
                 (unsigned long long)geo_volatile_, geo_off_ / 1048576.0, (unsigned long long)(kGeoSize >> 20),
                 (unsigned long long)geo_resets_);
            geo_hits_ = geo_misses_ = geo_bytes_ = geo_volatile_ = 0;
        }
        std::string phases;
        for (int i = 0; i < 7; i++) {
            phases += " " + std::string(kPhaseNames[i]) + "=" + std::to_string(int(phase_ms_[i]));
            phase_ms_[i] = 0;
        }
        LOGI(kTag, "draw time by phase (ms):%s | %d pipelines compiled in %.0f ms (%llu draws waited for one) | %llu fast-path draws | %llu descriptor sets reused",
             phases.c_str(), new_pipelines_, compile_ms_, (unsigned long long)pipeline_waits_,
             (unsigned long long)fast_draws_, (unsigned long long)desc_reused_);
        pipeline_waits_ = 0;
        desc_reused_ = 0;
        if (new_pipelines_) SavePipelineCache();
        new_pipelines_ = 0;
        compile_ms_ = 0;
        fast_draws_ = 0;
        exec_ms_ = wait_ms_ = 0;
        upload_bytes_ = upload_hits_ = 0;
    }

    void LogTargets(const std::vector<TexRef>& rts, int num_rt) {
        if (target_sets_logged_ >= 24) return;
        std::string desc;
        uint64_t key = 1469598103934665603ull;
        for (size_t i = 0; i < rts.size(); i++) {
            const TexRef& t = rts[i];
            if (!t.valid) continue;
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%s%dx%d fmt %d", desc.empty() ? "" : ", ",
                     int(i) == num_rt ? "depth " : "", t.tex.width, t.tex.height, t.tex.format);
            desc += buf;
            key = (key ^ (t.addr + uint64_t(t.tex.format) * 7919 + i)) * 1099511628211ull;
        }
        if (desc.empty() || !logged_targets_.insert(key).second) return;
        target_sets_logged_++;
        LOGI(kTag, "render targets: %s", desc.c_str());
    }

    template <typename... A>
    void LogOnce(const char* key, const char* fmt, A... args) {
        if (!logged_once_.insert(key).second) return;
        if constexpr (sizeof...(A) == 0)
            LOGW(kTag, "%s", fmt);
        else
            LOGW(kTag, fmt, args...);
    }

    std::mutex m_;
    bool init_failed_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties mem_props_{};
    VkPhysicalDeviceFeatures features_{};
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkSemaphore acquire_sem_ = VK_NULL_HANDLE, render_sem_ = VK_NULL_HANDLE;
    PFN_vkCmdBeginRendering begin_rendering_ = nullptr;
    PFN_vkCmdEndRendering end_rendering_ = nullptr;
    bool recording_ = false, rendering_ = false;
    std::vector<uint64_t> rendering_key_;

    VkBuffer stream_ = VK_NULL_HANDLE;
    VkDeviceMemory stream_memory_ = VK_NULL_HANDLE;
    uint8_t* stream_ptr_ = nullptr;
    VkDeviceSize stream_off_ = 0, zero_offset_ = 0;
    std::unordered_map<uint64_t, VkDeviceSize> stream_cache_;
    uint64_t upload_hits_ = 0, upload_bytes_ = 0;
    uint64_t depth_gen_ = 0;
    int viewports_logged_ = 0;
    int image_binds_logged_ = 0;
    std::set<uint64_t> mismatch_logged_;
    std::set<int> unfilterable_logged_;
    // frame capture
    bool capture_active_ = false, capture_clock_started_ = false;
    int captures_done_ = 0;
    std::atomic<int64_t> capture_request_ms_{0};
    uint32_t capture_pass_ = 0, pass_draws_ = 0;
    uint64_t pass_fs_hash_ = 0, last_list_draws_ = 0, list_start_draws_ = 0;
    std::chrono::steady_clock::time_point next_capture_;
    std::vector<Image*> pass_written_;
    std::string pass_tex_desc_, pass_ubo_desc_, pass_state_desc_;
    static constexpr uint32_t kMaxQueries = 2048;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    static constexpr uint32_t kMaxTimestamps = 256;
    VkQueryPool ts_pool_ = VK_NULL_HANDLE;
    uint32_t ts_count_ = 0, ts_pass_[kMaxTimestamps] = {};
    uint64_t render_passes_ = 0, barriers_ = 0;
    std::map<int, uint64_t> pass_breaks_;
    std::map<std::string, uint64_t> sampled_breaks_;
    std::vector<uint32_t> query_pass_;
    uint64_t ubo_desc_draw_ = ~0ull;
    uint64_t tex_desc_draw_ = ~0ull;
    std::vector<CaptureItem> capture_items_;
    VkExtent2D no_attachment_extent_{0, 0};  // render area for a draw with no attachments (storage-only)
    uint64_t fast_draws_ = 0, big_checks_ = 0, big_changes_ = 0;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> index_ranges_;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    int new_pipelines_ = 0;
    double compile_ms_ = 0;
    std::set<std::string> viewport_keys_;  // bumped by every depth write (draw or clear)
    std::unordered_map<uint64_t, uint64_t> snapshot_gen_;
    double exec_ms_ = 0, wait_ms_ = 0;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;

    void* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    bool swapchain_dirty_ = false;
    VkExtent2D sc_extent_{};
    std::vector<VkImage> sc_images_;

    DrawState st_;
    Shader no_fragment_;  // stands in for an absent pixel program (ok = false, no module)
    std::unordered_map<uint64_t, Image> images_;
    std::unordered_map<int, std::pair<VkFormat, VkFormatFeatureFlags>> format_cache_;
    std::unordered_map<VkFormat, bool> vertex_format_ok_;
    std::unordered_map<std::vector<uint32_t>, Shader, VecHash> shaders_;
    std::unordered_map<std::vector<uint32_t>, VkDescriptorSetLayout, VecHash> set_layouts_;
    std::unordered_map<std::vector<uint32_t>, VkPipelineLayout, VecHash> pipeline_layouts_;
    std::unordered_map<std::vector<uint32_t>, VkPipeline, VecHash> pipelines_;
    std::map<std::vector<uint32_t>, VkSampler> samplers_;
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool pending = false;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkBuffer stream = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        uint8_t* ptr = nullptr;
    };
    // Each frame makes two submissions (its command list, then the present): four slots let the next frame
    // record while the previous frame is still on the GPU (with two, it waited for the frame just submitted).
    static constexpr int kSlots = 4;
    Slot slots_[kSlots];
    int slot_ = 0;
    // Hashed fronts of the caches above (no allocation per lookup).
    std::unordered_map<uint64_t, VkSampler> samplers_fast_;
    std::unordered_map<uint64_t, Shader*> shaders_fast_;
    std::unordered_map<uint64_t, VkPipeline> pipelines_fast_;
    std::unordered_map<uint64_t, std::pair<std::vector<uint32_t>, VkDescriptorSetLayout>> set_layouts_fast_;
    VkDescriptorSetLayout last_pl_vs_ = VK_NULL_HANDLE, last_pl_fs_ = VK_NULL_HANDLE;
    VkPipelineLayout last_pl_ = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::pair<VkDescriptorSet, VkDescriptorSet>> desc_cache_;
    uint64_t desc_reused_ = 0;
    VkPipeline bound_pipeline_ = VK_NULL_HANDLE;
    // Last recorded dynamic state, vertex/index bindings and push constants (skipped when unchanged).
    struct DynState {
        VkViewport vp;
        VkRect2D sc;
        float blend[4];
        uint32_t stencil[6];
    };
    DynState dyn_{};
    bool dyn_valid_ = false;
    std::pair<VkBuffer, VkDeviceSize> bound_vb_[16] = {};
    std::pair<VkBuffer, VkDeviceSize> bound_ib_ = {};
    VkIndexType bound_ib_type_ = VK_INDEX_TYPE_MAX_ENUM;
    float bound_push_[4] = {};
    bool push_valid_ = false;
    VkPipelineLayout bound_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet bound_sets_[2] = {};
    uint32_t bound_dyn_[36] = {};
    uint32_t bound_dyn_count_ = 0;

    // diagnostics
    uint64_t presented_ = 0, drawn_ = 0, list_counter_ = 0, reuploads_ = 0;
    int swapchains_created_ = 0, images_created_ = 0, clear_colors_logged_ = 0, copies_logged_ = 0,
        target_sets_logged_ = 0, shaders_translated_ = 0, shader_failures_logged_ = 0, uploads_ = 0, flushes_ = 0,
        b2t_logged_ = 0, constants_logged_ = 0, handles_logged_ = 0;
    bool logged_first_image_ = false;
    std::map<std::string, uint64_t> skipped_;
    std::set<int> unknown_formats_, program_binds_logged_;
    std::map<int, Image*> fallbacks_;
    std::set<uint64_t> logged_clears_, logged_targets_;
    std::set<std::string> logged_once_, missing_ubo_logged_, ubo_binds_logged_;
    std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
};

}  // namespace

Backend* CreatePlatformBackend() { return new VulkanBackend; }

}  // namespace dsr::hle::nvn

#endif  // __ANDROID__
