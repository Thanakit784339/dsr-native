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
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "hle/nvn/backend.h"
#include "hle/nvn/layout.h"
#include "hle/nvn/shader/translator.h"
#include "hle/nvn/state.h"
#include "hle/video.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "vulkan";
constexpr VkDeviceSize kStreamSize = 64ull << 20;
constexpr VkDeviceSize kMaxUniformRange = 65536;
constexpr uint32_t kCbufBase = 3;      // NVN uniform-buffer binding b is read by shaders as c[b + 3]
constexpr uint32_t kCompilerConstants = 1;  // c[1]: literal constants from the shader's control section

bool IsDepthFormat(int f) { return f >= 49 && f <= 54; }
bool HasDepth(int f) { return f >= 50 && f <= 54; }
bool HasStencil(int f) { return f == 49 || f == 53 || f == 54; }
bool IsCompressed(int f) { return f >= 65 && f <= 80; }
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
    std::map<std::vector<uint32_t>, VkImageView> views;
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

    void Execute(const CmdList& cmds) override {
        std::lock_guard<std::mutex> lk(m_);
        if (!EnsureDevice()) return;
        ResetTransient();
        for (const Cmd& c : cmds) {
            switch (c.type) {
                case CmdType::SetRenderTargets:
                    st_.num_rt = c.num_rt;
                    st_.rts = c.rts ? *c.rts : std::vector<TexRef>{};
                    LogTargets(st_.rts, st_.num_rt);
                    break;
                case CmdType::ClearColor:
                    if (c.mask == 0 || c.index < 0 || c.index >= st_.num_rt) break;
                    if (Image* img = GetImage(st_.rts[c.index], false)) {
                        Begin();
                        EndRendering();
                        ClearColorImage(*img, c.color, c.kind);
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
                    Image* src = GetImage(c.src, false);
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
                    if (data && GuestRangeValid(c.dst_addr, data->size()))
                        memcpy(reinterpret_cast<void*>(c.dst_addr), data->data(), data->size());
                    break;
                }
                case CmdType::BindProgram: {
                    auto code = std::static_pointer_cast<const ShaderCode>(c.obj);
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
            }
        }
        if (recording_) {
            EndRendering();
            SubmitAndWait(VK_NULL_HANDLE, VK_NULL_HANDLE);
            recording_ = false;
        }
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
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cai, &cmd_);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(device_, &fci, nullptr, &fence_);
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device_, &sci, nullptr, &acquire_sem_);
        vkCreateSemaphore(device_, &sci, nullptr, &render_sem_);
        if (!CreateStream() || !CreateDescriptorPool()) return false;
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
        bi.size = kStreamSize;
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
        if (vkMapMemory(device_, stream_memory_, 0, kStreamSize, 0, &p) != VK_SUCCESS) return false;
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

    bool CreateDescriptorPool() {
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384}};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets = 4096;
        ci.poolSizeCount = 2;
        ci.pPoolSizes = sizes;
        return vkCreateDescriptorPool(device_, &ci, nullptr, &desc_pool_) == VK_SUCCESS;
    }

    void ResetTransient() {
        stream_off_ = kMaxUniformRange;  // [0, 64K) is the zero buffer
        vkResetDescriptorPool(device_, desc_pool_, 0);
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
        if (img.nvn_format != t.tex.format || img.width != t.tex.width || img.height != t.tex.height) return nullptr;
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
        if (upload && !img->written && !img->uploaded) Upload(*img, t);
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

    // Fills a sampled texture from game memory: deswizzle each level/layer into the stream, then copy.
    void Upload(Image& img, const TexRef& t) {
        img.uploaded = true;
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
                     VkImageAspectFlags aspect, const uint8_t swz[4]) {
        std::vector<uint32_t> key = {uint32_t(type), base_layer, layers, levels, aspect, swz[0], swz[1], swz[2], swz[3]};
        auto it = img.views.find(key);
        if (it != img.views.end()) return it->second;
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = img.image;
        ci.viewType = type;
        ci.format = img.vk_format;
        ci.components = {Swizzle(swz[0]), Swizzle(swz[1]), Swizzle(swz[2]), Swizzle(swz[3])};
        ci.subresourceRange = {aspect, 0, levels, base_layer, layers};
        VkImageView v = VK_NULL_HANDLE;
        if (vkCreateImageView(device_, &ci, nullptr, &v) != VK_SUCCESS) {
            LogOnce("view", "vkCreateImageView failed (format %d, type %d)", img.nvn_format, int(type));
            v = VK_NULL_HANDLE;
        }
        img.views[key] = v;
        return v;
    }
    VkImageView AttachmentView(Image& img) {
        static const uint8_t kIdentity[4] = {2, 3, 4, 5};
        return View(img, img.type == VK_IMAGE_TYPE_1D ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_2D, 0, 1, 1,
                    img.aspect, kIdentity);
    }
    // View matching what the shader declared for the slot.
    VkImageView SampledView(Image& img, shader::TexDim dim, bool array, const uint8_t swz[4]) {
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
        if (array && dim != shader::TexDim::Cube) layers = uint32_t(img.layers);
        const bool type_ok = (dim == shader::TexDim::D3) == (img.type == VK_IMAGE_TYPE_3D) &&
                             (dim == shader::TexDim::D1) == (img.type == VK_IMAGE_TYPE_1D) &&
                             (dim != shader::TexDim::Cube || img.layers >= 6) && layers <= uint32_t(img.layers);
        if (!type_ok) return VK_NULL_HANDLE;
        VkImageAspectFlags aspect = img.aspect;
        if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        return View(img, type, 0, layers, uint32_t(img.levels), aspect, swz);
    }

    VkSampler Sampler(const SamplerObj& s, bool shadow) {
        std::vector<uint32_t> key(sizeof(SamplerObj) / 4 + 1);
        memcpy(key.data(), &s, sizeof(SamplerObj));
        key.back() = shadow;
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = s.mag_filter == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.minFilter = (s.min_filter == 0 || s.min_filter == 2 || s.min_filter == 4) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        ci.mipmapMode = (s.min_filter == 4 || s.min_filter == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = AddressMode(s.wrap[0]);
        ci.addressModeV = AddressMode(s.wrap[1]);
        ci.addressModeW = AddressMode(s.wrap[2]);
        ci.mipLodBias = s.lod_bias;
        ci.minLod = std::max(0.0f, s.lod_min);
        ci.maxLod = (s.min_filter <= 1) ? 0.25f : std::max(ci.minLod, std::min(s.lod_max, 16.0f));
        if (features_.samplerAnisotropy && s.max_anisotropy > 1.0f) {
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
        return smp;
    }

    // ======================================================================================================
    // command buffer helpers
    // ======================================================================================================
    void BeginCommands() {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
    }

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
        r = vkWaitForFences(device_, 1, &fence_, VK_TRUE, 5000000000ull);
        if (r != VK_SUCCESS) LogOnce("fence", "GPU did not finish within 5 s (%d)", r);
        vkResetFences(device_, 1, &fence_);
    }

    void RawBarrier(VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                    uint32_t levels = 1, uint32_t layers = 1) {
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
    void ClearColorImage(Image& img, const uint32_t color[4], uint8_t kind) {
        if (IsDepthFormat(img.nvn_format)) return;
        Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue v;
        memcpy(v.uint32, color, sizeof(v.uint32));  // float/int/uint share the bits
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        img.written = true;
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
        VkImageAspectFlags a = 0;
        if (c.depth_mask && HasDepth(img->nvn_format)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (c.stencil_mask && HasStencil(img->nvn_format)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
        a &= img->aspect;
        if (!a) return;
        Begin();
        EndRendering();
        Transition(*img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearDepthStencilValue v{std::clamp(c.depth, 0.0f, 1.0f), uint32_t(c.stencil)};
        VkImageSubresourceRange range{a, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd_, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        img->written = true;
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
        if (b2t_logged_++ < 8)
            LOGI(kTag, "buffer -> texture %dx%d at %d,%d (format %d)", w, h, r.x, r.y, img->nvn_format);
    }

    // ======================================================================================================
    // rendering
    // ======================================================================================================
    void EndRendering() {
        if (rendering_) {
            end_rendering_(cmd_);
            rendering_ = false;
        }
    }

    // Starts rendering into the current render targets (or keeps the current pass if they match).
    bool EnsureRendering(const std::vector<Image*>& colors, Image* depth, VkExtent2D* extent) {
        std::vector<uint64_t> key;
        for (Image* c : colors) key.push_back(reinterpret_cast<uint64_t>(c ? c->image : VK_NULL_HANDLE));
        key.push_back(reinterpret_cast<uint64_t>(depth ? depth->image : VK_NULL_HANDLE));
        uint32_t w = UINT32_MAX, h = UINT32_MAX;
        for (Image* c : colors)
            if (c) {
                w = std::min(w, uint32_t(c->width));
                h = std::min(h, uint32_t(c->height));
            }
        if (depth) {
            w = std::min(w, uint32_t(depth->width));
            h = std::min(h, uint32_t(depth->height));
        }
        if (w == UINT32_MAX) return false;
        *extent = {w, h};
        if (rendering_ && key == rendering_key_) return true;
        EndRendering();
        std::vector<VkRenderingAttachmentInfo> atts;
        for (Image* c : colors) {
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            if (c) {
                Transition(*c, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                a.imageView = AttachmentView(*c);
                a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                a.loadOp = c->written ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
                a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                c->written = true;
            }
            atts.push_back(a);
        }
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (depth) {
            Transition(*depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            da.imageView = AttachmentView(*depth);
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            da.loadOp = depth->written ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            da.clearValue.depthStencil = {1.0f, 0};
            depth->written = true;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {w, h}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = uint32_t(atts.size());
        ri.pColorAttachments = atts.data();
        if (depth && HasDepth(depth->nvn_format)) ri.pDepthAttachment = &da;
        if (depth && HasStencil(depth->nvn_format)) ri.pStencilAttachment = &da;
        begin_rendering_(cmd_, &ri);
        rendering_ = true;
        rendering_key_ = key;
        return true;
    }

    // ======================================================================================================
    // shaders, layouts, pipelines
    // ======================================================================================================
    Shader* GetShader(const ShaderCode& code, const shader::TranslateOptions& opt) {
        std::vector<uint32_t> key = {uint32_t(code.hash), uint32_t(code.hash >> 32), opt.int_attrib_mask,
                                     opt.signed_attrib_mask, uint32_t(opt.gl_depth_range), uint32_t(opt.alpha_func)};
        auto it = shaders_.find(key);
        if (it != shaders_.end()) return &it->second;
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

    VkDescriptorSetLayout SetLayout(VkShaderStageFlags stage, const shader::ShaderInfo& info) {
        std::vector<uint32_t> key = {uint32_t(stage), info.cbuf_mask, info.texture_mask, info.shadow_mask};
        auto it = set_layouts_.find(key);
        if (it != set_layouts_.end()) return it->second;
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i < 18; i++)
            if (info.cbuf_mask >> i & 1) b.push_back({i, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr});
        for (uint32_t i = 0; i < 32; i++) {
            if (info.texture_mask >> i & 1) b.push_back({32 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stage, nullptr});
            if (info.shadow_mask >> i & 1) b.push_back({64 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stage, nullptr});
        }
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = uint32_t(b.size());
        ci.pBindings = b.data();
        VkDescriptorSetLayout l = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(device_, &ci, nullptr, &l);
        set_layouts_[key] = l;
        return l;
    }

    VkPipelineLayout PipelineLayout(VkDescriptorSetLayout vs, VkDescriptorSetLayout fs) {
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

    void Draw(const DrawParams& p) {
        if (p.count <= 0 || p.instances <= 0) return;
        if (!begin_rendering_) return Skip("no dynamic rendering");
        if (!st_.program[0] || !st_.program[1]) return Skip("no vertex+pixel program bound");
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
        Shader* fs = GetShader(*st_.program[1], fopt);
        if (!vs->ok || !fs->ok) return Skip("shader not translated");

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
        if (StreamRemaining() < (24ull << 20)) Flush();

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
                if (!bt.valid) return Skip("texture handle not found");
                Image* img = GetImage(bt.texture, true);
                if (!img) return Skip("texture format not supported");
                for (Image* c : colors)
                    if (c == img) return Skip("texture is also a render target");
                if (depth == img) return Skip("texture is also the depth target");
                if (img->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    EndRendering();
                    Transition(*img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
                VkImageView view = SampledView(*img, info.texture_dim[slot], info.texture_array[slot], bt.texture.tex.swizzle);
                if (!view) return Skip("texture view does not match the shader");
                if (plain)
                    tex_writes.push_back({uint32_t(stage), 32 + slot,
                                          {Sampler(bt.sampler, false), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
                if (shadow)
                    tex_writes.push_back({uint32_t(stage), 64 + slot,
                                          {Sampler(bt.sampler, true), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
            }
        }

        // ---- uniform data ------------------------------------------------------------------------------------
        struct UboWrite {
            uint32_t set, binding;
            VkDescriptorBufferInfo info;
        };
        std::vector<UboWrite> ubo_writes;
        const VkDeviceSize ubo_align = std::max<VkDeviceSize>(16, props_.limits.minUniformBufferOffsetAlignment);
        for (int stage = 0; stage < 2; stage++) {
            const shader::ShaderInfo& info = stage == 0 ? vs->info : fs->info;
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
                    if (r.address && size && GuestRangeValid(r.address, size))
                        src = reinterpret_cast<const uint8_t*>(r.address);
                    else
                        size = 0;
                }
                VkDescriptorBufferInfo bi{stream_, zero_offset_, kMaxUniformRange};
                if (src && size) {
                    VkDeviceSize off = StreamAlloc((size + 15) & ~size_t(15), ubo_align);
                    memcpy(stream_ptr_ + off, src, size);
                    bi = {stream_, off, (size + 15) & ~size_t(15)};
                } else {
                    MissingUniform(stage, n);
                }
                ubo_writes.push_back({uint32_t(stage), n, bi});
            }
        }

        // ---- vertex + index data -------------------------------------------------------------------------------
        int topology_mode = p.mode;
        std::vector<uint32_t> expanded;  // converted index list (quads, line loops, u8 indices)
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
        VkPrimitiveTopology topology;
        bool use_indices = indexed;
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

        VkDeviceSize vb_offsets[16] = {};
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
            VkDeviceSize off = StreamAlloc(bytes, 16);
            memcpy(stream_ptr_ + off, reinterpret_cast<const uint8_t*>(r.address + start), size_t(bytes));
            vb_offsets[b.binding] = off;
        }
        VkDeviceSize index_off = 0;
        if (use_indices) {
            index_off = StreamAlloc(expanded.size() * 4, 4);
            memcpy(stream_ptr_ + index_off, expanded.data(), expanded.size() * 4);
        }

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
        if (!pipeline) return Skip("pipeline creation failed");

        // ---- descriptors -------------------------------------------------------------------------------------
        VkDescriptorSet sets[2] = {};
        VkDescriptorSetLayout layouts[2] = {vsl, fsl};
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = desc_pool_;
        dai.descriptorSetCount = 2;
        dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(device_, &dai, sets) != VK_SUCCESS) {
            // Pool exhausted: everything recorded so far must run first. Retry this draw from the top.
            Flush();
            if (vkAllocateDescriptorSets(device_, &dai, sets) != VK_SUCCESS) return Skip("descriptor allocation failed");
            return Draw(p);
        }
        std::vector<VkWriteDescriptorSet> writes;
        for (const UboWrite& u : ubo_writes) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[u.set];
            w.dstBinding = u.binding;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo = &u.info;
            writes.push_back(w);
        }
        for (const TexWrite& t : tex_writes) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[t.set];
            w.dstBinding = t.binding;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &t.info;
            writes.push_back(w);
        }
        if (!writes.empty()) vkUpdateDescriptorSets(device_, uint32_t(writes.size()), writes.data(), 0, nullptr);

        // ---- record --------------------------------------------------------------------------------------------
        if (!EnsureRendering(colors, depth, &extent)) return Skip("no render target");
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, 0, nullptr);
        for (const VkVertexInputBindingDescription& b : vin.bindings)
            vkCmdBindVertexBuffers(cmd_, b.binding, 1, &stream_, &vb_offsets[b.binding]);
        float push[4] = {st_.alpha_ref, 0, 0, 0};
        vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);
        SetDynamicState(extent, modes);
        if (use_indices) {
            vkCmdBindIndexBuffer(cmd_, stream_, index_off, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd_, uint32_t(expanded.size()), uint32_t(p.instances), 0, 0, uint32_t(p.base_instance));
        } else {
            vkCmdDraw(cmd_, uint32_t(p.count), uint32_t(p.instances), 0, uint32_t(p.base_instance));
        }
        drawn_++;
        if (drawn_ == 1) LOGI(kTag, "first draw recorded");
    }

    void SetDynamicState(VkExtent2D extent, const DeviceModes& modes) {
        // NVN viewports/scissors are in window coordinates; with the OpenGL-style lower-left origin, flip Y
        // with a negative-height viewport so the image ends up upright in Vulkan's top-left space.
        const bool flip = modes.origin_mode == 0;
        float vx = 0, vy = 0, vw = float(extent.width), vh = float(extent.height);
        if (st_.viewport_set && st_.viewport[2] > 0 && st_.viewport[3] > 0) {
            vx = st_.viewport[0];
            vy = st_.viewport[1];
            vw = st_.viewport[2];
            vh = st_.viewport[3];
        }
        VkViewport vp;
        vp.x = vx;
        vp.width = vw;
        if (flip) {
            vp.y = float(extent.height) - vy;
            vp.height = -vh;
        } else {
            vp.y = vy;
            vp.height = vh;
        }
        vp.minDepth = std::clamp(st_.depth_range[0], 0.0f, 1.0f);
        vp.maxDepth = std::clamp(st_.depth_range[1], 0.0f, 1.0f);
        vkCmdSetViewport(cmd_, 0, 1, &vp);
        VkRect2D sc{{0, 0}, extent};
        if (st_.scissor_set && st_.scissor[2] > 0 && st_.scissor[3] > 0) {
            int x = int(st_.scissor[0]), y = int(st_.scissor[1]), w = int(st_.scissor[2]), h = int(st_.scissor[3]);
            if (flip) y = int(extent.height) - (y + h);
            x = std::max(0, x);
            y = std::max(0, y);
            w = std::max(0, std::min(w, int(extent.width) - x));
            h = std::max(0, std::min(h, int(extent.height) - y));
            sc = {{x, y}, {uint32_t(w), uint32_t(h)}};
        }
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        vkCmdSetBlendConstants(cmd_, st_.blend_color);
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_BIT, st_.stencil_ref[0]);
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_BACK_BIT, st_.stencil_ref[1]);
        vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_FRONT_BIT, st_.stencil_write[0]);
        vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_BACK_BIT, st_.stencil_write[1]);
        vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_FRONT_BIT, st_.stencil_compare[0]);
        vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_BACK_BIT, st_.stencil_compare[1]);
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

    VkPipeline GetPipeline(Shader* vs, Shader* fs, const VertexInput& vin, VkPrimitiveTopology topology,
                           const std::vector<VkFormat>& color_formats, int depth_nvn_format, VkFormat depth_format,
                           VkPipelineLayout layout) {
        // Key: everything that shapes the pipeline.
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
        if (it != pipelines_.end()) return it->second;

        VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                     {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs->module;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs->module;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = uint32_t(vin.bindings.size());
        vi.pVertexBindingDescriptions = vin.bindings.data();
        vi.vertexAttributeDescriptionCount = uint32_t(vin.attributes.size());
        vi.pVertexAttributeDescriptions = vin.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topology;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = (st_.polygon.mode == 0 && features_.fillModeNonSolid) ? VK_POLYGON_MODE_POINT
                         : (st_.polygon.mode == 1 && features_.fillModeNonSolid) ? VK_POLYGON_MODE_LINE
                                                                                 : VK_POLYGON_MODE_FILL;
        rs.cullMode = st_.polygon.cull == 1 ? VK_CULL_MODE_FRONT_BIT
                      : st_.polygon.cull == 2 ? VK_CULL_MODE_BACK_BIT
                      : st_.polygon.cull == 3 ? VK_CULL_MODE_FRONT_AND_BACK
                                              : VK_CULL_MODE_NONE;
        rs.frontFace = st_.polygon.front_face == 0 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        const bool has_depth = depth_format != VK_FORMAT_UNDEFINED && HasDepth(depth_nvn_format);
        const bool has_stencil = depth_format != VK_FORMAT_UNDEFINED && HasStencil(depth_nvn_format);
        ds.depthTestEnable = has_depth && st_.depth.depth_test;
        ds.depthWriteEnable = has_depth && st_.depth.depth_test && st_.depth.depth_write;
        ds.depthCompareOp = CompareOp(st_.depth.depth_func);
        ds.stencilTestEnable = has_stencil && st_.depth.stencil_test;
        for (int face = 0; face < 2; face++) {
            VkStencilOpState& s = face ? ds.back : ds.front;
            s.failOp = StencilOp(st_.depth.stencil_fail[face]);
            s.passOp = StencilOp(st_.depth.depth_pass[face]);
            s.depthFailOp = StencilOp(st_.depth.depth_fail[face]);
            s.compareOp = CompareOp(st_.depth.stencil_func[face]);
            s.compareMask = 0xFF;
            s.writeMask = 0xFF;
        }
        std::vector<VkPipelineColorBlendAttachmentState> blend(color_formats.size());
        for (size_t i = 0; i < blend.size(); i++) {
            VkPipelineColorBlendAttachmentState& b = blend[i];
            const BlendStateObj& bs = st_.blend[features_.independentBlend ? i : 0];
            b.blendEnable = (st_.color.blend_enable >> (features_.independentBlend ? i : 0)) & 1;
            b.srcColorBlendFactor = BlendFactor(bs.src_rgb);
            b.dstColorBlendFactor = BlendFactor(bs.dst_rgb);
            b.colorBlendOp = BlendOp(bs.eq_rgb);
            b.srcAlphaBlendFactor = BlendFactor(bs.src_a);
            b.dstAlphaBlendFactor = BlendFactor(bs.dst_a);
            b.alphaBlendOp = BlendOp(bs.eq_a);
            b.colorWriteMask = (st_.channel.mask >> (4 * i)) & 0xF;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = uint32_t(blend.size());
        cb.pAttachments = blend.data();
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(sizeof(dyn) / sizeof(dyn[0]));
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rinfo.colorAttachmentCount = uint32_t(color_formats.size());
        rinfo.pColorAttachmentFormats = color_formats.data();
        rinfo.depthAttachmentFormat = has_depth ? depth_format : VK_FORMAT_UNDEFINED;
        rinfo.stencilAttachmentFormat = has_stencil ? depth_format : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.pNext = &rinfo;
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState = &vp;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds;
        ci.pColorBlendState = &cb;
        ci.pDynamicState = &dy;
        ci.layout = layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
        if (r != VK_SUCCESS) {
            LogOnce("pipeline", "vkCreateGraphicsPipelines failed: %d", r);
            pipeline = VK_NULL_HANDLE;
        }
        pipelines_[key] = pipeline;
        if (pipeline && pipelines_.size() <= 8)
            LOGI(kTag, "pipeline %zu: %zu attributes, %zu targets, depth format %d, topology %d, blend %s",
                 pipelines_.size(), vin.attributes.size(), color_formats.size(), int(depth_format), int(topology),
                 (st_.color.blend_enable & 1) ? "on" : "off");
        return pipeline;
    }

    // c[1]: the literal constants NVN's compiler stored with the shader. The control section header gives
    // its size (word 5) and the constants' size (word 11); the constants sit at the end of the section.
    void ShaderConstants(const ShaderCode& code, const uint8_t** data, size_t* size) {
        *data = nullptr;
        *size = 0;
        if (code.control.size() < 64) return;
        uint32_t hdr[16];
        memcpy(hdr, code.control.data(), sizeof(hdr));
        const uint32_t total = hdr[5], csize = hdr[11];
        if (hdr[0] != 0x98761234u || !csize || csize > total || total > code.control.size()) return;
        *data = code.control.data() + (total - csize);
        *size = csize;
        if (constants_logged_ < 4) {
            constants_logged_++;
            float f[4] = {};
            memcpy(f, *data, std::min<size_t>(16, csize));
            LOGI(kTag, "shader constants c[1] (%u bytes): %g %g %g %g ...", csize, f[0], f[1], f[2], f[3]);
        }
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
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;

    void* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    bool swapchain_dirty_ = false;
    VkExtent2D sc_extent_{};
    std::vector<VkImage> sc_images_;

    DrawState st_;
    std::unordered_map<uint64_t, Image> images_;
    std::unordered_map<int, std::pair<VkFormat, VkFormatFeatureFlags>> format_cache_;
    std::unordered_map<VkFormat, bool> vertex_format_ok_;
    std::unordered_map<std::vector<uint32_t>, Shader, VecHash> shaders_;
    std::unordered_map<std::vector<uint32_t>, VkDescriptorSetLayout, VecHash> set_layouts_;
    std::unordered_map<std::vector<uint32_t>, VkPipelineLayout, VecHash> pipeline_layouts_;
    std::unordered_map<std::vector<uint32_t>, VkPipeline, VecHash> pipelines_;
    std::map<std::vector<uint32_t>, VkSampler> samplers_;

    // diagnostics
    uint64_t presented_ = 0, drawn_ = 0;
    int swapchains_created_ = 0, images_created_ = 0, clear_colors_logged_ = 0, copies_logged_ = 0,
        target_sets_logged_ = 0, shaders_translated_ = 0, shader_failures_logged_ = 0, uploads_ = 0, flushes_ = 0,
        b2t_logged_ = 0, constants_logged_ = 0, handles_logged_ = 0;
    bool logged_first_image_ = false;
    std::map<std::string, uint64_t> skipped_;
    std::set<int> unknown_formats_;
    std::set<uint64_t> logged_clears_, logged_targets_;
    std::set<std::string> logged_once_, missing_ubo_logged_, ubo_binds_logged_;
    std::chrono::steady_clock::time_point last_report_ = std::chrono::steady_clock::now();
};

}  // namespace

Backend* CreatePlatformBackend() { return new VulkanBackend; }

}  // namespace dsr::hle::nvn

#endif  // __ANDROID__
