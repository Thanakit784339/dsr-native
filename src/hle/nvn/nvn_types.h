#pragma once
// NVN constants and the host-side object model shared by the NVN backends.
//
// NVN objects (NVNdevice, NVNtexture, builders, ...) are memory blocks owned by the game; their layout
// is private to the driver. We never interpret their contents: each object's state lives in a host
// table keyed by the object's address. Builders are often stack objects whose addresses are reused,
// which is fine because every builder sequence starts with SetDefaults, which resets the entry.
//
// Enum values below were taken from how this game uses them (e.g. device-info queries whose results
// are used as texture/sampler descriptor sizes and alignments) — see tools/nvn_map.py.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

namespace dsr::hle::nvn {

enum DeviceInfo : int {
    kApiMajorVersion = 0,
    kApiMinorVersion = 1,
    kUniformBufferAlignment = 4,
    kTextureDescriptorSize = 15,
    kSamplerDescriptorSize = 16,
    kCommandBufferCommandAlignment = 19,
    kCommandBufferControlAlignment = 20,
    kLinearTextureStrideAlignment = 56,
    kQueueCommandMemoryDefaultSize = 82,
};

enum TextureTarget : int {
    kTarget1D = 0, kTarget2D = 1, kTarget3D = 2, kTarget1DArray = 3, kTarget2DArray = 4,
    kTarget2DMultisample = 5, kTarget2DMultisampleArray = 6, kTargetRectangle = 7,
    kTargetCubemap = 8, kTargetCubemapArray = 9, kTargetBuffer = 10,
};

struct FormatInfo {
    int block_w, block_h, bytes_per_block;
};
FormatInfo GetFormatInfo(int format);

// ---- object state --------------------------------------------------------------------------------
struct MemoryPoolState {
    uint8_t* storage = nullptr;
    size_t size = 0;
    int flags = 0;
};

struct BufferState {
    const void* pool = nullptr;  // NVNmemoryPool*
    int64_t offset = 0;
    size_t size = 0;
};

struct TextureState {
    const void* pool = nullptr;
    int64_t offset = 0;
    int target = kTarget2D;
    int format = 0;
    int width = 1, height = 1, depth = 1;
    int levels = 1;
    int samples = 0;
    int flags = 0;
    int stride = 0;
    uint8_t swizzle[4] = {2, 3, 4, 5};  // NVNtextureSwizzle: 0 ZERO, 1 ONE, 2 R, 3 G, 4 B, 5 A
    const void* packaged_data = nullptr;
    uint8_t packaged_layout[8] = {};
};

// ---- draw state objects (NVN keeps these in game memory; we mirror them, keyed by address) -------------
// Enum values follow NVN (GL-style ordering; see vk_backend.cpp for the Vulkan mapping).
struct BlendStateObj {
    int target = 0;
    int src_rgb = 2, dst_rgb = 1, src_a = 2, dst_a = 1;  // ONE, ZERO
    int eq_rgb = 1, eq_a = 1;                             // ADD
};
struct ColorStateObj {
    uint32_t blend_enable = 0;  // bit per render target
    int logic_op = 3;           // COPY
    int alpha_func = 8;         // ALWAYS
};
struct ChannelMaskStateObj {
    uint32_t mask = 0xFFFFFFFF;  // 4 bits (RGBA) per render target
};
struct DepthStencilStateObj {
    bool depth_test = false, depth_write = true, stencil_test = false;
    int depth_func = 2;  // LESS
    int stencil_func[2] = {8, 8};
    int stencil_fail[2] = {1, 1}, depth_fail[2] = {1, 1}, depth_pass[2] = {1, 1};  // KEEP
};
struct PolygonStateObj {
    int cull = 0;        // NONE
    int front_face = 1;  // CCW
    int mode = 2;        // FILL
    int offset_enables = 0;
};
struct MultisampleStateObj {
    bool enable = false;
    int samples = 0;
};
struct VertexAttribObj {
    int format = 0;
    int offset = 0;
    int stream = 0;
};
// NVNtextureView: selects part of a texture (one mip level / a layer range), or reinterprets it.
struct TextureViewObj {
    int base_level = 0, num_levels = 0;  // 0 levels: all (view leaves levels alone)
    int base_layer = 0, num_layers = 0;  // 0 layers: all
    int target = -1;                      // -1: the texture's own
    int format = 0;                       // 0: the texture's own (e.g. an sRGB view of an RGBA8 texture)
    bool has_levels = false, has_layers = false;
};
struct VertexStreamObj {
    int stride = 0;
    int divisor = 0;
};
struct SamplerObj {
    int min_filter = 1, mag_filter = 1;  // LINEAR
    int wrap[3] = {1, 1, 1};             // REPEAT
    float lod_min = -1000.0f, lod_max = 1000.0f, lod_bias = 0.0f;
    int compare_mode = 0, compare_func = 8;
    float max_anisotropy = 1.0f;
    bool operator==(const SamplerObj& o) const { return memcmp(this, &o, sizeof(*this)) == 0; }
};

struct ShaderCode {
    uint64_t hash = 0;
    uint64_t address = 0;  // GPU address of the code (matches the address in the shader dump file)
    int stage = -1;  // 0 vertex, 1 fragment (from the SPH)
    std::vector<uint8_t> code;
    std::vector<uint8_t> control;  // NVN control section
    // The compiler's literal pool, read by the shader as c[1]. It lives in the GPU section after the code, at
    // the offset/size given by control words 0x700 / 0x6fc.
    std::vector<uint8_t> constants;
};

struct WindowState {
    int num_textures = 0;
    const void* textures[4] = {};
    int next = 0;
    int present_interval = 1;
};

struct QueueBuilderState {
    size_t command_memory = 0x10000;
    size_t compute_memory = 0x40000;
};

// Generic address-keyed table.
template <typename T>
class Table {
public:
    T& Get(const void* key) {
        std::lock_guard<std::mutex> lk(m_);
        return map_[key];
    }
    T Copy(const void* key) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = map_.find(key);
        return it == map_.end() ? T{} : it->second;
    }
    void Reset(const void* key) {
        std::lock_guard<std::mutex> lk(m_);
        map_[key] = T{};
    }
    void Erase(const void* key) {
        std::lock_guard<std::mutex> lk(m_);
        map_.erase(key);
    }

private:
    std::mutex m_;
    std::unordered_map<const void*, T> map_;
};

}  // namespace dsr::hle::nvn
