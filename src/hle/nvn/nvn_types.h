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
#include <mutex>
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
