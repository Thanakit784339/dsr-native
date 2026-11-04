#pragma once
// The NVN front end (null_driver.cpp + commands.cpp) records what the game asks for into host command
// lists; a Backend executes them. On Android the backend is Vulkan (vk_backend.cpp); elsewhere it is a
// null backend that only counts, so the recorder can be unit-tested on any host.

#include <cstdint>
#include <memory>
#include <vector>

#include "hle/nvn/nvn_types.h"

namespace dsr::hle::nvn {

// A texture as seen at record time: its description plus the address of its storage. Images on the
// host GPU are keyed by storage address, so two NVN texture objects over the same memory share one.
struct TexRef {
    bool valid = false;
    TextureState tex;
    uint64_t addr = 0;
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
    Draw,               // counted only (shaders come later)
};

enum ClearKind : uint8_t { kClearFloat = 0, kClearInt = 1, kClearUint = 2 };

struct Cmd {
    CmdType type{};
    uint8_t kind = kClearFloat;
    bool depth_mask = false;
    int index = 0, mask = 0, stencil = 0, stencil_mask = 0, flags = 0;
    uint32_t color[4] = {};
    float depth = 0;
    // SetRenderTargets
    int num_rt = 0;
    std::shared_ptr<const std::vector<TexRef>> rts;  // colors[0..num_rt) then depth at [num_rt]
    // copies / clears of explicit textures
    TexRef src, dst;
    CopyRegion src_region, dst_region;
};

using CmdList = std::vector<Cmd>;

class Backend {
public:
    virtual ~Backend() = default;
    virtual void Execute(const CmdList& cmds) = 0;
    // Show the window texture (null ref: nothing rendered yet).
    virtual void Present(const TexRef& texture) = 0;
    virtual const char* Name() const = 0;
};

Backend& GetBackend();
void SetBackendForTesting(Backend* b);  // nullptr restores the default
Backend* CreatePlatformBackend();       // Vulkan on Android, null elsewhere

// Lookups provided by null_driver.cpp.
TexRef MakeTexRef(const void* nvn_texture);
TexRef WindowTexture(const void* nvn_window, int index);

// Per-frame statistics, reported every 10 s by the present path.
struct GpuStats {
    uint64_t submits = 0, commands = 0, draws = 0, clears = 0, copies = 0, rt_binds = 0;
};
GpuStats& Stats();

}  // namespace dsr::hle::nvn
