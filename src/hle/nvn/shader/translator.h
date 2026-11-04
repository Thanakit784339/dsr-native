#pragma once
// Maxwell (Tegra X1, SM 5.3) -> SPIR-V translator for the shaders NVN games hand to
// nvnProgramSetShaders. Input is the NVN code blob: a 0x30-byte NVN header, the 0x50-byte Shader
// Program Header (SPH) and the instruction stream at +0x80.
//
// Scope: the instruction subset Dark Souls Remastered uses (float ALU, MUFU, interpolation, bindless
// texture sampling incl. depth compare, attribute load/store, constant-buffer loads, the integer ops
// its vertex shaders use for skinning). Control flow is limited to straight-line code with predicated
// instructions; shaders with branches report failure for now.
//
// Resource interface (descriptor set = 0 for vertex, 1 for fragment):
//   constant buffer c[n]                      -> uniform block, binding n            (n < 18)
//   texture slot k (handle from c[13][0x20+8k]) -> combined image sampler, binding 32+k
//   ... same slot sampled with depth compare  -> combined image sampler, binding 64+k
//   vertex attribute (a[0x80 + 16*i])         -> location i
//   varyings (generic attribute i)            -> location i
//   fragment render target t                  -> location t

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dsr::hle::nvn::shader {

enum class Stage { Vertex, Fragment };

enum class TexDim : uint8_t { None, D1, D2, D3, Cube };

struct TranslateOptions {
    uint32_t int_attrib_mask = 0;     // vertex attributes fetched as integers (format is *I / *UI)
    uint32_t signed_attrib_mask = 0;  // ... of which signed
    bool gl_depth_range = true;       // NVN depth mode -1..1: remap clip z to Vulkan's 0..1
};

struct ShaderInfo {
    Stage stage = Stage::Vertex;
    uint32_t cbuf_mask = 0;             // constant buffers read
    uint32_t texture_mask = 0;          // texture slots sampled without compare
    uint32_t shadow_mask = 0;           // texture slots sampled with depth compare
    TexDim texture_dim[32] = {};        // per slot
    bool texture_array[32] = {};
    uint32_t input_mask = 0;            // vertex attributes / varyings read (by location)
    uint32_t flat_mask = 0;             // varyings read with constant (flat) interpolation
    uint32_t output_mask = 0;           // varyings written (vertex) or render targets written (fragment)
    bool writes_depth = false;
    bool uses_discard = false;
    bool writes_point_size = false;
    uint32_t instruction_count = 0;
};

struct TranslateResult {
    bool ok = false;
    std::string error;
    std::vector<uint32_t> spirv;
    ShaderInfo info;
};

// Reads the stage from the SPH (type 1 = vertex, 5 = pixel). Returns false for other stages.
bool GetStage(const uint8_t* code, size_t size, Stage* stage);

TranslateResult Translate(const uint8_t* code, size_t size, const TranslateOptions& options);

}  // namespace dsr::hle::nvn::shader
