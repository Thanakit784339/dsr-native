#pragma once
#include <cstddef>
#include <cstdint>

namespace dsr::hle::nvn {
// Records each distinct shader the game hands to nvnProgramSetShaders into
// Download/DSRNative/logs/dsr_shaders_<time>.bin, for building and testing the shader translator offline.
//
// File format (little endian): per shader
//   char magic[4] = "DSHD"; u32 version = 1; u32 index; u32 code_size; u32 control_size; u32 reserved;
//   u64 code_gpu_address; u8 code[code_size]; u8 control[control_size]
void DumpShader(uint64_t code_address, const void* code, size_t code_size, const void* control, size_t control_size);
}  // namespace dsr::hle::nvn
