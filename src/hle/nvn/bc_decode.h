// Software decoding of BC (DXT) compressed textures, for GPUs without BC support (Adreno before the 7xx series,
// Mali). Textures are decoded once at upload into uncompressed formats.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dsr::hle::nvn {

enum class BcKind : uint8_t {
    None,
    BC1,       // -> RGBA8
    BC2,       // -> RGBA8
    BC3,       // -> RGBA8
    BC4U,      // -> R8 unorm
    BC4S,      // -> R8 snorm
    BC5U,      // -> RG8 unorm
    BC5S,      // -> RG8 snorm
    BC6H_U,    // -> RGBA16F
    BC6H_S,    // -> RGBA16F
    BC7,       // -> RGBA8
};

// Bytes per decoded texel of a kind.
int BcDecodedBytes(BcKind k);

// Decodes `blocks_x` x `blocks_y` blocks (tightly packed, `block_bytes` = 8 or 16 each) into texels, writing a
// `width` x `height` image (rows of width * BcDecodedBytes bytes); edge blocks are cropped.
void BcDecode(BcKind k, const uint8_t* blocks, int blocks_x, int blocks_y, uint8_t* out, int width, int height);

}  // namespace dsr::hle::nvn
