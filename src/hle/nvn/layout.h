#pragma once
// Tegra X1 texture memory layouts.
//
// Block-linear: memory is split into GOBs ("groups of bytes", 64 bytes x 8 rows = 512 bytes); GOBs are
// stacked into blocks of 2^bh GOBs vertically and 2^bd GOBs in depth; blocks are laid out left to
// right, then top to bottom, then front to back. Inside a GOB, 16-byte pieces follow a fixed pattern.
// Each mip level picks the largest block height/depth (up to the level-0 choice) that is not wasted on
// that level, levels follow one another, and array layers / cube faces are spaced by the whole mip
// chain rounded up to a level-0 block.
//
// Pitch-linear (textures with a row stride): plain rows.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hle/nvn/nvn_types.h"

namespace dsr::hle::nvn {

struct LevelLayout {
    size_t offset = 0;      // from the start of a layer
    size_t size = 0;
    int width = 1, height = 1, depth = 1;  // in texels
    int row_bytes = 0;      // bytes per row of blocks (compressed blocks count as one element)
    int rows = 0;           // rows of blocks
    int bh_log2 = 0, bd_log2 = 0;
};

struct TextureLayout {
    bool linear = false;
    int layers = 1;
    size_t layer_stride = 0;
    size_t total = 0;
    std::vector<LevelLayout> levels;
};

TextureLayout ComputeLayout(const TextureState& t);

// Converts one level of one layer between block-linear (src/dst at the level's offset) and tightly
// packed rows (row_bytes per row, rows per slice, depth slices).
void DeswizzleLevel(uint8_t* linear, const uint8_t* swizzled, const LevelLayout& l);
void SwizzleLevel(uint8_t* swizzled, const uint8_t* linear, const LevelLayout& l);

// Writes a box of tightly or explicitly pitched linear data into one level of one layer
// (x in bytes of block rows, y in block rows, z in slices).
void WriteLevelRegion(uint8_t* level_base, const LevelLayout& l, bool linear, uint32_t x_bytes, uint32_t y,
                      uint32_t z, uint32_t w_bytes, uint32_t rows, uint32_t depth, const uint8_t* src,
                      size_t row_pitch, size_t slice_pitch);

}  // namespace dsr::hle::nvn
