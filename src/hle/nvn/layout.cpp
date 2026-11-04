#include "hle/nvn/layout.h"

#include <algorithm>
#include <cstring>

namespace dsr::hle::nvn {
namespace {
constexpr int kGobWidth = 64, kGobHeight = 8, kGobSize = 512;

size_t DivCeil(size_t a, size_t b) { return (a + b - 1) / b; }

// Smallest block height (log2, in GOBs) that does not exceed what `rows` needs, capped at `max_log2`.
int FitBlockHeight(int rows, int max_log2) {
    int bh = max_log2;
    while (bh > 0 && rows <= (kGobHeight << (bh - 1))) bh--;
    return bh;
}
int FitBlockDepth(int depth, int max_log2) {
    int bd = max_log2;
    while (bd > 0 && depth <= (1 << (bd - 1))) bd--;
    return bd;
}

// Byte offset inside a GOB of byte x (0..63) on row y (0..7).
inline uint32_t GobOffset(uint32_t x, uint32_t y) {
    return ((x % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + (x % 16);
}

template <bool kToLinear>
void Convert(uint8_t* linear, uint8_t* swizzled, const LevelLayout& l) {
    const uint32_t gobs_x = uint32_t(DivCeil(size_t(l.row_bytes), kGobWidth));
    const uint32_t block_size = gobs_x << (9 + l.bh_log2 + l.bd_log2);  // one row of blocks
    const uint32_t slice_size = uint32_t(DivCeil(size_t(l.rows), size_t(kGobHeight) << l.bh_log2)) * block_size;
    const uint32_t bh_mask = (1u << l.bh_log2) - 1, bd_mask = (1u << l.bd_log2) - 1;
    const uint32_t x_shift = 9 + l.bh_log2 + l.bd_log2;
    for (int z = 0; z < l.depth; z++) {
        const uint32_t off_z = (uint32_t(z) >> l.bd_log2) * slice_size + ((uint32_t(z) & bd_mask) << (9 + l.bh_log2));
        for (int y = 0; y < l.rows; y++) {
            const uint32_t gob_y = uint32_t(y) / kGobHeight;
            const uint32_t off_y = (gob_y >> l.bh_log2) * block_size + ((gob_y & bh_mask) << 9);
            uint8_t* lin_row = linear + (size_t(z) * l.rows + size_t(y)) * size_t(l.row_bytes);
            for (int x = 0; x < l.row_bytes; x += 16) {
                const uint32_t off = off_z + off_y + ((uint32_t(x) / kGobWidth) << x_shift) + GobOffset(uint32_t(x), uint32_t(y));
                const int n = std::min(16, l.row_bytes - x);
                if (kToLinear)
                    memcpy(lin_row + x, swizzled + off, size_t(n));
                else
                    memcpy(swizzled + off, lin_row + x, size_t(n));
            }
        }
    }
}
}  // namespace

TextureLayout ComputeLayout(const TextureState& t) {
    TextureLayout out;
    const FormatInfo fi = GetFormatInfo(t.format);
    const int levels = std::max(1, t.levels);
    int depth = std::max(1, t.depth);
    // Array-like targets keep their layers in `depth`; 3D textures have real depth.
    const bool is_3d = t.target == kTarget3D;
    out.layers = is_3d ? 1 : depth;
    if (t.target == kTargetCubemap) out.layers = 6;
    if (t.target == kTargetCubemapArray) out.layers = depth * 6;
    if (!is_3d) depth = 1;
    const int samples = std::max(1, t.samples);

    if (t.stride > 0) {  // pitch-linear
        out.linear = true;
        LevelLayout l;
        l.width = t.width;
        l.height = t.height;
        l.row_bytes = t.stride;
        l.rows = int(DivCeil(size_t(t.height), size_t(fi.block_h)));
        l.size = size_t(t.stride) * size_t(l.rows);
        out.levels.push_back(l);
        out.layer_stride = l.size;
        out.total = l.size * size_t(out.layers);
        return out;
    }

    const int rows0 = int(DivCeil(size_t(std::max(1, t.height)), size_t(fi.block_h)));
    const int bh0 = FitBlockHeight(rows0, 4);
    const int bd0 = is_3d ? FitBlockDepth(depth, 5) : 0;
    size_t offset = 0;
    for (int lv = 0; lv < levels; lv++) {
        LevelLayout l;
        l.width = std::max(1, t.width >> lv);
        l.height = std::max(1, t.height >> lv);
        l.depth = is_3d ? std::max(1, depth >> lv) : 1;
        l.row_bytes = int(DivCeil(size_t(l.width), size_t(fi.block_w))) * fi.bytes_per_block * samples;
        l.rows = int(DivCeil(size_t(l.height), size_t(fi.block_h)));
        l.bh_log2 = FitBlockHeight(l.rows, bh0);
        l.bd_log2 = is_3d ? FitBlockDepth(l.depth, bd0) : 0;
        const size_t gobs_x = DivCeil(size_t(l.row_bytes), kGobWidth);
        const size_t blocks_y = DivCeil(size_t(l.rows), size_t(kGobHeight) << l.bh_log2);
        const size_t blocks_z = DivCeil(size_t(l.depth), size_t(1) << l.bd_log2);
        l.size = gobs_x * blocks_y * blocks_z * (size_t(kGobSize) << (l.bh_log2 + l.bd_log2));
        l.offset = offset;
        offset += l.size;
        out.levels.push_back(l);
    }
    const size_t align = size_t(kGobSize) << (bh0 + bd0);
    out.layer_stride = out.layers > 1 ? DivCeil(offset, align) * align : offset;
    out.total = out.layer_stride * size_t(out.layers);
    return out;
}

void WriteLevelRegion(uint8_t* level_base, const LevelLayout& l, bool linear, uint32_t x_bytes, uint32_t y0,
                      uint32_t z0, uint32_t w_bytes, uint32_t rows, uint32_t depth, const uint8_t* src,
                      size_t row_pitch, size_t slice_pitch) {
    if (linear) {
        for (uint32_t z = 0; z < depth; z++)
            for (uint32_t y = 0; y < rows; y++)
                memcpy(level_base + (size_t(z0 + z) * l.rows + y0 + y) * size_t(l.row_bytes) + x_bytes,
                       src + z * slice_pitch + y * row_pitch, w_bytes);
        return;
    }
    const uint32_t gobs_x = uint32_t(DivCeil(size_t(l.row_bytes), kGobWidth));
    const uint32_t block_size = gobs_x << (9 + l.bh_log2 + l.bd_log2);
    const uint32_t slice_size = uint32_t(DivCeil(size_t(l.rows), size_t(kGobHeight) << l.bh_log2)) * block_size;
    const uint32_t bh_mask = (1u << l.bh_log2) - 1, bd_mask = (1u << l.bd_log2) - 1;
    const uint32_t x_shift = 9 + l.bh_log2 + l.bd_log2;
    for (uint32_t zi = 0; zi < depth; zi++) {
        const uint32_t z = z0 + zi;
        const uint32_t off_z = (z >> l.bd_log2) * slice_size + ((z & bd_mask) << (9 + l.bh_log2));
        for (uint32_t yi = 0; yi < rows; yi++) {
            const uint32_t y = y0 + yi;
            const uint32_t gob_y = y / kGobHeight;
            const uint32_t off_y = (gob_y >> l.bh_log2) * block_size + ((gob_y & bh_mask) << 9);
            const uint8_t* row = src + zi * slice_pitch + yi * row_pitch;
            uint32_t x = 0;
            while (x < w_bytes) {
                const uint32_t bx = x_bytes + x;
                const uint32_t n = std::min(16u - (bx % 16u), w_bytes - x);  // stay inside a 16-byte piece
                const uint32_t off = off_z + off_y + ((bx / kGobWidth) << x_shift) + GobOffset(bx, y);
                memcpy(level_base + off, row + x, n);
                x += n;
            }
        }
    }
}

void DeswizzleLevel(uint8_t* linear, const uint8_t* swizzled, const LevelLayout& l) {
    Convert<true>(linear, const_cast<uint8_t*>(swizzled), l);
}
void SwizzleLevel(uint8_t* swizzled, const uint8_t* linear, const LevelLayout& l) {
    Convert<false>(const_cast<uint8_t*>(linear), swizzled, l);
}

}  // namespace dsr::hle::nvn
