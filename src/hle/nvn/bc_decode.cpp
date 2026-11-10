#include "hle/nvn/bc_decode.h"

#include <algorithm>
#include <cstring>

namespace dsr::hle::nvn {
namespace {

void Rgb565(uint16_t c, int* r, int* g, int* b) {
    *r = ((c >> 11) & 31) * 255 / 31;
    *g = ((c >> 5) & 63) * 255 / 63;
    *b = (c & 31) * 255 / 31;
}

// BC1 colour block into 16 RGBA8 texels. `force4` (BC2/BC3 colour): always the four-colour mode.
void ColorBlock(const uint8_t* b, uint8_t out[16][4], bool force4) {
    const uint16_t c0 = uint16_t(b[0] | b[1] << 8), c1 = uint16_t(b[2] | b[3] << 8);
    int r[4], g[4], bl[4], a[4] = {255, 255, 255, 255};
    Rgb565(c0, &r[0], &g[0], &bl[0]);
    Rgb565(c1, &r[1], &g[1], &bl[1]);
    if (c0 > c1 || force4) {
        r[2] = (2 * r[0] + r[1]) / 3, g[2] = (2 * g[0] + g[1]) / 3, bl[2] = (2 * bl[0] + bl[1]) / 3;
        r[3] = (r[0] + 2 * r[1]) / 3, g[3] = (g[0] + 2 * g[1]) / 3, bl[3] = (bl[0] + 2 * bl[1]) / 3;
    } else {
        r[2] = (r[0] + r[1]) / 2, g[2] = (g[0] + g[1]) / 2, bl[2] = (bl[0] + bl[1]) / 2;
        r[3] = g[3] = bl[3] = 0;
        a[3] = 0;
    }
    const uint32_t idx = uint32_t(b[4]) | uint32_t(b[5]) << 8 | uint32_t(b[6]) << 16 | uint32_t(b[7]) << 24;
    for (int i = 0; i < 16; i++) {
        const int k = (idx >> (2 * i)) & 3;
        out[i][0] = uint8_t(r[k]);
        out[i][1] = uint8_t(g[k]);
        out[i][2] = uint8_t(bl[k]);
        out[i][3] = uint8_t(a[k]);
    }
}

// BC3 alpha / BC4 / BC5 channel block (8 bytes) into 16 values (0..255, or -127..127 when signed).
void ChannelBlock(const uint8_t* b, int out[16], bool is_signed) {
    int e[8];
    if (is_signed) {
        e[0] = std::max<int>(-127, int8_t(b[0]));
        e[1] = std::max<int>(-127, int8_t(b[1]));
    } else {
        e[0] = b[0];
        e[1] = b[1];
    }
    if (e[0] > e[1]) {
        for (int i = 1; i < 7; i++) e[i + 1] = ((7 - i) * e[0] + i * e[1]) / 7;
    } else {
        for (int i = 1; i < 5; i++) e[i + 1] = ((5 - i) * e[0] + i * e[1]) / 5;
        e[6] = is_signed ? -127 : 0;
        e[7] = is_signed ? 127 : 255;
    }
    uint64_t idx = 0;
    for (int i = 0; i < 6; i++) idx |= uint64_t(b[2 + i]) << (8 * i);
    for (int i = 0; i < 16; i++) out[i] = e[(idx >> (3 * i)) & 7];
}

}  // namespace

int BcDecodedBytes(BcKind k) {
    switch (k) {
        case BcKind::BC4U:
        case BcKind::BC4S: return 1;
        case BcKind::BC5U:
        case BcKind::BC5S: return 2;
        case BcKind::BC6H_U:
        case BcKind::BC6H_S: return 8;
        default: return 4;
    }
}

void BcDecode(BcKind k, const uint8_t* blocks, int blocks_x, int blocks_y, uint8_t* out, int width, int height) {
    const int bpp = BcDecodedBytes(k);
    const int block_bytes = (k == BcKind::BC1 || k == BcKind::BC4U || k == BcKind::BC4S) ? 8 : 16;
    const size_t row = size_t(width) * size_t(bpp);
    for (int by = 0; by < blocks_y; by++)
        for (int bx = 0; bx < blocks_x; bx++) {
            const uint8_t* b = blocks + (size_t(by) * size_t(blocks_x) + size_t(bx)) * size_t(block_bytes);
            uint8_t texels[16][8] = {};
            switch (k) {
                case BcKind::BC1: {
                    uint8_t c[16][4];
                    ColorBlock(b, c, false);
                    for (int i = 0; i < 16; i++) memcpy(texels[i], c[i], 4);
                    break;
                }
                case BcKind::BC2: {
                    uint8_t c[16][4];
                    ColorBlock(b + 8, c, true);
                    for (int i = 0; i < 16; i++) {
                        memcpy(texels[i], c[i], 3);
                        const int a4 = (b[i / 2] >> (4 * (i & 1))) & 15;
                        texels[i][3] = uint8_t(a4 * 17);
                    }
                    break;
                }
                case BcKind::BC3: {
                    uint8_t c[16][4];
                    int a[16];
                    ChannelBlock(b, a, false);
                    ColorBlock(b + 8, c, true);
                    for (int i = 0; i < 16; i++) {
                        memcpy(texels[i], c[i], 3);
                        texels[i][3] = uint8_t(a[i]);
                    }
                    break;
                }
                case BcKind::BC4U:
                case BcKind::BC4S: {
                    int v[16];
                    ChannelBlock(b, v, k == BcKind::BC4S);
                    for (int i = 0; i < 16; i++) texels[i][0] = uint8_t(v[i]);
                    break;
                }
                case BcKind::BC5U:
                case BcKind::BC5S: {
                    int x[16], y[16];
                    ChannelBlock(b, x, k == BcKind::BC5S);
                    ChannelBlock(b + 8, y, k == BcKind::BC5S);
                    for (int i = 0; i < 16; i++) {
                        texels[i][0] = uint8_t(x[i]);
                        texels[i][1] = uint8_t(y[i]);
                    }
                    break;
                }
                case BcKind::BC6H_U:
                case BcKind::BC6H_S: {
                    // Not decoded yet: mid grey (half-float 0.5 = 0x3800), opaque.
                    for (int i = 0; i < 16; i++) {
                        const uint16_t h[4] = {0x3800, 0x3800, 0x3800, 0x3c00};
                        memcpy(texels[i], h, 8);
                    }
                    break;
                }
                default: {  // BC7: not decoded yet, mid grey
                    for (int i = 0; i < 16; i++) texels[i][0] = texels[i][1] = texels[i][2] = 128, texels[i][3] = 255;
                    break;
                }
            }
            for (int y = 0; y < 4; y++) {
                const int py = by * 4 + y;
                if (py >= height) break;
                for (int x = 0; x < 4; x++) {
                    const int px = bx * 4 + x;
                    if (px >= width) break;
                    memcpy(out + size_t(py) * row + size_t(px) * size_t(bpp), texels[y * 4 + x], size_t(bpp));
                }
            }
        }
}

}  // namespace dsr::hle::nvn
