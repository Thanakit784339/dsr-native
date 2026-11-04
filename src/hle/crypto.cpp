// nn::crypto MD5 / SHA-1. The game embeds Md5Impl/Sha1Impl objects whose exact size we do not rely on:
// hash state lives in host objects keyed by the guest object's address.
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "hle/registry.h"

namespace {

inline uint32_t Rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

struct Md5 {
    uint32_t s[4];
    uint64_t len;
    uint8_t buf[64];
    void Init() {
        s[0] = 0x67452301; s[1] = 0xefcdab89; s[2] = 0x98badcfe; s[3] = 0x10325476;
        len = 0;
    }
    void Block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
        static const int R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                  5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
        uint32_t m[16];
        memcpy(m, p, 64);  // little-endian host
        uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
        for (int i = 0; i < 64; i++) {
            uint32_t f;
            int g;
            if (i < 16) { f = (b & c) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) & 15; }
            else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) & 15; }
            else { f = c ^ (b | ~d); g = (7 * i) & 15; }
            uint32_t t = d;
            d = c;
            c = b;
            b = b + Rol(a + f + K[i] + m[g], R[i]);
            a = t;
        }
        s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    }
    void Update(const uint8_t* p, size_t n) {
        size_t have = len & 63;
        len += n;
        if (have) {
            size_t k = 64 - have < n ? 64 - have : n;
            memcpy(buf + have, p, k);
            p += k; n -= k; have += k;
            if (have < 64) return;
            Block(buf);
        }
        for (; n >= 64; p += 64, n -= 64) Block(p);
        memcpy(buf, p, n);
    }
    void Final(uint8_t out[16]) {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80, zero = 0;
        Update(&pad, 1);
        while ((len & 63) != 56) Update(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; i++) l[i] = uint8_t(bits >> (8 * i));
        Update(l, 8);
        memcpy(out, s, 16);
    }
};

struct Sha1 {
    uint32_t h[5];
    uint64_t len;
    uint8_t buf[64];
    void Init() {
        h[0] = 0x67452301; h[1] = 0xEFCDAB89; h[2] = 0x98BADCFE; h[3] = 0x10325476; h[4] = 0xC3D2E1F0;
        len = 0;
    }
    void Block(const uint8_t* p) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 80; i++) w[i] = Rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t t = Rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = Rol(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    void Update(const uint8_t* p, size_t n) {
        size_t have = len & 63;
        len += n;
        if (have) {
            size_t k = 64 - have < n ? 64 - have : n;
            memcpy(buf + have, p, k);
            p += k; n -= k; have += k;
            if (have < 64) return;
            Block(buf);
        }
        for (; n >= 64; p += 64, n -= 64) Block(p);
        memcpy(buf, p, n);
    }
    void Final(uint8_t out[20]) {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80, zero = 0;
        Update(&pad, 1);
        while ((len & 63) != 56) Update(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; i++) l[i] = uint8_t(bits >> (56 - 8 * i));
        Update(l, 8);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 4; j++) out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
    }
};

template <typename T>
T& StateFor(const void* guest) {
    static std::mutex m;
    static auto* map = new std::unordered_map<const void*, T>;
    std::lock_guard<std::mutex> lk(m);
    return (*map)[guest];  // references into unordered_map stay valid across rehashing
}
}  // namespace

HLE_FUNC(_ZN2nn6crypto6detail7Md5Impl10InitializeEv, void, crypto_Md5Init, (void* self)) { StateFor<Md5>(self).Init(); }
HLE_FUNC(_ZN2nn6crypto6detail7Md5Impl6UpdateEPKvm, void, crypto_Md5Update, (void* self, const void* p, size_t n)) {
    StateFor<Md5>(self).Update(static_cast<const uint8_t*>(p), n);
}
HLE_FUNC(_ZN2nn6crypto6detail7Md5Impl7GetHashEPvm, void, crypto_Md5GetHash, (void* self, void* out, size_t n)) {
    uint8_t d[16];
    StateFor<Md5>(self).Final(d);
    memcpy(out, d, n < 16 ? n : 16);
}
HLE_FUNC(_ZN2nn6crypto6detail8Sha1Impl10InitializeEv, void, crypto_Sha1Init, (void* self)) {
    StateFor<Sha1>(self).Init();
}
HLE_FUNC(_ZN2nn6crypto6detail8Sha1Impl6UpdateEPKvm, void, crypto_Sha1Update, (void* self, const void* p, size_t n)) {
    StateFor<Sha1>(self).Update(static_cast<const uint8_t*>(p), n);
}
HLE_FUNC(_ZN2nn6crypto6detail8Sha1Impl7GetHashEPvm, void, crypto_Sha1GetHash, (void* self, void* out, size_t n)) {
    uint8_t d[20];
    StateFor<Sha1>(self).Final(d);
    memcpy(out, d, n < 20 ? n : 20);
}
HLE_FUNC(_ZN2nn6crypto16GenerateSha1HashEPvmPKvm, void, crypto_GenerateSha1Hash,
         (void* out, size_t out_size, const void* data, size_t size)) {
    Sha1 s;
    s.Init();
    s.Update(static_cast<const uint8_t*>(data), size);
    uint8_t d[20];
    s.Final(d);
    memcpy(out, d, out_size < 20 ? out_size : 20);
}
