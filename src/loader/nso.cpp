#include "loader/nso.h"

#include <cstdio>
#include <cstring>

#include "loader/sha256.h"

namespace dsr::loader {

namespace {

template <typename T>
T Rd(const uint8_t* p) {
    T v;
    memcpy(&v, p, sizeof(T));
    return v;
}

constexpr int64_t DT_NULL = 0, DT_NEEDED = 1, DT_PLTRELSZ = 2, DT_HASH = 4, DT_STRTAB = 5, DT_SYMTAB = 6,
                  DT_RELA = 7, DT_RELASZ = 8, DT_STRSZ = 10, DT_INIT = 12, DT_FINI = 13, DT_JMPREL = 23,
                  DT_INIT_ARRAY = 25, DT_FINI_ARRAY = 26, DT_INIT_ARRAYSZ = 27, DT_FINI_ARRAYSZ = 28,
                  DT_GNU_HASH = 0x6ffffef5;

}  // namespace

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize(size_t(sz));
    bool ok = fread(out->data(), 1, out->size(), f) == out->size();
    fclose(f);
    return ok;
}

bool Lz4DecompressBlock(const uint8_t* src, size_t n, uint8_t* dst, size_t dst_size) {
    size_t i = 0, o = 0;
    while (i < n) {
        uint8_t tok = src[i++];
        size_t lit = tok >> 4;
        if (lit == 15) {
            uint8_t b;
            do {
                if (i >= n) return false;
                b = src[i++];
                lit += b;
            } while (b == 255);
        }
        if (i + lit > n || o + lit > dst_size) return false;
        memcpy(dst + o, src + i, lit);
        i += lit;
        o += lit;
        if (i >= n) break;  // last sequence has literals only
        if (i + 2 > n) return false;
        size_t off = size_t(src[i]) | (size_t(src[i + 1]) << 8);
        i += 2;
        if (off == 0 || off > o) return false;
        size_t ml = tok & 15;
        if (ml == 15) {
            uint8_t b;
            do {
                if (i >= n) return false;
                b = src[i++];
                ml += b;
            } while (b == 255);
        }
        ml += 4;
        if (o + ml > dst_size) return false;
        const uint8_t* from = dst + o - off;
        for (size_t k = 0; k < ml; k++) dst[o + k] = from[k];  // overlapping copy is intended
        o += ml;
    }
    return o == dst_size;
}

uint64_t NsoImage::FindExport(const std::string& n) const {
    for (const auto& s : symbols)
        if (s.defined() && s.name == n) return s.value;
    return 0;
}

bool ParseNso(const std::vector<uint8_t>& f, const std::string& name, NsoImage* out, std::string* err,
              bool verify_hashes) {
    auto fail = [&](const char* m) {
        *err = name + ": " + m;
        return false;
    };
    if (f.size() < 0x100 || memcmp(f.data(), "NSO0", 4) != 0) return fail("not an NSO0 file");
    const uint8_t* h = f.data();
    uint32_t flags = Rd<uint32_t>(h + 0xC);
    out->name = name;
    char hex[65];
    for (int k = 0; k < 32; k++) snprintf(hex + 2 * k, 3, "%02x", h[0x40 + k]);
    out->build_id_hex = hex;

    NsoSegment* segs[3] = {&out->text, &out->rodata, &out->data};
    uint32_t foff[3], csize[3];
    for (int k = 0; k < 3; k++) {
        foff[k] = Rd<uint32_t>(h + 0x10 + k * 0x10);
        segs[k]->mem_offset = Rd<uint32_t>(h + 0x14 + k * 0x10);
        segs[k]->size = Rd<uint32_t>(h + 0x18 + k * 0x10);
        csize[k] = Rd<uint32_t>(h + 0x60 + k * 4);
    }
    out->bss_size = Rd<uint32_t>(h + 0x3C);
    if (out->text.mem_offset != 0 || out->rodata.mem_offset < out->text.size ||
        out->data.mem_offset < out->rodata.mem_offset + out->rodata.size)
        return fail("unexpected segment layout");

    out->image.assign(uint64_t(out->data.mem_offset) + out->data.size, 0);
    for (int k = 0; k < 3; k++) {
        bool compressed = flags & (1u << k);
        size_t in_size = compressed ? csize[k] : segs[k]->size;
        if (uint64_t(foff[k]) + in_size > f.size()) return fail("segment beyond end of file");
        uint8_t* dst = out->image.data() + segs[k]->mem_offset;
        if (compressed) {
            if (!Lz4DecompressBlock(f.data() + foff[k], in_size, dst, segs[k]->size))
                return fail("LZ4 decompression failed");
        } else {
            memcpy(dst, f.data() + foff[k], in_size);
        }
        if (verify_hashes && (flags & (1u << (k + 3)))) {
            uint8_t digest[32];
            Sha256(dst, segs[k]->size, digest);
            if (memcmp(digest, h + 0xA0 + k * 0x20, 32) != 0) return fail("segment hash mismatch");
        }
    }

    // MOD0 → .dynamic
    const std::vector<uint8_t>& img = out->image;
    auto in_img = [&](uint64_t off, uint64_t len) { return off + len <= img.size(); };
    uint32_t mod0 = Rd<uint32_t>(img.data() + 4);
    if (!in_img(mod0, 0x1C) || memcmp(img.data() + mod0, "MOD0", 4) != 0) return fail("MOD0 not found");
    out->mod0 = mod0;
    out->dynamic = mod0 + int64_t(Rd<int32_t>(img.data() + mod0 + 4));
    out->eh_frame_hdr_start = mod0 + int64_t(Rd<int32_t>(img.data() + mod0 + 0x10));
    out->eh_frame_hdr_end = mod0 + int64_t(Rd<int32_t>(img.data() + mod0 + 0x14));

    uint64_t strtab = 0, symtab = 0, strsz = 0, hash = 0, gnu_hash = 0;
    uint64_t rela = 0, relasz = 0, jmprel = 0, pltrelsz = 0;
    std::vector<uint64_t> needed;
    for (uint64_t p = out->dynamic;; p += 16) {
        if (!in_img(p, 16)) return fail(".dynamic runs past image");
        int64_t tag = Rd<int64_t>(img.data() + p);
        uint64_t val = Rd<uint64_t>(img.data() + p + 8);
        if (tag == DT_NULL) break;
        switch (tag) {
            case DT_NEEDED: needed.push_back(val); break;
            case DT_STRTAB: strtab = val; break;
            case DT_SYMTAB: symtab = val; break;
            case DT_STRSZ: strsz = val; break;
            case DT_HASH: hash = val; break;
            case DT_GNU_HASH: gnu_hash = val; break;
            case DT_RELA: rela = val; break;
            case DT_RELASZ: relasz = val; break;
            case DT_JMPREL: jmprel = val; break;
            case DT_PLTRELSZ: pltrelsz = val; break;
            case DT_INIT: out->init = val; break;
            case DT_FINI: out->fini = val; break;
            case DT_INIT_ARRAY: out->init_array = val; break;
            case DT_INIT_ARRAYSZ: out->init_array_size = val; break;
            case DT_FINI_ARRAY: out->fini_array = val; break;
            case DT_FINI_ARRAYSZ: out->fini_array_size = val; break;
            default: break;
        }
    }
    if (!strtab || !symtab) return fail("missing DT_STRTAB/DT_SYMTAB");
    auto cstr = [&](uint64_t off) -> std::string {
        uint64_t s = strtab + off;
        if (off >= strsz || !in_img(s, 1)) return {};
        const char* p = reinterpret_cast<const char*>(img.data() + s);
        return std::string(p, strnlen(p, size_t(strsz - off)));
    };
    for (uint64_t v : needed) out->needed.push_back(cstr(v));

    uint64_t nsyms = 0;
    if (hash) {
        nsyms = Rd<uint32_t>(img.data() + hash + 4);
    } else if (gnu_hash) {
        (void)gnu_hash;
        nsyms = (strtab - symtab) / 24;  // symtab immediately precedes strtab in Nintendo's layout
    } else {
        nsyms = (strtab - symtab) / 24;
    }
    out->symbols.reserve(nsyms);
    for (uint64_t k = 0; k < nsyms; k++) {
        const uint8_t* s = img.data() + symtab + k * 24;
        if (!in_img(symtab + k * 24, 24)) return fail("symtab runs past image");
        Symbol sym;
        sym.name = cstr(Rd<uint32_t>(s));
        sym.type = s[4] & 0xF;
        sym.bind = s[4] >> 4;
        sym.shndx = Rd<uint16_t>(s + 6);
        sym.value = Rd<uint64_t>(s + 8);
        sym.size = Rd<uint64_t>(s + 16);
        out->symbols.push_back(std::move(sym));
    }

    auto read_relas = [&](uint64_t base, uint64_t size) -> bool {
        for (uint64_t p = base; p < base + size; p += 24) {
            if (!in_img(p, 24)) return false;
            uint64_t info = Rd<uint64_t>(img.data() + p + 8);
            Reloc r{Rd<uint64_t>(img.data() + p), uint32_t(info & 0xffffffff), uint32_t(info >> 32),
                    Rd<int64_t>(img.data() + p + 16)};
            if (r.sym >= out->symbols.size()) return false;
            out->relocs.push_back(r);
        }
        return true;
    };
    if (rela && !read_relas(rela, relasz)) return fail("bad DT_RELA table");
    if (jmprel && !read_relas(jmprel, pltrelsz)) return fail("bad DT_JMPREL table");
    return true;
}

}  // namespace dsr::loader
