#pragma once
// Parsing of Switch NSO0 executables (header, LZ4 segments, MOD0, .dynamic, symbols, relocations).
// Pure data processing: no memory mapping or execution happens here, so it builds and is tested on any host.

#include <cstdint>
#include <string>
#include <vector>

namespace dsr::loader {

struct NsoSegment {
    uint32_t mem_offset = 0;
    uint32_t size = 0;
};

// Elf64 constants we need (kept local so the loader does not depend on <elf.h> layouts per platform).
enum : uint32_t {
    R_AARCH64_ABS64 = 257,
    R_AARCH64_GLOB_DAT = 1025,
    R_AARCH64_JUMP_SLOT = 1026,
    R_AARCH64_RELATIVE = 1027,
};
enum : uint8_t { STT_NOTYPE = 0, STT_OBJECT = 1, STT_FUNC = 2 };

struct Symbol {
    std::string name;
    uint8_t type = 0;
    uint8_t bind = 0;
    uint16_t shndx = 0;
    uint64_t value = 0;
    uint64_t size = 0;
    bool defined() const { return shndx != 0; }
};

struct Reloc {
    uint64_t offset;
    uint32_t type;
    uint32_t sym;
    int64_t addend;
};

struct NsoImage {
    std::string name;
    std::string build_id_hex;
    NsoSegment text, rodata, data;
    uint32_t bss_size = 0;
    std::vector<uint8_t> image;  // flat image: text|rodata|data (bss not included; image_size() covers it)

    // From MOD0 / .dynamic
    uint64_t mod0 = 0, dynamic = 0;
    uint64_t eh_frame_hdr_start = 0, eh_frame_hdr_end = 0;
    uint64_t init = 0, fini = 0;
    uint64_t init_array = 0, init_array_size = 0;
    uint64_t fini_array = 0, fini_array_size = 0;
    std::vector<std::string> needed;
    std::vector<Symbol> symbols;
    std::vector<Reloc> relocs;

    uint64_t image_size() const { return uint64_t(data.mem_offset) + data.size + bss_size; }
    uint64_t bss_start() const { return uint64_t(data.mem_offset) + data.size; }

    // Returns the address (image offset) of a defined symbol, or 0.
    uint64_t FindExport(const std::string& name) const;
};

// Decompress an LZ4 block. Returns false on malformed input.
bool Lz4DecompressBlock(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);

// Parse an NSO file held in memory. On error returns false and fills *err.
bool ParseNso(const std::vector<uint8_t>& file, const std::string& name, NsoImage* out, std::string* err,
              bool verify_hashes = true);

bool ReadFile(const std::string& path, std::vector<uint8_t>* out);

}  // namespace dsr::loader
