// Host tool: translates every shader in a dsr_shaders_*.bin dump (written by the app) to SPIR-V.
//   dsr_shader_tool <dump.bin> <out_dir> [index...]
// Writes <out_dir>/<index>_<vs|fs>.spv for each shader that translates and prints a summary of failures.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "hle/nvn/shader/translator.h"

using namespace dsr::hle::nvn::shader;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dump.bin> <out_dir> [index...]\n", argv[0]);
        return 2;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) return perror(argv[1]), 1;
    std::vector<uint8_t> d;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
    fclose(f);
    std::vector<int> only;
    for (int k = 3; k < argc; k++) only.push_back(atoi(argv[k]));

    struct Header {
        char magic[4];
        uint32_t version, index, code_size, control_size, reserved;
        uint64_t address;
    };
    size_t off = 0;
    int total = 0, ok = 0;
    std::map<std::string, int> failures;
    while (off + sizeof(Header) <= d.size()) {
        Header h;
        memcpy(&h, &d[off], sizeof(h));
        if (memcmp(h.magic, "DSHD", 4) != 0) break;
        const uint8_t* code = &d[off + sizeof(h)];
        off += sizeof(h) + h.code_size + h.control_size;
        if (!only.empty() && std::find(only.begin(), only.end(), int(h.index)) == only.end()) continue;
        total++;
        TranslateOptions opt;
        if (const char* a = getenv("DSR_ALPHA_FUNC")) opt.alpha_func = atoi(a);  // validate alpha-test variants
        TranslateResult r = Translate(code, h.code_size, opt);
        if (!r.ok) {
            std::string key = r.error.substr(0, r.error.find(" (at"));
            failures[key]++;
            if (!only.empty()) printf("shader %u: %s\n", h.index, r.error.c_str());
            continue;
        }
        ok++;
        if (getenv("DSR_LIST"))
            printf("index %u %s inputs 0x%x outputs 0x%x code %u @%llx\n", h.index,
                   r.info.stage == Stage::Vertex ? "vs" : "fs", r.info.input_mask, r.info.output_mask, h.code_size,
                   (unsigned long long)h.address);
        char path[512];
        snprintf(path, sizeof(path), "%s/%u_%s.spv", argv[2], h.index, r.info.stage == Stage::Vertex ? "vs" : "fs");
        FILE* o = fopen(path, "wb");
        if (!o) return perror(path), 1;
        fwrite(r.spirv.data(), 4, r.spirv.size(), o);
        fclose(o);
    }
    printf("%d/%d shaders translated\n", ok, total);
    for (auto& [k, v] : failures) printf("  %5d  %s\n", v, k.c_str());
    return 0;
}
