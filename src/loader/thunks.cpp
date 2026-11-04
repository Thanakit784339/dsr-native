#include "loader/thunks.h"

#include <cstring>

namespace dsr::loader {

namespace {
constexpr uint32_t LdrLit(unsigned rt, int byte_offset) {
    return 0x58000000u | ((uint32_t(byte_offset / 4) & 0x7FFFF) << 5) | rt;
}
constexpr uint32_t kMovX17X30 = 0xAA1E03F1;  // orr x17, xzr, x30
constexpr uint32_t kMovX30X17 = 0xAA1103FE;  // orr x30, xzr, x17
constexpr uint32_t kBlrX16 = 0xD63F0200;
constexpr uint32_t kBrX17 = 0xD61F0220;
constexpr uint32_t kNop = 0xD503201F;
}  // namespace

void EmitImportThunk(uint8_t* dst, uint64_t restore_fn, uint64_t data, uint64_t target) {
    const uint32_t code[8] = {
        LdrLit(16, 32 - 0),   // ldr x16, L_restore
        kMovX17X30,
        kBlrX16,
        kMovX30X17,
        LdrLit(16, 40 - 16),  // ldr x16, L_data
        LdrLit(17, 48 - 20),  // ldr x17, L_target
        kBrX17,
        kNop,
    };
    memcpy(dst, code, sizeof(code));
    memcpy(dst + 32, &restore_fn, 8);
    memcpy(dst + 40, &data, 8);
    memcpy(dst + 48, &target, 8);
}

void EmitIndirectThunk(uint8_t* dst, uint64_t restore_fn, uint64_t slot_addr) {
    constexpr uint32_t kLdpX16X17AtX17 = 0xA9404630;  // ldp x16, x17, [x17]
    const uint32_t code[8] = {
        LdrLit(16, 32 - 0),   // ldr x16, L_restore
        kMovX17X30,
        kBlrX16,
        kMovX30X17,
        LdrLit(17, 40 - 16),  // ldr x17, L_slot
        kLdpX16X17AtX17,
        kBrX17,
        kNop,
    };
    const uint64_t zero = 0;
    memcpy(dst, code, sizeof(code));
    memcpy(dst + 32, &restore_fn, 8);
    memcpy(dst + 40, &slot_addr, 8);
    memcpy(dst + 48, &zero, 8);
}

}  // namespace dsr::loader
