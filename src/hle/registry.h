#pragma once
// Registry of host implementations for guest imports, keyed by mangled symbol name.
//
// Usage (in any hle/*.cpp file compiled into the object library):
//
//   HLE_FUNC(_ZN2nn2os11SleepThreadENS_8TimeSpanE, void, nn_os_SleepThread, (int64_t ns)) {
//       ...
//   }
//   HLE_DATA(stdout, g_guest_stdout);
//
// The host function must have exactly the AAPCS64 signature of the guest-side declaration. Notes:
//  * nn::Result is a trivially copyable 32-bit class -> return uint32_t.
//  * nn::TimeSpan / nn::os::Tick are trivially copyable 64-bit classes -> pass int64_t.
//  * Small handle structs (nn::fs::FileHandle = {void*}) are passed in a register -> pass the field.
//  * `const T&` parameters are pointers.

#include <cstdint>
#include <string>
#include <vector>

namespace dsr::hle {

struct Entry {
    const char* mangled;
    void* ptr;
    bool is_data;
};

void Register(const char* mangled, void* ptr, bool is_data);
void* FindFunction(const std::string& mangled);
void* FindData(const std::string& mangled);
const std::vector<Entry>& AllEntries();

struct AutoRegister {
    AutoRegister(const char* mangled, void* ptr, bool is_data) { Register(mangled, ptr, is_data); }
};

}  // namespace dsr::hle

// Token pasting is done directly (no extra expansion level) so names that are also libc macros
// (stdout, errno, ...) are not expanded before pasting.

#define HLE_FUNC(mangled, ret, name, params)                                                         \
    static ret name params;                                                                          \
    static ::dsr::hle::AutoRegister hle_reg_##name(#mangled, reinterpret_cast<void*>(&name), \
                                                                false);                               \
    static ret name params

// Register an existing host function (e.g. a libc function with an identical ABI) under a guest name.
#define HLE_ALIAS(mangled, fn)                                                                        \
    static ::dsr::hle::AutoRegister hle_alias_##mangled(#mangled, reinterpret_cast<void*>(&fn), \
                                                                     false)

#define HLE_DATA(mangled, object)                                                                     \
    static ::dsr::hle::AutoRegister hle_data_##mangled(#mangled, reinterpret_cast<void*>(&object), \
                                                                    true)
