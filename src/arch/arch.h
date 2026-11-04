#pragma once
// Architecture glue between host (Android/bionic) code and guest (Switch) code.

#include <cstddef>
#include <cstdint>

extern "C" {
// Per-thread host x18 (shadow call stack pointer) captured on every host -> guest transition.
extern thread_local uint64_t dsr_host_x18;

// Assembly entry points (arm64). Not callable from C directly except dsr_guest_call.
void dsr_restore_host_x18();
void dsr_unimplemented_entry();
void dsr_unimplemented_named_entry();
uint64_t dsr_guest_call(const void* fn, ...);  // see GuestCall below for the real contract
int dsr_setjmp(void* buf);
void dsr_longjmp(void* buf, int val);

// Implemented in C++ (hle/unimplemented.cpp); called by dsr_unimplemented_entry.
uint64_t dsr_hle_unimplemented(uint64_t import_index);
uint64_t dsr_hle_unimplemented_named(const char* name);
}

namespace dsr::arch {

void* RestoreHostX18Entry();
void* UnimplementedEntry();
void* UnimplementedNamedEntry();
void FlushICache(void* start, size_t size);
bool CanExecuteGuest();  // false on non-arm64 hosts (dry-run builds)

// Calls a guest function. The assembly shim shifts x1..x7 into x0..x6 and leaves v0-v7 alone, so any
// signature whose arguments all fit in registers (<= 7 integer/pointer args, <= 8 FP args) works.
// Arguments passed on the stack are NOT supported. Host x18 is saved and restored around the call.
template <typename R = void, typename... Args>
R GuestCall(const void* fn, Args... args) {
    static_assert(sizeof...(Args) <= 7, "GuestCall supports at most 7 register arguments");
    using Shim = R (*)(const void*, Args...);
    return reinterpret_cast<Shim>(&dsr_guest_call)(fn, args...);
}

}  // namespace dsr::arch
