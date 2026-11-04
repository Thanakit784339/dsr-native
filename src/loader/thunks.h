#pragma once
// Generation of AArch64 import thunks.
//
// Guest (Switch) code treats x18 as a scratch register, while Android reserves x18 for the
// shadow call stack (SCS). Every guest -> host call therefore goes through a thunk that restores
// the host x18 for the current thread before branching to the host function:
//
//   +0  ldr  x16, L_restore      ; hle_restore_x18 (preserves all regs except x16, x18, x30)
//   +4  mov  x17, x30            ; keep guest return address
//   +8  blr  x16
//   +12 mov  x30, x17
//   +16 ldr  x16, L_data         ; per-import data (import index), visible to asm entry points
//   +20 ldr  x17, L_target
//   +24 br   x17                 ; tail-call: host returns straight to guest
//   +28 nop
//   +32 L_restore: .quad
//   +40 L_data:    .quad
//   +48 L_target:  .quad
//
// x16/x17 are the AAPCS64 intra-procedure-call scratch registers, so clobbering them at a call
// boundary is legal; argument registers x0-x8 and v0-v7 pass through untouched.

#include <cstddef>
#include <cstdint>

namespace dsr::loader {

constexpr size_t kThunkSize = 56;

void EmitImportThunk(uint8_t* dst, uint64_t restore_fn, uint64_t data, uint64_t target);

// Variant whose data/target live in a separate writable slot {uint64 data; uint64 target}, so the
// code page never has to become writable again after creation:
//   +16 ldr x17, L_slot ; +20 ldp x16, x17, [x17] ; +24 br x17 ; +40 L_slot: .quad
void EmitIndirectThunk(uint8_t* dst, uint64_t restore_fn, uint64_t slot_addr);

}  // namespace dsr::loader
