// AArch64 host <-> guest transition helpers. Assembled only for arm64 builds.
//
// Register contract recap:
//  * Guest code uses x18 as an ordinary caller-saved scratch register.
//  * Host (Android) code expects x18 to hold the per-thread shadow call stack pointer.
//  * dsr_host_x18 (thread_local, accessed via TLSDESC) holds the host x18 value that was live when the
//    current thread last entered guest code. Nested host->guest->host->guest chains save/restore it.

    .text

// x0 = &\sym for the current thread. Clobbers x1 and x30; TLSDESC resolvers preserve everything else.
.macro TLS_ADDR_X0 sym
    adrp    x0, :tlsdesc:\sym
    ldr     x1, [x0, #:tlsdesc_lo12:\sym]
    add     x0, x0, #:tlsdesc_lo12:\sym
    .tlsdesccall \sym
    blr     x1
    mrs     x1, tpidr_el0
    add     x0, x1, x0
.endm

// ---------------------------------------------------------------------------------------------
// void dsr_restore_host_x18()
// Called from import thunks. Preserves every register except x16 (scratch), x18 (output) and x30.
    .globl  dsr_restore_host_x18
    .type   dsr_restore_host_x18, %function
    .p2align 4
dsr_restore_host_x18:
    stp     x0, x1, [sp, #-32]!
    str     x30, [sp, #16]
    TLS_ADDR_X0 dsr_host_x18
    ldr     x18, [x0]
    ldr     x30, [sp, #16]
    ldp     x0, x1, [sp], #32
    ret
    .size   dsr_restore_host_x18, . - dsr_restore_host_x18

// ---------------------------------------------------------------------------------------------
// R dsr_guest_call(const void* fn, a0..a6)   — calls fn(a0..a6), v0-v7 passed through untouched.
    .globl  dsr_guest_call
    .type   dsr_guest_call, %function
    .p2align 4
dsr_guest_call:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x18, [sp, #32]         // [sp+40] = host x18 at entry
    mov     x19, x0                     // fn
    mov     x21, x1                     // a0 (x1 is clobbered by the TLS sequence)
    TLS_ADDR_X0 dsr_host_x18
    mov     x20, x0                     // &dsr_host_x18 (x20 is callee-saved in guest code too)
    ldr     x16, [x20]
    str     x16, [sp, #48]              // previous value, for nesting
    str     x18, [x20]
    mov     x0, x21
    mov     x1, x2
    mov     x2, x3
    mov     x3, x4
    mov     x4, x5
    mov     x5, x6
    mov     x6, x7
    blr     x19
    ldr     x18, [sp, #40]              // back to host x18
    ldr     x16, [sp, #48]
    str     x16, [x20]
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
    .size   dsr_guest_call, . - dsr_guest_call

// ---------------------------------------------------------------------------------------------
// Target of thunks for imports without a host implementation. x16 = import index.
    .globl  dsr_unimplemented_entry
    .type   dsr_unimplemented_entry, %function
    .p2align 4
dsr_unimplemented_entry:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x0, x16
    bl      dsr_hle_unimplemented
    movi    d0, #0
    ldp     x29, x30, [sp], #16
    ret
    .size   dsr_unimplemented_entry, . - dsr_unimplemented_entry

// Same, for dynamically created thunks: x16 = const char* function name.
    .globl  dsr_unimplemented_named_entry
    .type   dsr_unimplemented_named_entry, %function
    .p2align 4
dsr_unimplemented_named_entry:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x0, x16
    bl      dsr_hle_unimplemented_named
    movi    d0, #0
    ldp     x29, x30, [sp], #16
    ret
    .size   dsr_unimplemented_named_entry, . - dsr_unimplemented_named_entry

// ---------------------------------------------------------------------------------------------
// musl-compatible setjmp/longjmp (jmp_buf: x19-x30, pad, sp, d8-d15 = 22 words). Called by guest
// code through an import thunk, so x30/sp are the guest's.
    .globl  dsr_setjmp
    .type   dsr_setjmp, %function
    .p2align 4
dsr_setjmp:
    stp     x19, x20, [x0, #0]
    stp     x21, x22, [x0, #16]
    stp     x23, x24, [x0, #32]
    stp     x25, x26, [x0, #48]
    stp     x27, x28, [x0, #64]
    stp     x29, x30, [x0, #80]
    mov     x2, sp
    str     x2, [x0, #104]
    stp     d8, d9, [x0, #112]
    stp     d10, d11, [x0, #128]
    stp     d12, d13, [x0, #144]
    stp     d14, d15, [x0, #160]
    mov     x0, #0
    ret
    .size   dsr_setjmp, . - dsr_setjmp

    .globl  dsr_longjmp
    .type   dsr_longjmp, %function
    .p2align 4
dsr_longjmp:
    ldp     x19, x20, [x0, #0]
    ldp     x21, x22, [x0, #16]
    ldp     x23, x24, [x0, #32]
    ldp     x25, x26, [x0, #48]
    ldp     x27, x28, [x0, #64]
    ldp     x29, x30, [x0, #80]
    ldr     x2, [x0, #104]
    mov     sp, x2
    ldp     d8, d9, [x0, #112]
    ldp     d10, d11, [x0, #128]
    ldp     d12, d13, [x0, #144]
    ldp     d14, d15, [x0, #160]
    cmp     w1, #0
    csinc   w0, w1, wzr, ne
    br      x30
    .size   dsr_longjmp, . - dsr_longjmp

    .section .note.GNU-stack, "", %progbits
