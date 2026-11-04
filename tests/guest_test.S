// Fake "guest" code for the ABI tests: behaves like Switch code by trashing x18.
    .text

// uint64_t test_guest_call_import(void* thunk, uint64_t arg)
//   x18 = 0xdead; r = thunk(arg); x18 = 0xbeef; return r + 1
    .globl  test_guest_call_import
    .type   test_guest_call_import, %function
    .p2align 4
test_guest_call_import:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x18, #0xdead
    mov     x9, x0
    mov     x0, x1
    blr     x9
    add     x0, x0, #1
    mov     x18, #0xbeef
    ldp     x29, x30, [sp], #16
    ret
    .size   test_guest_call_import, . - test_guest_call_import

// uint64_t test_guest_sum7(a0..a6): checks GuestCall argument shifting
    .globl  test_guest_sum7
    .type   test_guest_sum7, %function
    .p2align 4
test_guest_sum7:
    mov     x18, #0x5555
    add     x0, x0, x1
    add     x0, x0, x2
    add     x0, x0, x3
    add     x0, x0, x4
    add     x0, x0, x5
    add     x0, x0, x6
    ret
    .size   test_guest_sum7, . - test_guest_sum7

// double test_guest_fp(double a, double b): FP args pass through GuestCall untouched
    .globl  test_guest_fp
    .type   test_guest_fp, %function
    .p2align 4
test_guest_fp:
    fmul    d0, d0, d1
    ret
    .size   test_guest_fp, . - test_guest_fp

    .section .note.GNU-stack, "", %progbits
