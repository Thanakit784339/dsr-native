#pragma once
#include <cstddef>
#include <cstdint>

namespace dsr {

// Installs SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT/SIGTRAP handlers that write a crash report (signal,
// fault address, registers, and addresses translated to "main+0x..." offsets) to the log file and
// logcat, then hand the signal to the previous handler (Android's debuggerd).
void InstallCrashHandler();

// Register a guest module so crash addresses inside it are printed as module offsets
// (matching the addresses in Ghidra/IDA when the ELF from tools/mkelf.py is loaded at base 0).
void RegisterCrashModule(const char* name, uintptr_t base, size_t size);

}  // namespace dsr
