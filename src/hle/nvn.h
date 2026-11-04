#pragma once

namespace dsr::hle::nvn {
// Register a host implementation of an NVN entry point (e.g. "nvnDeviceInitialize").
void RegisterNvnProc(const char* name, void* host_fn);
// Returns a guest-callable pointer for an NVN entry point (implemented or logging stub).
void* GetProc(const char* name);
}  // namespace dsr::hle::nvn
