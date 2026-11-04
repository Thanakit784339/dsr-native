#pragma once

namespace dsr::hle::nvn {
// Register a host implementation of an NVN entry point (e.g. "nvnDeviceInitialize").
void RegisterNvnProc(const char* name, void* host_fn);
// Fallback for every entry point starting with `prefix` that has no exact registration.
void RegisterNvnPrefix(const char* prefix, void* host_fn);
// Host implementation registered for `name` (exact or prefix), without a thunk (tests).
void* FindNvnImpl(const char* name);
// Returns a guest-callable pointer for an NVN entry point (implemented or logging stub).
void* GetProc(const char* name);
}  // namespace dsr::hle::nvn
