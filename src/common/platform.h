#pragma once
// Small hooks the platform layer provides to the runtime.

namespace dsr::platform {

// Creates a new file the user can reach (Android: Download/DSRNative/<name> via MediaStore) and returns
// a writable fd, or -1. Safe to call from any thread.
using CreateUserFileFn = int (*)(const char* name, const char* mime_type);
void SetCreateUserFile(CreateUserFileFn fn);
int CreateUserFile(const char* name, const char* mime_type);

}  // namespace dsr::platform
