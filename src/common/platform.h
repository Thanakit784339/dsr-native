#pragma once
#include <string>
// Small hooks the platform layer provides to the runtime.

namespace dsr::platform {
// True if the core (dsrcore) was compiled with optimisation.
bool NativeCodeOptimized();

// Creates a new file the user can reach (Android: Download/DSRNative/logs/<name> via MediaStore) and returns
// a writable fd, or -1. Safe to call from any thread.
using CreateUserFileFn = int (*)(const char* name, const char* mime_type);
void SetCreateUserFile(CreateUserFileFn fn);
int CreateUserFile(const char* name, const char* mime_type);

// Asks the player for a line of text (the game's software keyboard). Asynchronous: the result is
// delivered with DeliverText and collected with TakeText. Returns false if no text input is available.
using RequestTextFn = bool (*)(const char* title, const std::u16string& initial, int max_length);
void SetRequestText(RequestTextFn fn);
bool RequestText(const char* title, const std::u16string& initial, int max_length);
void DeliverText(bool ok, const std::u16string& text);
// True once a result is available (then fills ok/text and clears it).
bool TakeText(bool* ok, std::u16string* text);

// Keeps the calling thread on the performance cores (off the slowest cluster, where the scheduler puts
// threads that sleep between bursts of work every frame). Once per thread; logs `who`. No-op elsewhere.
void PreferFastCores(const char* who);

}  // namespace dsr::platform
