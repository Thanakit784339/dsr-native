#pragma once
#include <string>
// Small hooks the platform layer provides to the runtime.

namespace dsr::platform {

// Creates a new file the user can reach (Android: Download/DSRNative/<name> via MediaStore) and returns
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

}  // namespace dsr::platform
