#include "common/platform.h"

#include <atomic>

namespace dsr::platform {
namespace {
std::atomic<CreateUserFileFn> g_create{nullptr};
}
void SetCreateUserFile(CreateUserFileFn fn) { g_create.store(fn); }
int CreateUserFile(const char* name, const char* mime_type) {
    CreateUserFileFn fn = g_create.load();
    return fn ? fn(name, mime_type) : -1;
}
}  // namespace dsr::platform
