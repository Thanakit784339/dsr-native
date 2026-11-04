#include "common/platform.h"

#include <atomic>
#include <mutex>

namespace dsr::platform {
namespace {
std::atomic<CreateUserFileFn> g_create{nullptr};
}
void SetCreateUserFile(CreateUserFileFn fn) { g_create.store(fn); }
int CreateUserFile(const char* name, const char* mime_type) {
    CreateUserFileFn fn = g_create.load();
    return fn ? fn(name, mime_type) : -1;
}
namespace {
std::atomic<RequestTextFn> g_request_text{nullptr};
std::mutex g_text_mutex;
bool g_text_ready = false, g_text_ok = false;
std::u16string g_text;
}  // namespace
void SetRequestText(RequestTextFn fn) { g_request_text.store(fn); }
bool RequestText(const char* title, const std::u16string& initial, int max_length) {
    {
        std::lock_guard<std::mutex> lk(g_text_mutex);
        g_text_ready = false;
    }
    RequestTextFn fn = g_request_text.load();
    return fn && fn(title, initial, max_length);
}
void DeliverText(bool ok, const std::u16string& text) {
    std::lock_guard<std::mutex> lk(g_text_mutex);
    g_text_ready = true;
    g_text_ok = ok;
    g_text = text;
}
bool TakeText(bool* ok, std::u16string* text) {
    std::lock_guard<std::mutex> lk(g_text_mutex);
    if (!g_text_ready) return false;
    g_text_ready = false;
    *ok = g_text_ok;
    *text = g_text;
    return true;
}
}  // namespace dsr::platform
