#include "hle/nvn/shader_dump.h"

#include <unistd.h>

#include <ctime>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "common/log.h"
#include "common/platform.h"

namespace dsr::hle::nvn {
namespace {
constexpr const char* kTag = "shader";
std::mutex g_mutex;
std::unordered_set<uint64_t> g_seen;
int g_fd = -2;  // -2: not opened yet, -1: unavailable
uint32_t g_count = 0;
size_t g_bytes = 0;

uint64_t Fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

bool WriteAll(int fd, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    while (n) {
        ssize_t w = write(fd, b, n);
        if (w <= 0) return false;
        b += w;
        n -= size_t(w);
    }
    return true;
}
}  // namespace

void DumpShader(uint64_t code_address, const void* code, size_t code_size, const void* control, size_t control_size) {
    if (!code || !code_size) return;
    uint64_t key = Fnv1a(code, code_size);
    if (control) key = Fnv1a(control, control_size, key);
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_seen.insert(key).second) return;
    if (g_fd == -2) {
        time_t now = time(nullptr);
        tm t;
        localtime_r(&now, &t);
        char name[64];
        strftime(name, sizeof(name), "dsr_shaders_%Y%m%d_%H%M%S.bin", &t);
        g_fd = dsr::platform::CreateUserFile(name, "application/octet-stream");
        if (g_fd >= 0)
            LOGI(kTag, "dumping shaders to Download/DSRNative/logs/%s", name);
        else
            LOGW(kTag, "could not create the shader dump file");
    }
    const uint32_t index = g_count++;
    if (index < 8 || index % 100 == 0)
        LOGI(kTag, "shader %u: %zu bytes of code, %zu bytes of control data", index, code_size, control_size);
    if (g_fd < 0) return;
    struct Header {
        char magic[4];
        uint32_t version, index, code_size, control_size, reserved;
        uint64_t address;
    } h{{'D', 'S', 'H', 'D'}, 1, index, uint32_t(code_size), uint32_t(control ? control_size : 0), 0, code_address};
    bool ok = WriteAll(g_fd, &h, sizeof(h)) && WriteAll(g_fd, code, code_size) &&
              (!control || WriteAll(g_fd, control, control_size));
    g_bytes += sizeof(h) + code_size + (control ? control_size : 0);
    if (!ok) {
        LOGW(kTag, "shader dump write failed; stopping the dump");
        close(g_fd);
        g_fd = -1;
    }
}

}  // namespace dsr::hle::nvn
