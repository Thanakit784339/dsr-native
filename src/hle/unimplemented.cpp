// Handlers for guest calls into imports / NVN procs without a host implementation.
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "arch/arch.h"
#include "common/log.h"
#include "runtime/runtime.h"

namespace {
std::mutex g_mutex;
std::unordered_map<std::string, uint64_t> g_counts;

uint64_t Report(const std::string& name) {
    uint64_t n;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        n = ++g_counts[name];
    }
    // Log the first call, then every 1000th, so hot unimplemented functions do not flood logcat.
    if (n == 1 || n % 1000 == 0) LOGW("unimpl", "%s (call #%llu)", name.c_str(), (unsigned long long)n);
    if (dsr::GetRuntime().config.abort_on_unimplemented) FATAL("unimpl", "aborting on unimplemented %s", name.c_str());
    return 0;
}
}  // namespace

extern "C" uint64_t dsr_hle_unimplemented(uint64_t import_index) {
    const auto* imp = dsr::GetRuntime().main.ImportByIndex(import_index);
    return Report(imp ? imp->name : "import#" + std::to_string(import_index));
}

// Named stubs are NVN entry points. Object creation calls (`...Initialize`, `...SetShaders`) return an
// NVNboolean; report success so the game keeps going, everything else returns 0.
extern "C" uint64_t dsr_hle_unimplemented_named(const char* name) {
    std::string n = name ? name : "(unnamed)";
    Report(n);
    const bool creates = n.find("Initialize") != std::string::npos ||
                         (n.size() >= 10 && n.compare(n.size() - 10, 10, "SetShaders") == 0);
    return creates ? 1 : 0;
}
