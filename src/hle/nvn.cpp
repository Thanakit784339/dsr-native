// NVN bootstrap. The game obtains every NVN entry point by name through nvnBootstrapLoader /
// nvnDeviceGetProcAddress and calls them through function pointers. Each pointer handed out is a
// dynamic thunk (restores host x18). Unimplemented entry points get a named logging stub, so the first
// runs on device produce an ordered trace of the NVN calls the renderer needs.
//
// The NVN -> Vulkan backend registers its implementations with RegisterNvnProc().

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "arch/arch.h"
#include "common/log.h"
#include "hle/nvn.h"
#include "hle/registry.h"
#include "loader/dynamic_thunks.h"

namespace dsr::hle::nvn {
namespace {
std::mutex g_mutex;
std::unordered_map<std::string, void*>& Impls() {
    static auto* m = new std::unordered_map<std::string, void*>;
    return *m;
}
std::vector<std::pair<std::string, void*>>& Prefixes() {
    static auto* v = new std::vector<std::pair<std::string, void*>>;
    return *v;
}
std::unordered_map<std::string, void*>& Handed() {  // name -> thunk given to the guest
    static auto* m = new std::unordered_map<std::string, void*>;
    return *m;
}
}  // namespace

void RegisterNvnProc(const char* name, void* host_fn) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto [it, inserted] = Impls().emplace(name, host_fn);
    // Two implementations of one entry point would make the winner depend on static-init order.
    if (!inserted && it->second != host_fn) LOGE("nvn", "NVN entry point %s registered twice", name);
    it->second = host_fn;
}

void RegisterNvnPrefix(const char* prefix, void* host_fn) {
    std::lock_guard<std::mutex> lk(g_mutex);
    Prefixes().emplace_back(prefix, host_fn);
}

void* FindNvnImpl(const char* name) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto impl = Impls().find(name);
    if (impl != Impls().end()) return impl->second;
    void* fn = nullptr;
    size_t best = 0;
    for (const auto& [prefix, f] : Prefixes())
        if (prefix.size() > best && strncmp(name, prefix.c_str(), prefix.size()) == 0) {
            best = prefix.size();
            fn = f;
        }
    return fn;
}

void* GetProc(const char* name) {
    if (!name) return nullptr;
    std::lock_guard<std::mutex> lk(g_mutex);
    auto& handed = Handed();
    auto it = handed.find(name);
    if (it != handed.end()) return it->second;
    void* thunk;
    auto impl = Impls().find(name);
    void* prefix_fn = nullptr;
    size_t best = 0;
    if (impl == Impls().end()) {
        for (const auto& [prefix, fn] : Prefixes())
            if (prefix.size() > best && strncmp(name, prefix.c_str(), prefix.size()) == 0) {
                best = prefix.size();
                prefix_fn = fn;
            }
    }
    if (impl != Impls().end()) {
        thunk = loader::MakeDynamicThunk(impl->second, 0);
    } else if (prefix_fn) {
        thunk = loader::MakeDynamicThunk(prefix_fn, 0);
    } else {
        // The name string must outlive the thunk; keep a private copy.
        const char* copy = strdup(name);
        thunk = loader::MakeDynamicThunk(arch::UnimplementedNamedEntry(), reinterpret_cast<uint64_t>(copy));
    }
    handed.emplace(name, thunk);
    return thunk;
}

}  // namespace dsr::hle::nvn

HLE_FUNC(nvnBootstrapLoader, void*, nvn_BootstrapLoader, (const char* name)) {
    LOGD("nvn", "nvnBootstrapLoader(%s)", name ? name : "(null)");
    return dsr::hle::nvn::GetProc(name);
}

// Registered as an NVN proc (not an import): returned by nvnBootstrapLoader("nvnDeviceGetProcAddress").
static void* nvn_DeviceGetProcAddress(const void* device, const char* name) {
    (void)device;
    return dsr::hle::nvn::GetProc(name);
}
static struct NvnBootstrapRegistrar {
    NvnBootstrapRegistrar() {
        dsr::hle::nvn::RegisterNvnProc("nvnDeviceGetProcAddress", reinterpret_cast<void*>(&nvn_DeviceGetProcAddress));
    }
} g_nvn_bootstrap_registrar;
