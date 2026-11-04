#include "loader/dynamic_thunks.h"

#include <sys/mman.h>

#include <atomic>
#include <mutex>

#include "arch/arch.h"
#include "common/log.h"
#include "loader/module.h"
#include "loader/thunks.h"

namespace dsr::loader {

namespace {
struct Slot {
    std::atomic<uint64_t> data;
    std::atomic<uint64_t> target;
};

std::mutex g_mutex;
uint8_t* g_code = nullptr;  // immutable RX page(s), every thunk pre-emitted
Slot* g_slots = nullptr;    // RW table read by the thunks
size_t g_capacity = 0;
size_t g_used = 0;

void NewBlock() {
    const size_t code_size = HostPageSize() * 4;
    const size_t n = code_size / kThunkSize;
    void* code = mmap(nullptr, code_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto* slots = static_cast<Slot*>(calloc(n, sizeof(Slot)));
    if (code == MAP_FAILED || !slots) FATAL("loader", "dynamic thunk block allocation failed");
    const auto restore = reinterpret_cast<uint64_t>(arch::RestoreHostX18Entry());
    for (size_t i = 0; i < n; i++)
        EmitIndirectThunk(static_cast<uint8_t*>(code) + i * kThunkSize, restore, reinterpret_cast<uint64_t>(&slots[i]));
    if (mprotect(code, code_size, PROT_READ | PROT_EXEC) != 0) FATAL("loader", "dynamic thunk mprotect failed");
    arch::FlushICache(code, code_size);
    g_code = static_cast<uint8_t*>(code);
    g_slots = slots;
    g_capacity = n;
    g_used = 0;
}
}  // namespace

void* MakeDynamicThunk(const void* target, uint64_t data) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_used == g_capacity) NewBlock();
    size_t i = g_used++;
    g_slots[i].data.store(data, std::memory_order_relaxed);
    g_slots[i].target.store(reinterpret_cast<uint64_t>(target), std::memory_order_release);
    return g_code + i * kThunkSize;
}

}  // namespace dsr::loader
