#pragma once
// Thunks created at runtime (e.g. for function pointers handed to the guest by nvnBootstrapLoader).
// Same layout/semantics as import thunks: restore host x18, load `data` into x16, branch to `target`.

#include <cstdint>

namespace dsr::loader {

// Thread-safe. Thunks are never freed.
void* MakeDynamicThunk(const void* target, uint64_t data);

}  // namespace dsr::loader
