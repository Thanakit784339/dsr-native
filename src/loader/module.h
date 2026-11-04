#pragma once
// Maps a parsed NSO into this process, applies relocations and binds imports to host (HLE) code.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "loader/nso.h"

namespace dsr::loader {

// What an import was bound to (for diagnostics).
enum class BindKind : uint8_t { HleFunction, Unimplemented, HleData, DummyData, Absolute };

struct ImportBinding {
    uint32_t sym_index;
    std::string name;
    BindKind kind;
    uintptr_t address;   // what the GOT/vtable slot receives
    uint32_t ref_count;  // number of relocations that referenced it
};

// Supplies host implementations for imported symbols. Return 0/nullptr if not available.
struct ImportResolver {
    std::function<void*(const std::string& mangled)> find_function;
    std::function<void*(const std::string& mangled, uint64_t size_hint)> find_data;
    // Called (via assembly) when the guest calls an import that has no host implementation.
    // The thunk passes the index into Module::imports() so the handler can name it.
    void* unimplemented_entry = nullptr;
};

class Module {
public:
    Module() = default;
    ~Module();
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Reserve, copy, relocate, bind, protect. On failure returns false with *err set.
    bool Load(const NsoImage& nso, const ImportResolver& resolver, std::string* err);

    uintptr_t base() const { return base_; }
    size_t size() const { return size_; }
    const NsoImage& nso() const { return *nso_; }
    uintptr_t Addr(uint64_t image_offset) const { return base_ + image_offset; }
    bool Contains(uintptr_t a) const { return a >= base_ && a < base_ + size_; }
    const std::vector<ImportBinding>& imports() const { return imports_; }
    const ImportBinding* ImportByIndex(size_t i) const { return i < imports_.size() ? &imports_[i] : nullptr; }

    // Exported symbol lookup (image-relative value → absolute).
    uintptr_t Export(const std::string& name) const;

private:
    bool MapAndCopy(std::string* err);
    bool Relocate(const ImportResolver& resolver, std::string* err);
    bool Protect(std::string* err);

    const NsoImage* nso_ = nullptr;
    uintptr_t base_ = 0;
    size_t size_ = 0;       // reserved size (page aligned)
    size_t host_page_ = 4096;
    std::vector<ImportBinding> imports_;
    void* thunk_block_ = nullptr;
    size_t thunk_block_size_ = 0;
    std::vector<void*> dummy_data_;
};

// Host page size (4K or 16K on Android).
size_t HostPageSize();

}  // namespace dsr::loader
