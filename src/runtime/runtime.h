#pragma once
// Process-wide state for the native runtime: loaded modules, configured paths, guest allocator hooks.

#include <cstdint>
#include <memory>
#include <string>

#include "loader/module.h"
#include "loader/nso.h"

namespace dsr {

struct RuntimeConfig {
    std::string game_dir;     // directory holding exefs/ and the romfs contents (dvdbnd*.bdt, ...)
    std::string save_dir;     // writable directory for save data
    std::string cache_dir;    // writable directory for caches (shaders etc.)
    bool dry_run = false;     // map + relocate + bind only, never execute guest code
    bool abort_on_unimplemented = false;
    bool docked = false;      // nn::oe::GetOperationMode: false = handheld (720p), true = console
    std::string language = "en-US";
};

struct Runtime {
    RuntimeConfig config;
    loader::NsoImage main_nso;
    loader::NsoImage sdk_nso;          // parsed (not mapped) to source data constants such as nn::util tables
    bool have_sdk = false;
    loader::Module main;

    // Guest allocator entry points exported by main (the game replaces malloc & co).
    uintptr_t guest_malloc = 0, guest_free = 0, guest_calloc = 0, guest_realloc = 0, guest_aligned_alloc = 0;
};

Runtime& GetRuntime();

// Loads main (and optionally sdk for data), binds imports. Returns false with *err on failure.
bool LoadGame(const RuntimeConfig& cfg, std::string* err);

// Runs the guest startup sequence (nninitStartup, static constructors, nnMain) on a dedicated
// thread sized like the Switch main thread. Blocks until nnMain returns.
int RunGame();

// Prints an import binding summary (implemented / unimplemented / data) to the log.
void LogImportReport(bool list_unimplemented);

// Guest heap helpers used by HLE code that must hand memory to the game (strdup, operator new, ...).
void* GuestMalloc(size_t size);
void GuestFree(void* p);
void* GuestAlignedAlloc(size_t align, size_t size);

}  // namespace dsr
