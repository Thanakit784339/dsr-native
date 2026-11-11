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
    // User settings (settings.txt next to the game folder on Android).
    int target_fps = 60;          // presentation pace: 30 (the Switch rate) or 60
    bool async_pipelines = true;  // compile new GPU pipelines off the render thread (no hitch; the draw waits a frame)
    bool auto_capture = false;    // diagnostics: save a few frame captures automatically during play
    bool profile = true;          // diagnostics: sampling CPU profile of the busy threads, logged every 10 s
    float gamma = 1.0f;           // extra display gamma applied through the game's brightness ramp (> 1 lifts shadows)
    std::string present_mode = "mailbox";  // swapchain present mode: mailbox, fifo, fifo_relaxed, immediate (FIFO if unsupported)
    bool audio_smooth = false;  // release audio buffers at a smoothed pace (experiment; movies were not timed by it)
    bool fast_cores = true;  // keep the game's frame-critical threads off the efficiency cores
    bool record_thread = true;  // record Vulkan commands on a second thread (draw preparation stays on the render thread)
    bool pin_render_thread = false;  // keep the render thread on the fastest CPU core(s) (slower on 8 Gen 2: off)
    float exposure = 1.37f;       // scales the tone-mapping exposure (1.0 = as the game draws it)
    // Internal render scale for the game's own presentable frame (its NVN window texture), 0.5-1.0.
    // <1 renders fewer pixels and lets Present()'s existing blit-to-screen upscale the result, trading
    // sharpness for fill-rate/bandwidth -- the same idea as a console's "dynamic resolution" setting.
    float render_scale = 1.0f;
    // How many of a big sampled texture's own top mip levels to skip allocating (0 = off, full quality).
    // A mip chain's top level is ~75% of its total bytes, so even 1 cuts real upload bandwidth hard during
    // a streaming burst, at the cost of a softer look on affected textures for as long as they stay loaded.
    int texture_mip_skip = 1;
    // Caps anisotropic filtering regardless of what the game itself requests (it can ask for up to 16x).
    // Past ~4x the visual gain is marginal but the GPU sampling cost keeps climbing, especially on ground/
    // floor textures viewed at a grazing angle -- a cheap, safe trade for the same "texture quality for
    // performance" budget as texture_mip_skip above.
    float max_anisotropy = 4.0f;
    // Narrows layout-transition barriers to the destination stages that actually use the image, instead of
    // always blocking ALL_COMMANDS (see RawBarrier()'s StagesUsing()). On tile-based GPUs this lets a new
    // pass's vertex/tiler work overlap the previous pass's fragment work instead of needlessly waiting for
    // it. Independently validated on real Mali hardware (a fork of this project called it mali_barriers).
    // On by default: still spec-correct (the source side stays ALL_COMMANDS, so every writer is still
    // waited for) -- off falls back to the old maximally-conservative behavior if this ever causes trouble.
    bool mali_barriers = true;
    // Transcode BC textures to ASTC instead of the RGBA8 fallback when the GPU can't do BC natively (see
    // Image::sw_transcode_astc). Off by default: at ASTC_6x6, reproducibly SIGSEGV'd on-device inside the
    // Mali driver (same PC/LR every crash) during heavy texture-streaming bursts; switching to ASTC_4x4
    // (the far more common/tested block size) fixed the crash, but a frame capture then showed the real
    // remaining problem -- the main world scene's texture-sampling pass came out solid black (100%
    // (0,0,0,0)) for every one of 205 real draws, while a later non-texture-sampling pass in the same
    // frame rendered correctly, and text never rendered at all. Confirmed by direct A/B: astc_transcode=0
    // on the same save, same spot -> everything renders correctly. So this isn't just an encode-time bug,
    // it breaks texture sampling/binding somewhere for the whole pass once ASTC-format images are mixed
    // in. Kept togglable without a rebuild, not ripped out, since RGBA8 (this default) is unaffected and
    // proven stable for hours.
    bool astc_transcode = false;
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

// Patches engine functions of this game build (log/panic output, frame rate). Called by LoadGame.
void InstallGameHooks();
// The game's current per-frame time step in seconds (1/30 unless the 60 fps patches are active).
float GameFrameTime();
// The game's movie player shows one video frame per game frame (30 fps on the Switch): while a video
// plays, frames are paced at 30 instead of the configured rate (at 60 the video ran at double speed and
// stopped every few frames to wait for its sound).
void SetMoviePlaying(bool playing);
int FrameRateNow();  // the configured rate (30..60), or 30 while a movie plays

// Prints an import binding summary (implemented / unimplemented / data) to the log.
void LogImportReport(bool list_unimplemented);

// Guest heap helpers used by HLE code that must hand memory to the game (strdup, operator new, ...).
void* GuestMalloc(size_t size);
void GuestFree(void* p);
void* GuestAlignedAlloc(size_t align, size_t size);

}  // namespace dsr
