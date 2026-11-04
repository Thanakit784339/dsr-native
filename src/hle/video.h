#pragma once

namespace dsr::video {
// Platform layer: the surface the game renders into (ANativeWindow* on Android). May change when the
// app is paused/resumed; the NVN backend re-reads it when (re)creating the swapchain.
void SetNativeWindow(void* window);
void* GetNativeWindow();
}  // namespace dsr::video
