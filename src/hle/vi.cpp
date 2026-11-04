// nn::vi — the game opens the default display, creates one layer and passes the layer's native window
// to NVN (nvnWindowBuilderSetNativeWindow). We hand out the host window object owned by the platform
// layer (an ANativeWindow* on Android, set via dsr::video::SetNativeWindow).
#include <atomic>

#include "common/log.h"
#include "hle/registry.h"
#include "hle/video.h"

namespace dsr::video {
namespace {
std::atomic<void*> g_window{nullptr};
}
void SetNativeWindow(void* w) { g_window.store(w); }
void* GetNativeWindow() { return g_window.load(); }
}  // namespace dsr::video

namespace {
constexpr uint32_t kOk = 0;
// Opaque handles: the game only passes them back to us.
char g_display_tag, g_layer_tag;
}  // namespace

HLE_FUNC(_ZN2nn2vi10InitializeEv, void, vi_Initialize, ()) {}
HLE_FUNC(_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE, uint32_t, vi_OpenDefaultDisplay, (void** out)) {
    *out = &g_display_tag;
    return kOk;
}
HLE_FUNC(_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE, uint32_t, vi_CreateLayer, (void** out, void*)) {
    *out = &g_layer_tag;
    return kOk;
}
HLE_FUNC(_ZN2nn2vi19SetLayerScalingModeEPNS0_5LayerENS0_11ScalingModeE, uint32_t, vi_SetLayerScalingMode,
         (void*, int)) {
    return kOk;
}
HLE_FUNC(_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE, uint32_t, vi_GetNativeWindow, (void** out, void*)) {
    *out = dsr::video::GetNativeWindow();
    LOGI("vi", "GetNativeWindow -> %p", *out);
    return kOk;
}
