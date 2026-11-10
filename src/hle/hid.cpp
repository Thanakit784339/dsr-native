// nn::hid — one virtual controller fed by the platform layer (dsr::input). It is reported both as
// player 1 (Pro Controller style) and as the handheld console, whichever the game polls.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "common/log.h"
#include <mutex>

#include "hle/input.h"
#include "hle/os/os.h"
#include "hle/registry.h"

namespace dsr::input {
namespace {
std::mutex g_mutex;
PadState g_state;
uint64_t g_touch_buttons = 0;
float g_touch_lx = 0, g_touch_ly = 0;
// Touch camera: swipe distance gathered since the last sample; the stick follows the swipe speed.
constexpr float kLookFullSpeed = 500.0f;  // dp per second for full right-stick deflection (speed 1)
// The game turns the stick into -1..1 by dividing, without clamping (main+0x81ac3c): faster swipes give values
// past full deflection, up to this many times, so the camera can turn faster than a real stick allows.
constexpr float kLookMaxDeflection = 6.0f;
float g_look_dx = 0, g_look_dy = 0;
int64_t g_look_last_sample = 0, g_look_last_move = 0;
float g_look_vx = 0, g_look_vy = 0;  // smoothed swipe speed, dp/s
float g_look_speed = 1.0f;
bool g_look_invert = false;
int32_t ToStick(float v) { return int32_t(std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f)); }
}  // namespace
void SetPad(const PadState& s) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_state = s;
}
void SetTouch(uint64_t buttons, float lx, float ly) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_touch_buttons = buttons;
    g_touch_lx = lx;
    g_touch_ly = ly;
}
void AddLook(float dx, float dy) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_look_dx += dx;
    g_look_dy += dy;
    g_look_last_move = dsr::hle::os::NowNs();
}
void SetLookSpeed(float speed, bool invert_y) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_look_speed = speed > 0.05f ? speed : 0.05f;
    g_look_invert = invert_y;
}
PadState GetPad(int64_t* sampling) {
    const int64_t now = dsr::hle::os::NowNs();
    if (sampling) *sampling = now / 4'000'000;  // ~250 Hz sampling counter
    std::lock_guard<std::mutex> lk(g_mutex);
    PadState s = g_state;
    s.buttons |= g_touch_buttons;
    if (g_touch_lx != 0 || g_touch_ly != 0) {
        s.lx = ToStick(g_touch_lx);
        s.ly = ToStick(g_touch_ly);
    }
    // Swipe speed over the time since the previous sample (at least 8 ms, so several reads in one frame see the
    // same value), smoothed; back to rest once the finger has not moved for 60 ms.
    const int64_t dt = now - g_look_last_sample;
    if (dt >= 8'000'000) {
        const float secs = float(std::min<int64_t>(dt, 100'000'000)) / 1e9f;
        const float vx = g_look_dx / secs, vy = g_look_dy / secs;
        g_look_dx = g_look_dy = 0;
        g_look_last_sample = now;
        if (now - g_look_last_move > 60'000'000) {
            g_look_vx = g_look_vy = 0;
        } else {
            g_look_vx = 0.5f * g_look_vx + 0.5f * vx;
            g_look_vy = 0.5f * g_look_vy + 0.5f * vy;
        }
    }
    if (g_look_vx != 0 || g_look_vy != 0) {
        const float k = g_look_speed / kLookFullSpeed;
        auto over = [](float v) {
            return int32_t(std::lround(std::max(-kLookMaxDeflection, std::min(kLookMaxDeflection, v)) * 32767.0f));
        };
        s.rx = over(g_look_vx * k);
        s.ry = over((g_look_invert ? g_look_vy : -g_look_vy) * k);  // finger up = look up
    }
    return s;
}
}  // namespace dsr::input

namespace {
constexpr uint32_t kOk = 0;
constexpr uint32_t kNpadNo1 = 0, kNpadHandheld = 0x20;
constexpr uint32_t kStyleFullKey = 1u << 0, kStyleHandheld = 1u << 1;

// NpadFullKeyState / NpadHandheldState / NpadJoyDualState share this layout (0x28 bytes).
struct NpadState {
    int64_t sampling_number;
    uint64_t buttons;
    int32_t lx, ly, rx, ry;
    uint32_t attributes;  // bit0 IsConnected
    uint32_t reserved;
};
static_assert(sizeof(NpadState) == 0x28, "NpadState layout");

void FillNpad(NpadState* out, uint32_t id) {
    memset(out, 0, sizeof(*out));
    if (id != kNpadNo1 && id != kNpadHandheld) return;
    dsr::input::PadState s = dsr::input::GetPad(&out->sampling_number);
    // Diagnostics: first poll, and button changes as the game sees them (first 40).
    static std::atomic<bool> first{true};
    static std::atomic<uint64_t> last{0};
    static std::atomic<int> changes{0};
    if (first.exchange(false)) LOGI("hid", "game polls controller (npad id 0x%x)", id);
    if (s.buttons != last.exchange(s.buttons) && changes.fetch_add(1) < 40)
        LOGI("hid", "game sees buttons 0x%llx (npad id 0x%x)", (unsigned long long)s.buttons, id);
    out->buttons = s.buttons;
    out->lx = s.lx;
    out->ly = s.ly;
    out->rx = s.rx;
    out->ry = s.ry;
    out->attributes = 1;
}

uint32_t g_npad_no1 = kNpadNo1;
uint32_t g_npad_handheld = kNpadHandheld;
}  // namespace

HLE_DATA(_ZN2nn3hid6NpadId3No1E, g_npad_no1);
HLE_DATA(_ZN2nn3hid6NpadId8HandheldE, g_npad_handheld);

HLE_FUNC(_ZN2nn3hid14InitializeNpadEv, void, hid_InitializeNpad, ()) {}
HLE_FUNC(_ZN2nn3hid18InitializeKeyboardEv, void, hid_InitializeKeyboard, ()) {}
HLE_FUNC(_ZN2nn3hid18InitializeDebugPadEv, void, hid_InitializeDebugPad, ()) {}
HLE_FUNC(_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE, void,
         hid_SetSupportedNpadStyleSet, (uint32_t)) {}
HLE_FUNC(_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm, void, hid_SetSupportedNpadIdType, (const uint32_t*, size_t)) {}
HLE_FUNC(_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE, void, hid_SetNpadJoyHoldType, (int)) {}
HLE_FUNC(_ZN2nn3hid15GetNpadStyleSetERKj, uint32_t, hid_GetNpadStyleSet, (const uint32_t* id)) {
    return *id == kNpadNo1 ? kStyleFullKey : *id == kNpadHandheld ? kStyleHandheld : 0;
}
HLE_FUNC(_ZN2nn3hid12GetNpadStateEPNS0_16NpadFullKeyStateERKj, void, hid_GetNpadFullKey,
         (NpadState * out, const uint32_t* id)) {
    FillNpad(out, *id);
}
HLE_FUNC(_ZN2nn3hid12GetNpadStateEPNS0_17NpadHandheldStateERKj, void, hid_GetNpadHandheld,
         (NpadState * out, const uint32_t* id)) {
    FillNpad(out, *id);
}
HLE_FUNC(_ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyDualStateERKj, void, hid_GetNpadJoyDual,
         (NpadState * out, const uint32_t* id)) {
    FillNpad(out, *id);
}
HLE_FUNC(_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE, void, hid_GetKeyboardState, (void* out)) {
    memset(out, 0, 0x30);
}
HLE_FUNC(_ZN2nn3hid16GetDebugPadStateEPNS0_13DebugPadStateE, void, hid_GetDebugPadState, (void* out)) {
    memset(out, 0, 0x20);
}

// ---- six-axis (unused by this game's controls; report a resting sensor) ----------------------------
HLE_FUNC(_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE,
         int, hid_GetSixAxisSensorHandles, (uint32_t * handles, int count, const uint32_t*, uint32_t)) {
    for (int i = 0; i < count; i++) handles[i] = 0;
    return count;
}
HLE_FUNC(_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE, void, hid_StartSixAxisSensor, (const void*)) {}
HLE_FUNC(_ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE, void,
         hid_GetSixAxisSensorState, (void* out, const void*)) {
    memset(out, 0, 0x60);
}
HLE_FUNC(_ZNK2nn3hid18SixAxisSensorState13GetQuaternionEPNS_4util4neon12Vector4fTypeE, void,
         hid_SixAxisGetQuaternion, (const void*, float* q)) {
    q[0] = q[1] = q[2] = 0.0f;
    q[3] = 1.0f;
}

// ---- vibration (accepted, not forwarded yet) --------------------------------------------------------
HLE_FUNC(_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE,
         int, hid_GetVibrationDeviceHandles, (uint32_t * handles, int count, const uint32_t*, uint32_t)) {
    for (int i = 0; i < count; i++) handles[i] = uint32_t(i);
    return count;
}
HLE_FUNC(_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE, void, hid_InitializeVibrationDevice,
         (const void*)) {}
HLE_FUNC(_ZN2nn3hid22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE, void,
         hid_GetVibrationDeviceInfo, (uint32_t * info, const void*)) {
    info[0] = 1;  // VibrationDeviceType_LinearResonantActuator
    info[1] = 0;
}
HLE_FUNC(_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE, void,
         hid_SendVibrationValue, (const void*, const void*)) {}
HLE_FUNC(_ZN2nn3hid23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE, void,
         hid_GetActualVibrationValue, (float* v, const void*)) {
    v[0] = 0.0f;
    v[1] = 160.0f;
    v[2] = 0.0f;
    v[3] = 320.0f;
}

// Controller applet: pretend the user confirmed one player.
HLE_FUNC(_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE, uint32_t,
         hid_ShowControllerSupport, (uint8_t * result, const void*)) {
    if (result) {
        result[0] = 1;                       // playerCount
        memcpy(result + 4, &kNpadNo1, 4);    // selectedId
    }
    return kOk;
}
