// nn::hid — one virtual controller fed by the platform layer (dsr::input). It is reported both as
// player 1 (Pro Controller style) and as the handheld console, whichever the game polls.
#include <cstring>
#include <mutex>

#include "hle/input.h"
#include "hle/os/os.h"
#include "hle/registry.h"

namespace dsr::input {
namespace {
std::mutex g_mutex;
PadState g_state;
}  // namespace
void SetPad(const PadState& s) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_state = s;
}
PadState GetPad(int64_t* sampling) {
    if (sampling) *sampling = dsr::hle::os::NowNs() / 4'000'000;  // ~250 Hz sampling counter
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_state;
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
