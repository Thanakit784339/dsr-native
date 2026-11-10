#pragma once
// Controller state shared between the platform layer (Android input events) and nn::hid.
#include <atomic>
#include <cstdint>

namespace dsr::input {

// nn::hid::NpadButton bit positions.
enum Button : uint64_t {
    kA = 1ull << 0, kB = 1ull << 1, kX = 1ull << 2, kY = 1ull << 3,
    kStickL = 1ull << 4, kStickR = 1ull << 5, kL = 1ull << 6, kR = 1ull << 7,
    kZL = 1ull << 8, kZR = 1ull << 9, kPlus = 1ull << 10, kMinus = 1ull << 11,
    kLeft = 1ull << 12, kUp = 1ull << 13, kRight = 1ull << 14, kDown = 1ull << 15,
};

struct PadState {
    uint64_t buttons = 0;
    int32_t lx = 0, ly = 0, rx = 0, ry = 0;  // -32767..32767, +y = up
};

// Platform side: update with the latest state (thread-safe).
void SetPad(const PadState& s);
// Touch controls (merged with the controller): buttons held and the on-screen stick (-1..1, +y = up).
void SetTouch(uint64_t buttons, float lx, float ly);
// Touch camera: a swipe of (dx, dy) density-independent pixels (+y = down the screen). Turned into right-stick
// deflection from the swipe speed: `speed` 1 = full deflection at kLookFullSpeed dp/s.
void AddLook(float dx, float dy);
void SetLookSpeed(float speed, bool invert_y);
// HLE side: snapshot plus a monotonically increasing sampling number.
PadState GetPad(int64_t* sampling_number);

}  // namespace dsr::input
