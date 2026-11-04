#pragma once

namespace dsr::hle::os {
// True if p points into the guest heap set up through nn::os::SetMemoryHeapSize. Memory there is
// owned by the game's allocator (main exports malloc/free), everything else by the host.
bool GuestHeapContains(const void* p);
}  // namespace dsr::hle::os
