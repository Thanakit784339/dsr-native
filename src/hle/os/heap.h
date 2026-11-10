#pragma once

namespace dsr::hle::os {
// True if p points into the guest heap set up through nn::os::SetMemoryHeapSize. Memory there is
// owned by the game's allocator (main exports malloc/free), everything else by the host.
bool GuestHeapContains(const void* p);
// The guest heap's committed range (base 0 before nn::os::SetMemoryHeapSize).
void GuestHeapRange(uintptr_t* base, size_t* size);
}  // namespace dsr::hle::os
