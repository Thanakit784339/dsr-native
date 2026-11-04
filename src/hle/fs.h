#pragma once
#include <string>

namespace dsr::hle::fs {
// Translates a guest path ("system:/movww/frpg_title.mp4") into a host path using the current mounts.
bool ResolveGuestPath(const char* guest, std::string* host);
}  // namespace dsr::hle::fs
