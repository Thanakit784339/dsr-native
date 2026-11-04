#pragma once
#include <cstddef>
#include <cstdint>

namespace dsr::loader {
void Sha256(const uint8_t* data, size_t len, uint8_t out[32]);
}
