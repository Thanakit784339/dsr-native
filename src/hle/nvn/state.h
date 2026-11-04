#pragma once
// Host mirrors of NVN state objects (blend, depth, vertex layout, samplers, ...), texture/sampler ID
// pools and program code, filled by the NVN setter entry points and read when commands are recorded.

#include <memory>

#include "hle/nvn/backend.h"
#include "hle/nvn/nvn_types.h"

namespace dsr::hle::nvn {

BlendStateObj GetBlendState(const void* obj);
ColorStateObj GetColorState(const void* obj);
ChannelMaskStateObj GetChannelMaskState(const void* obj);
DepthStencilStateObj GetDepthStencilState(const void* obj);
TextureViewObj GetTextureView(const void* view);
PolygonStateObj GetPolygonState(const void* obj);
MultisampleStateObj GetMultisampleState(const void* obj);
VertexAttribObj GetVertexAttrib(const void* obj);
VertexStreamObj GetVertexStream(const void* obj);

// Registered in texture / sampler pools by ID (NVN texture handles carry both IDs).
TexRef TextureById(int id);
TexRef ImageById(int id);  // storage images (nvnTexturePoolRegisterImage)
SamplerObj SamplerById(int id);

void SetProgramCode(const void* program, std::shared_ptr<const ShaderCode> code);
std::shared_ptr<const ShaderCode> GetProgramCode(const void* program);

// Device-wide modes set by the game (nvnDeviceSetDepthMode / nvnDeviceSetWindowOriginMode).
struct DeviceModes {
    int depth_mode = 0;   // 0: NEAR_IS_MINUS_W (OpenGL -1..1), 1: NEAR_IS_ZERO
    int origin_mode = 0;  // 0: LOWER_LEFT, 1: UPPER_LEFT
};
DeviceModes GetDeviceModes();

}  // namespace dsr::hle::nvn
