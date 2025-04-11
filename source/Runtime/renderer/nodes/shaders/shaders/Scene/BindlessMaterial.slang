#pragma once
#include "SceneDefines.slangh"
#include "utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "utils/Math/PackedFormats.h"
using uint = uint32_t;
#endif

struct MaterialDataBlob {
    uint data[128];
};

#ifndef HOST_CODE
StructuredBuffer<MaterialDataBlob> materialBuffer : register(t3);

[[vk::binding(1, 1)]]
Texture2D t_BindlessTextures[] : register(t0, space2);

SamplerState sampler : register(s2);

#endif