#include "utils/view_cb.h"
import Scene.SceneTypes;

// ---[ Constant Buffers ]---
ConstantBuffer<PlanarViewConstants> viewConstant;

import Scene.BindlessVertexBuffer;

// Vertex shader output structure
struct VSOutput {
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 worldPos : POSITION;

#if defined(ENABLE_TEXCOORD)
    float2 texcoord : TEXCOORD0;
#endif
};

void main(
    uint i_vertexID: SV_VertexID,
    uint i_instanceID: SV_InstanceID,
    out VSOutput o_output)
{
    var InstanceData = instanceDescBuffer[i_instanceID];
    var MeshData = meshDescBuffer[InstanceData.geometryID];

    // Get the model matrix for the current instance
    float4x4 modelMatrix = InstanceData.transform;

    StaticVertexData vertex = load_vertex(MeshData, i_vertexID);

    // Transform the vertex position to world space
    float4 worldPos = mul(float4(vertex.position, 1.0), modelMatrix);
    // Transform the vertex position to clip space
    o_output.position = mul(worldPos, viewConstant.matWorldToClip);

    o_output.worldPos = worldPos.xyz;
    // Flip Y coordinate in final position
    o_output.position.y *= -1;

    o_output.normal =
        normalize(mul(float4(vertex.normal, 0.0), modelMatrix).xyz);

#if defined(ENABLE_TEXCOORD)
    o_output.texcoord = vertex.texCrd;
#endif
}
