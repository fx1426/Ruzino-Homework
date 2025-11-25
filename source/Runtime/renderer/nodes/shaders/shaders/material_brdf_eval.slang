// Material BRDF Eval and PDF Visualization Shader
// Computes BRDF eval and PDF for all outgoing directions on hemisphere

import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.BindlessVertexBuffer;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

RWTexture2D<float4> eval_output;
RWTexture2D<float4> pdf_output;
RaytracingAccelerationStructure SceneBVH;

struct AnalysisConstants {
    float3 incident_direction;  // Wi in world space
    float _pad0;
    float2 uv_coord;
    uint material_id;
    uint resolution;
};

ConstantBuffer<AnalysisConstants> analysis_params;

// Convert 2D texture coordinate to 3D direction using equal-area projection
// This provides more uniform sampling than spherical coordinates
float3 texcoord_to_direction(float2 uv)
{
    // Equal-area projection (Lambert azimuthal)
    // Maps unit disk to hemisphere uniformly by area
    float2 p = uv * 2.0 - 1.0;  // Map to [-1, 1]
    
    float r = length(p);
    if (r > 1.0) {
        // Outside hemisphere, return zero direction
        return float3(0, 0, -1);
    }
    
    // Lambert azimuthal equal-area projection
    float z = 1.0 - r * r;
    float scale = sqrt(1.0 - z);
    
    return float3(p.x * scale, p.y * scale, z);
}

// Compute Jacobian for the equal-area projection
// This is needed to convert between direction probability and texture space probability
float compute_jacobian(float3 dir)
{
    float z = dir.z;
    if (z <= 0.0) return 0.0;
    
    // For Lambert azimuthal projection:
    // Jacobian = 1 / (2 * pi * cos(theta)) where cos(theta) = z
    // But since we're mapping to unit disk, need to account for that
    // Actually for equal-area, Jacobian is constant = 1/pi
    return 1.0 / 3.14159265359;
}

struct EvalPayload {
    float3 brdf_value;
    float pdf_value;
    bool valid;
};

[shader("raygeneration")]
void EvalPdfRayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint resolution = analysis_params.resolution;
    
    if (any(pixel >= resolution)) {
        return;
    }
    
    // Convert pixel to UV [0, 1]
    float2 uv = (float2(pixel) + 0.5) / float(resolution);
    
    // Convert UV to outgoing direction (local space, z-up hemisphere)
    float3 local_wo = texcoord_to_direction(uv);
    
    // Check if direction is valid (inside hemisphere)
    if (local_wo.z <= 0.0) {
        eval_output[pixel] = float4(0, 0, 0, 0);
        pdf_output[pixel] = float4(0, 0, 0, 0);
        return;
    }
    
    // Create shading frame from incident direction
    float3 wi = normalize(analysis_params.incident_direction);
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(
        float3(0, 0, 1),  // Normal (z-up)
        float4(1, 0, 0, 1),  // Tangent
        valid
    );
    
    if (!valid) {
        eval_output[pixel] = float4(1, 0, 0, 1);  // Red = error
        pdf_output[pixel] = float4(1, 0, 0, 1);
        return;
    }
    
    // Transform wi and wo to world space
    float3 world_wi = sf.fromLocal(wi);
    float3 world_wo = sf.fromLocal(local_wo);
    
    // Create a dummy vertex info for material evaluation
    VertexInfo vertex;
    vertex.posW = float3(0, 0, 0);
    vertex.normalW = float3(0, 0, 1);
    vertex.tangentW = float4(1, 0, 0, 1);
    vertex.texC = analysis_params.uv_coord;
    vertex.subsetMaterialID = -1;
    
    // Get material
    uint material_id = analysis_params.material_id;
    var material_header = materialHeaderBuffer[material_id];
    
    // Prepare callable data
    CallableData callData;
    callData.vertexInfo = vertex;
    callData.L = world_wi;  // Incident direction (toward light)
    callData.V = -world_wo; // View direction (toward camera)
    callData.color = float4(0, 0, 0, 1);
    callData.seed = 12345;  // Dummy seed
    callData.eta_flipped = 0;
    callData.materialBlobID = material_header.material_blob_id;
    
    // Call material evaluation
    // Note: This assumes material callable computes BRDF eval
    // You may need to modify the callable interface to return PDF separately
    CallShader(material_header.material_type_id, callData);
    
    // Extract BRDF value and PDF
    float3 brdf = callData.throughput;
    float pdf = callData.pdf;
    
    // Apply Jacobian correction
    float jacobian = compute_jacobian(local_wo);
    
    // Store results
    // RGB = BRDF value, A = cos(theta) for reference
    eval_output[pixel] = float4(brdf, local_wo.z);
    
    // RGB = PDF, A = PDF * Jacobian (for validation)
    pdf_output[pixel] = float4(pdf, pdf, pdf, pdf * jacobian);
}

[shader("closesthit")]
void EvalHit(inout EvalPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Not used for this shader
    payload.valid = false;
}

[shader("miss")]
void EvalMiss(inout EvalPayload payload)
{
    // Not used for this shader
    payload.valid = false;
}
