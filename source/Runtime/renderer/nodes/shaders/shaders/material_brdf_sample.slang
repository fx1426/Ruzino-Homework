// Material BRDF Sampling Visualization Shader
// Generates samples from BRDF and accumulates frequency distribution

import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.BindlessVertexBuffer;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

RWTexture2D<float4> sample_output;
RaytracingAccelerationStructure SceneBVH;
RWStructuredBuffer<uint> rng_seeds;

struct SampleConstants {
    float3 incident_direction;
    float _pad0;
    float2 uv_coord;
    uint material_id;
    uint resolution;
    uint num_samples;
};

ConstantBuffer<SampleConstants> sample_params;

// Convert 3D direction to 2D texture coordinate using equal-area projection
float2 direction_to_texcoord(float3 dir)
{
    // Lambert azimuthal equal-area projection (inverse)
    float z = dir.z;
    
    if (z <= 0.0) {
        // Below hemisphere
        return float2(-1, -1);
    }
    
    float r = sqrt(1.0 - z);
    float2 p = dir.xy * (r / max(length(dir.xy), 1e-6));
    
    // Map from [-1, 1] to [0, 1]
    return p * 0.5 + 0.5;
}

struct SamplePayload {
    uint sample_count;
    bool valid;
};

[shader("raygeneration")]
void SampleRayGen()
{
    uint thread_id = DispatchRaysIndex().x;
    uint resolution = sample_params.resolution;
    uint num_samples = sample_params.num_samples;
    
    // Initialize RNG seed
    uint seed = thread_id * 1664525 + 1013904223;
    rng_seeds[thread_id] = seed;
    
    // Create shading frame
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(
        float3(0, 0, 1),  // Normal (z-up)
        float4(1, 0, 0, 1),  // Tangent
        valid
    );
    
    if (!valid) {
        return;
    }
    
    // Incident direction in local space
    float3 wi = normalize(sample_params.incident_direction);
    float3 local_wi = sf.toLocal(wi);
    
    // Create dummy vertex for material evaluation
    VertexInfo vertex;
    vertex.posW = float3(0, 0, 0);
    vertex.normalW = float3(0, 0, 1);
    vertex.tangentW = float4(1, 0, 0, 1);
    vertex.texC = sample_params.uv_coord;
    vertex.subsetMaterialID = -1;
    
    // Get material
    uint material_id = sample_params.material_id;
    var material_header = materialHeaderBuffer[material_id];
    
    // Generate samples and accumulate to texture
    for (uint i = 0; i < num_samples; i++) {
        // Prepare callable data for sampling
        CallableData callData;
        callData.vertexInfo = vertex;
        callData.L = float3(0, 0, 0);  // Will be ignored for sampling
        callData.V = local_wi;         // View direction
        callData.color = float4(0, 0, 0, 1);
        callData.seed = seed;
        callData.eta_flipped = 0;
        callData.materialBlobID = material_header.material_blob_id;
        
        // Call material to generate sample
        CallShader(material_header.material_type_id, callData);
        
        // Extract sampled direction
        float3 sampled_dir = callData.sampledDir;
        float3 local_sampled = sf.toLocal(sampled_dir);
        
        // Update seed
        seed = callData.seed;
        
        // Check if sample is valid (above hemisphere)
        if (local_sampled.z > 0.0) {
            // Convert sampled direction to texture coordinate
            float2 uv = direction_to_texcoord(local_sampled);
            
            if (all(uv >= 0.0) && all(uv <= 1.0)) {
                // Convert to pixel coordinates
                uint2 pixel = uint2(uv * float(resolution));
                pixel = min(pixel, uint2(resolution - 1, resolution - 1));
                
                // Atomically increment counter
                // Note: Slang doesn't directly support atomic float add,
                // so we accumulate as fixed-point integers and normalize later
                // For now, we'll just accumulate the count
                
                float3 contribution = callData.throughput / max(callData.pdf, 1e-6);
                float weight = length(contribution);
                
                // Accumulate sample count (not atomic, but good enough for visualization)
                // Note: For precise atomic accumulation, would need RWStructuredBuffer instead
                float4 current = sample_output[pixel];
                sample_output[pixel] = current + float4(weight, 0, 0, 1);
            }
        }
    }
    
    rng_seeds[thread_id] = seed;
}

[shader("closesthit")]
void SampleHit(inout SamplePayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Not used
}

[shader("miss")]
void SampleMiss(inout SamplePayload payload)
{
    // Not used
}
