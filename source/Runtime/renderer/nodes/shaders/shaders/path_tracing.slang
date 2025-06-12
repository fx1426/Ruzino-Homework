import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.BindlessVertexBuffer;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

RWStructuredBuffer<uint> random_seeds;
RWTexture2D<float4> output;
RaytracingAccelerationStructure SceneBVH;
StructuredBuffer<uint2> inPixelTarget;
StructuredBuffer<RayInfo> rays;

struct RayPayload {
    float3 color;        // Accumulated color
    float3 throughput;   // Path throughput
    float3 hitPosition;  // Hit position in world space
    float3 hitNormal;    // Hit normal in world space
    uint depth;          // Current ray bounce depth
    uint seed;           // Random seed for this ray
    bool terminated;     // Flag to indicate if path is terminated
};

struct ShadowPayload {
    bool isVisible;  // Whether light is visible or occluded
};

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = inPixelTarget[DispatchRaysIndex().x];
    RayInfo ray = rays[DispatchRaysIndex().x];

    RayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.throughput = float3(1, 1, 1);
    payload.depth = 0;
    payload.seed = random_seeds[DispatchRaysIndex().x];
    payload.terminated = false;

    TraceRay(
        SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, get_ray_desc(ray), payload);

    output[pixel] = float4(payload.color, 1.0);

    random_seeds[DispatchRaysIndex().x] = payload.seed;
}

[shader("closesthit")]
void ClosestHit(
    inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    // Check if maximum depth is reached
    if (payload.depth >= 30) {
        payload.terminated = true;
        return;
    }

    uint id = DispatchRaysIndex().x;
    var instance_id = InstanceID();
    var prim_id = PrimitiveIndex();

    // Get vertex information
    VertexInfo vertex = get_interpolated_vertex(
        instance_id,
        prim_id,
        float2(attribs.barycentrics.x, attribs.barycentrics.y));

    // Fix normal orientation
    uint eta_flipped = 0;
    if (dot(WorldRayDirection(), vertex.normalW) >= 0.0) {
        vertex.normalW = -vertex.normalW;
        eta_flipped = 1;
    }

    // Create shading frame
    bool valid;
    ShadingFrame sf =
        ShadingFrame.createSafe(vertex.normalW, float4(0, 0, 0, 1), valid);
    vertex.tangentW.xyz = sf.T;

    // Get material ID
    var material_id = instanceDescBuffer[instance_id].materialID;
    if (vertex.subsetMaterialID != -1) {
        material_id = vertex.subsetMaterialID;
    }

    // Sample next ray direction
    var seed = payload.seed;

    var material_header = materialHeaderBuffer[material_id];

    // Evaluate material
    CallableData callData;
    callData.vertexInfo = vertex;
    callData.L = float3(0, 0, 0);
    callData.V = -WorldRayDirection();
    callData.color = float4(0, 0, 0, 1);
    callData.seed = seed;
    callData.eta_flipped = eta_flipped;
    callData.materialBlobID = material_header.material_blob_id;
    CallShader(material_header.material_type_id, callData);

    float pdf = callData.pdf;

    float3 ray_dir = callData.sampledDir;
    float3 local_ray_dir = sf.toLocal(ray_dir);

    float cosine_factor = abs(local_ray_dir.z);
    {
        payload.throughput *= cosine_factor * callData.throughput / pdf;
    }

    payload.hitPosition = vertex.posW;
    payload.hitNormal = vertex.normalW;
    payload.depth++;
    payload.seed = callData.seed;

    // Trace next ray (light comes from environment/miss shader)
    if (!payload.terminated) {
        RayDesc nextRay;
        nextRay.Origin = vertex.posW + (dot(ray_dir, vertex.normalW) < 0.0
                                            ? -vertex.normalW
                                            : vertex.normalW) *
                                           0.001;
        nextRay.Direction = ray_dir;
        nextRay.TMin = 0.0;
        nextRay.TMax = 1000.0;

        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, nextRay, payload);
    }

    // Russian roulette termination
    if (payload.depth >= 3) {
        float p =
            max(payload.throughput.x,
                max(payload.throughput.y, payload.throughput.z));
        if (random_float(seed) > p) {
            payload.terminated = true;
        }
        else {
            payload.throughput /= p;
        }
    }
}
[shader("miss")]
void Miss(inout RayPayload payload)
{
    // Enhanced environment lighting with realistic sky model
    float3 rayDir = WorldRayDirection();

    // Sky gradient parameters
    float3 horizonColor = float3(0.8, 0.9, 1.0);
    float3 zenithColor = float3(0.2, 0.4, 0.8);
    float3 sunsetColor = float3(0.9, 0.5, 0.2);

    // Calculate sky color based on ray direction
    float skyBlend = pow(max(0.0, rayDir.z + 0.1), 0.8);
    float3 skyColor = lerp(horizonColor, zenithColor, skyBlend);

    // Add sun and sunset glow
    float3 sunDir = normalize(float3(0.5, 0.5, 0.5));
    float sunDot = max(0.0, dot(rayDir, sunDir));
    float sunIntensity = pow(sunDot, 128.0) * 0.1;
    float sunsetIntensity = pow(sunDot, 8.0) * (1.0 - rayDir.z * 0.5);

    skyColor += float3(1.0, 0.9, 0.7) * sunIntensity;
    skyColor += sunsetColor * sunsetIntensity * (1.0 - skyBlend) * 0.4;

    // Add contribution to path
    payload.color += payload.throughput * skyColor * 1.0;
    payload.terminated = true;
}

[shader("closesthit")]
void ShadowHit(
    inout ShadowPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.isVisible = false;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload)
{
    payload.isVisible = true;
}
