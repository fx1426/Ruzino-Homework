import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.BindlessVertexBuffer;
import Scene.Lights.LightData;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

RWStructuredBuffer<uint> random_seeds;
RWTexture2D<float4> output;
RaytracingAccelerationStructure SceneBVH;
StructuredBuffer<uint2> inPixelTarget;
StructuredBuffer<RayInfo> rays;
StructuredBuffer<LightData> lightBuffer;
ConstantBuffer<uint> lightCount;

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
    payload.seed = random_seeds[DispatchRaysIndex().x];

    float3 output_color = float3(0, 0, 0);

    const int spp = 1;

    for (int i = 0; i < spp; i++) {
        payload.color = float3(0, 0, 0);
        payload.throughput = float3(1, 1, 1);
        payload.depth = 0;
        payload.terminated = false;
        TraceRay(
            SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, get_ray_desc(ray), payload);
        output_color += payload.color;
    }
    output_color /= float(spp);  // Average color over multiple samples
    // output_color = ray.Direction*0.5+0.5;
    output[pixel] = float4(output_color, 1.0);

    random_seeds[DispatchRaysIndex().x] = payload.seed;
}

// Sample a random light and return its contribution
float3 sampleLights(float3 hitPos, float3 hitNormal, float3 V, inout uint seed)
{
    if (lightCount == 0)
        return float3(0, 0, 0);

    // Randomly select a light
    uint lightIndex =
        min(uint(random_float(seed) * lightCount), lightCount - 1);
    LightData light = lightBuffer[lightIndex];

    float3 lightContrib = float3(0, 0, 0);
    float3 lightDir;
    float lightDistance;
    float lightPdf =
        1.0 / float(lightCount);  // Uniform light selection probability

    // Sample the selected light based on its type
    if (light.type == uint(LightType::Point)) {
        // Point light
        lightDir = light.posW - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        // Check if light is visible
        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            // Trace shadow ray
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            TraceRay(
                SceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xFF,
                1,  // Shadow hit group index
                0,
                1,  // Shadow miss index
                shadowRay,
                shadowPayload);

            if (shadowPayload.isVisible) {
                // Compute lighting with inverse square falloff
                float attenuation = 1.0 / (lightDistance * lightDistance);
                lightContrib = light.intensity * attenuation * NdotL / lightPdf;
            }
        }
    }
    else if (light.type == uint(LightType::Sphere)) {
        // Sphere light - sample a random point on the sphere
        // For now, treat as point light at center
        // TODO: Proper sphere light sampling
        lightDir = light.posW - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            TraceRay(
                SceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            if (shadowPayload.isVisible) {
                float attenuation = 1.0 / (lightDistance * lightDistance);
                lightContrib = light.intensity * attenuation * NdotL / lightPdf;
            }
        }
    }
    else if (light.type == uint(LightType::Distant)) {
        // Distant/Directional light
        lightDir = -light.dirW;
        lightDistance = 1e10;  // Very large distance

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            TraceRay(
                SceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            if (shadowPayload.isVisible) {
                lightContrib = light.intensity * NdotL / lightPdf;
            }
        }
    }
    else if (light.type == uint(LightType::Rect)) {
        // Rectangle area light - sample a random point on the rectangle
        float u = random_float(seed);
        float v = random_float(seed);

        float3 localPoint = float3((u - 0.5), (v - 0.5), 0.0);

        // Transform to world space using tangent and bitangent
        float3 lightSamplePos = light.posW + localPoint.x * light.tangent +
                                localPoint.y * light.bitangent;

        lightDir = lightSamplePos - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        float LdotN = max(0.0, dot(-lightDir, light.dirW));  // Light's normal

        if (NdotL > 0.0 && LdotN > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            TraceRay(
                SceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            if (shadowPayload.isVisible) {
                // Geometric term: NdotL * LdotN / distance^2
                float geometricTerm =
                    (NdotL * LdotN) / (lightDistance * lightDistance);
                // PDF = 1 / area (uniform sampling on rectangle)
                float areaPdf = 1.0 / light.surfaceArea;
                lightContrib =
                    light.intensity * geometricTerm / (lightPdf * areaPdf);
            }
        }
    }
    else if (light.type == uint(LightType::Disc)) {
        // Disc area light - sample a random point on the disc
        float r = sqrt(random_float(seed));  // sqrt for uniform distribution
        float theta = 2.0 * 3.14159265359 * random_float(seed);

        // Sample point on disc in local space
        float2 localPoint = float2(r * cos(theta), r * sin(theta));

        // Transform to world space
        float3 lightSamplePos = light.posW + localPoint.x * light.tangent +
                                localPoint.y * light.bitangent;

        lightDir = lightSamplePos - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        float LdotN = max(0.0, dot(-lightDir, light.dirW));

        if (NdotL > 0.0 && LdotN > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = max(0.0, lightDistance - 0.001);

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            TraceRay(
                SceneBVH,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            if (shadowPayload.isVisible) {
                float geometricTerm =
                    (NdotL * LdotN) / (lightDistance * lightDistance);
                float areaPdf = 1.0 / light.surfaceArea;
                lightContrib =
                    light.intensity * geometricTerm / (lightPdf * areaPdf);
            }
        }
    }

    return lightContrib;
}

[shader("closesthit")]
void ClosestHit(
    inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    // Check if maximum depth is reached
    if (payload.depth >= 30) {
        payload.throughput = float3(1, 0, 0);
        payload.color = float3(0, 0, 0);
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
        vertex.tangentW = -vertex.tangentW;
        eta_flipped = 1;
    }

    // Create shading frame
    bool valid;
    ShadingFrame sf =
        ShadingFrame.createSafe(vertex.normalW, vertex.tangentW, valid);

    // vertex.tangentW.xyz = sf.T;

    // coordinates return;

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

    if (pdf == 0.0) {
        payload.seed = callData.seed;
        payload.terminated = true;
        return;
    }

    // float3 visualize = float3(vertex.texC, 0.0);
    // visualize = dot(vertex.normalW, WorldRayDirection());
    // visualize = vertex.normalW.xyz * 0.5 + 0.5;  // Visualize tangent space
    // visualize = vertex.tangentW.xyz * 0.5 + 0.5;

    // float3 normal = normalize(vertex.normalW);
    // float3 tangent = normalize(vertex.tangentW.xyz);
    // // visualize = sf.B * 0.5 + 0.5;
    // // visualize = sf.N * 0.5 + 0.5;
    // // visualize = sf.T * 0.5 + 0.5;

    // // visualize = callData.sampledDir * 0.5 + 0.5;  // For debugging, visualize
    // if (eta_flipped) {
    //     visualize = float3(1, 0, 0);
    // }
    // payload.color = visualize;  // For debugging, visualize texture
    // return;

    float3 ray_dir = callData.sampledDir;
    float3 local_ray_dir =
        sf.toLocal(ray_dir);  // Check if this is transmission or reflection
    bool isTransmission = local_ray_dir.z < 0.0;  // if (isTransmission) {
    //     // payload.color = float3(1,0,0);  // For debugging, visualize
    //     transmission
    //     // payload.seed = callData.seed;
    //     // payload.terminated = true;
    //     // return;
    //     payload.throughput = 1.0f;
    // }    {
    float cosine_factor = abs(local_ray_dir.z);

    // For transmission, don't apply cosine factor to avoid double-counting
    // if (isTransmission) {
    //     payload.throughput *= callData.throughput / pdf;

    //     payload.throughput *= 1.0f;
    //     if (any(isnan(callData.throughput)) ||
    //         any(isinf(callData.throughput)) || pdf <= 0.0) {
    //         payload.terminated = true;
    //         return;
    //     }
    //     // payload.terminated = true;
    // }
    // else {
    // }

    // Validate throughput before continuing
    if (any(isnan(callData.throughput))) {
        payload.color = float3(1, 0, 0);
        payload.terminated = true;
        return;
    }

    if (any(isinf(callData.throughput))) {
        payload.color = float3(0, 0, 1);
        payload.terminated = true;
        return;
    }

    payload.throughput *= cosine_factor * callData.throughput / pdf;
    // if (payload.depth == 0) {
    //     // payload.color =  callData.pdf / 1000 / 5;
    //     payload.color = cosine_factor * callData.throughput / callData.pdf /
    //     2; payload.color = callData.throughput / callData.pdf / 2;
    //     // payload.color = callData.throughput / callData.pdf / 2;
    //     // payload.color = callData.throughput / callData.pdf/ 2;
    //     // payload.color = callData.throughput;//  / callData.pdf / 2;
    //     if (!isTransmission)
    //         payload.color = 0;
    //     // payload.color = callData.sampledDir * 0.5 + 0.5;  // For
    //     debugging, payload.seed = callData.seed; return;
    // }

    // payload.color = payload.throughput/2;
    // payload.seed = callData.seed;
    // return;

    payload.hitPosition = vertex.posW;
    payload.hitNormal = vertex.normalW;
    payload.depth++;
    payload.seed = callData.seed;

    // Add direct lighting contribution
    if (!payload.terminated && !isTransmission) {
        float3 directLight = sampleLights(
            vertex.posW, vertex.normalW, -WorldRayDirection(), payload.seed);
        // Check for NaN/Inf before accumulating
        if (!any(isnan(directLight)) && !any(isinf(directLight)) &&
            !any(isnan(callData.throughput)) &&
            !any(isinf(callData.throughput))) {
            payload.color +=
                payload.throughput * directLight * callData.throughput;
        }
        else {
            payload.color = float3(1, 0, 1);  // Magenta for debugging NaN/Inf
            payload.terminated = true;
        }
    }

    // Trace next ray (light comes from environment/miss shader)
    if (!payload.terminated) {
        RayDesc nextRay;

        // Better ray offset handling for transmission vs reflection
        // Use adaptive offset based on distance from origin - increased for AMD
        // compatibility
        float offset = max(0.01, length(vertex.posW) * 1e-4);

        // For transmission, offset against the normal; for reflection, offset
        // with the normal
        float3 offset_normal =
            isTransmission ? -vertex.normalW : vertex.normalW;

        // Additional offset along ray direction to avoid grazing angle issues
        float3 ray_offset =
            offset_normal * offset + ray_dir * max(1e-3, offset * 0.1);

        nextRay.Origin = vertex.posW + ray_offset;
        nextRay.Direction = ray_dir;
        nextRay.TMin = 0.0;
        nextRay.TMax = 1000.0;

        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, nextRay, payload);
    }

    // // Russian roulette termination
    // if (payload.depth >= 3) {
    //     float p =
    //         max(payload.throughput.x,
    //             max(payload.throughput.y, payload.throughput.z));
    //     if (random_float(seed) > p) {
    //         payload.terminated = true;
    //     }
    //     else {
    //         payload.throughput /= p;
    //     }
    // }
}
[shader("miss")]
void Miss(inout RayPayload payload)
{
    float3 rayDir = WorldRayDirection();
    float3 envColor = float3(0, 0, 0);

    // Check if we have a dome light
    bool hasDomeLight = false;
    LightData domeLight;

    for (uint i = 0; i < lightCount; i++) {
        LightData light = lightBuffer[i];
        if (light.type == uint(LightType::Dome)) {
            hasDomeLight = true;
            domeLight = light;
            break;
        }
    }

    if (hasDomeLight) {
        // Apply dome light transformation to ray direction
        // Transform ray direction from world space to dome light local space
        float3 localDir = mul(float4(rayDir, 0.0), domeLight.transMat).xyz;
        localDir = normalize(localDir);

        if (domeLight.textureIndex >= 0) {
            // Sample environment texture using spherical coordinates
            // Convert direction to spherical coordinates (latitude-longitude
            // mapping)
            float phi = atan2(localDir.z, localDir.x);
            float theta = acos(clamp(localDir.y, -1.0, 1.0));

            // Convert to UV coordinates [0,1]
            float u = (phi + 3.14159265359) / (2.0 * 3.14159265359);
            float v = theta / 3.14159265359;

            // Sample from bindless texture array
            // Use trilinear filtering (sampler index 1 = clamp + linear)
            float4 texColor =
                t_BindlessTextures[domeLight.textureIndex].SampleLevel(
                    samplers[1], float2(u, v), 0);
            envColor = texColor.rgb * domeLight.intensity;
        }
        else {
            // No texture, use solid color intensity
            envColor = domeLight.intensity;
        }
    }
    else {
        // Fallback: Enhanced environment lighting with realistic sky model
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

        envColor = skyColor * 0.0;  // Fallback disabled by default
    }

    // Add contribution to path
    payload.color += payload.throughput * envColor;

    // Debug: Check for invalid values
    if (any(isnan(payload.color)) || any(isinf(payload.color))) {
        payload.color = float3(1, 0, 1);  // Magenta for debugging NaN/Inf
        payload.terminated = true;
    }

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
