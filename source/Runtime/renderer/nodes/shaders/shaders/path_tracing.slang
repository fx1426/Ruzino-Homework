import callable_data;
#include "utils/random.slangh"
#include "utils/ray.slang"
import Scene.SceneTypes;
import Scene.BindlessMaterial;
import Scene.MaterialParams;
import Scene.MaterialParamsBuffer;
import Scene.BindlessVertexBuffer;
import Scene.Lights.LightData;
import Scene.StandardSurface;
import Scene.UsdPreviewSurface;
import utils.Math.ShadingFrame;
import Utils.Math.MathHelpers;

RWStructuredBuffer<uint> random_seeds;
RWTexture2D<float4> output;
RaytracingAccelerationStructure SceneBVH;
StructuredBuffer<uint2> inPixelTarget;
StructuredBuffer<RayInfo> rays;
StructuredBuffer<LightData> lightBuffer;

struct PathTracingConstants {
    uint lightCount;
    uint domeLightCallableIndex;  // Only valid when USE_DOME_LIGHT_CALLABLE
    uint materialFetchCallableBaseIndex;
    uint materialOpacityCallableOffset;  // Offset to add to fetch index for
                                         // opacity callables
};

ConstantBuffer<PathTracingConstants> ptConstants;

struct RayPayload {
    Spectrum color;       // Accumulated color
    Spectrum throughput;  // Path throughput
    float3 hitPosition;   // Hit position in world space
    uint depth;           // Current ray bounce depth
    uint seed;            // Random seed for this ray
    bool terminated;      // Flag to indicate if path is terminated
};

struct ShadowPayload {
    float3 hitPos;        // Hit position for continuation rays
    float3 rayDir;        // Ray direction for continuation
    float rayTMax;        // Maximum ray distance
    uint recursionDepth;  // Current recursion depth for opacity testing
    uint seed;
    bool isVisible;  // Whether light is visible or occluded
};

struct LightSample {
    float3 direction;  // Normalized direction to light
    float3 intensity;  // Light intensity
    float pdf;         // PDF of selecting this light and sampling point
    bool valid;        // Whether the sample is valid
};

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = inPixelTarget[DispatchRaysIndex().x];
    RayInfo ray = rays[DispatchRaysIndex().x];

    RayPayload payload;
    payload.seed = random_seeds[DispatchRaysIndex().x];

    Spectrum output_color = Spectrum(0.0);

    const int spp = 1;

    ray.TMax = 50000.0;

    for (int i = 0; i < spp; i++) {
        payload.color = Spectrum(0.0);
        payload.throughput = Spectrum(1.0);
        payload.depth = 0;
        payload.terminated = false;
        TraceRay(
            SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, get_ray_desc(ray), payload);
        output_color = add(output_color, payload.color);
    }
    output_color =
        div(output_color, float(spp));  // Average color over multiple samples
    // output_color = ray.Direction*0.5+0.5;
    output[pixel] = float4(output_color.toFloat3(), 1.0);

    random_seeds[DispatchRaysIndex().x] = payload.seed;
}

// Sample a random light and return its direction, intensity, and PDF
LightSample sampleLights(float3 hitPos, float3 hitNormal, inout uint seed)
{
    LightSample sample;
    sample.valid = false;

    if (ptConstants.lightCount == 0)
        return sample;

    // Count non-dome lights (dome lights are handled by miss shader)
    uint sampleableLightCount = 0;
    for (uint i = 0; i < ptConstants.lightCount; i++) {
        if (lightBuffer[i].type != uint(LightType::Dome)) {
            sampleableLightCount++;
        }
    }

    if (sampleableLightCount == 0)
        return sample;

    // Randomly select a non-dome light
    uint targetLight = uint(random_float(seed) * sampleableLightCount);
    uint lightIndex = 0;
    uint nonDomeCount = 0;
    for (uint i = 0; i < ptConstants.lightCount; i++) {
        if (lightBuffer[i].type != uint(LightType::Dome)) {
            if (nonDomeCount == targetLight) {
                lightIndex = i;
                break;
            }
            nonDomeCount++;
        }
    }

    LightData light = lightBuffer[lightIndex];
    float lightPdf =
        1.0 / float(sampleableLightCount);  // PDF for light selection
    float3 lightDir;
    float lightDistance;
    float areaPdf = 1.0;

    // Sample the selected light based on its type
    if (light.type == uint(LightType::Point)) {
        lightDir = light.posW - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.0001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.0001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = lightDistance - 0.0001;
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;
            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);
            seed = shadowPayload.seed;
            if (shadowPayload.isVisible) {
                float attenuation = 1.0 / (lightDistance * lightDistance);
                sample.direction = lightDir;
                sample.intensity = light.intensity * attenuation;
                sample.pdf = lightPdf;
                sample.valid = true;
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
            shadowRay.Origin = hitPos + hitNormal * 0.0001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.0001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = lightDistance - 0.0001;
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;

            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            seed = shadowPayload.seed;

            if (shadowPayload.isVisible) {
                float attenuation = 1.0 / (lightDistance * lightDistance);
                sample.direction = lightDir;
                sample.intensity = light.intensity * attenuation;
                sample.pdf = lightPdf;
                sample.valid = true;
            }
        }
    }
    else if (light.type == uint(LightType::Distant)) {
        lightDir = -light.dirW;
        lightDistance = 1e5;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.0001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = lightDistance;
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;
            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);
            seed = shadowPayload.seed;

            if (shadowPayload.isVisible) {
                sample.direction = lightDir;
                sample.intensity = light.intensity;
                sample.pdf = lightPdf;
                sample.valid = true;
            }
        }
    }
    else if (light.type == uint(LightType::Rect)) {
        float u = random_float(seed);
        float v = random_float(seed);

        float3 localPoint = float3((u - 0.5), (v - 0.5), 0.0);
        float3 lightSamplePos = light.posW + localPoint.x * light.tangent +
                                localPoint.y * light.bitangent;

        lightDir = lightSamplePos - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        float LdotN = max(0.0, dot(-lightDir, light.dirW));

        if (LdotN > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.0001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = lightDistance - 0.0001;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = lightDistance - 0.0001;
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;
            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);
            seed = shadowPayload.seed;

            if (shadowPayload.isVisible) {
                float geometricTerm =
                    (NdotL * LdotN) / (lightDistance * lightDistance);
                areaPdf = 1.0 / light.surfaceArea;
                sample.direction = lightDir;
                sample.intensity = light.intensity * geometricTerm;
                sample.pdf = lightPdf * areaPdf;
                sample.valid = true;
            }
        }
    }
    else if (light.type == uint(LightType::Disc)) {
        float r = sqrt(random_float(seed));
        float theta = 2.0 * 3.14159265359 * random_float(seed);

        float2 localPoint = float2(r * cos(theta), r * sin(theta));
        float3 lightSamplePos = light.posW + localPoint.x * light.tangent +
                                localPoint.y * light.bitangent;

        lightDir = lightSamplePos - hitPos;
        lightDistance = length(lightDir);
        lightDir /= lightDistance;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        float LdotN = max(0.0, dot(-lightDir, light.dirW));

        if (NdotL > 0.0 && LdotN > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.0001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.0;
            shadowRay.TMax = max(0.0, lightDistance - 0.0001);

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = max(0.0, lightDistance - 0.0001);
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;
            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,
                0,
                1,
                shadowRay,
                shadowPayload);

            seed = shadowPayload.seed;

            if (shadowPayload.isVisible) {
                float geometricTerm =
                    (NdotL * LdotN) / (lightDistance * lightDistance);
                areaPdf = 1.0 / light.surfaceArea;
                sample.direction = lightDir;
                sample.intensity = light.intensity * geometricTerm;
                sample.pdf = lightPdf * areaPdf;
                sample.valid = true;
            }
        }
    }

    return sample;
}

[shader("closesthit")]
void ClosestHit(
    inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    // Check if maximum depth is reached
    if (payload.depth >= 16) {
        payload.throughput = Spectrum(float3(1, 0, 0));
        payload.color = Spectrum(0.0);
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
        vertex.faceNormalW = -vertex.faceNormalW;
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

    // Sample a light source for direct lighting
    LightSample lightSample =
        sampleLights(vertex.posW, vertex.faceNormalW, seed);

    // Step 1: Call material data fetch callable to populate parameters
    FetchCallableData fetchData;
    fetchData.vertexInfo = vertex;
    fetchData.materialBlobID = material_header.material_blob_id;
    fetchData.material_params_index =
        id;  // Use ray index as material params index
    fetchData.vertexInfo = vertex;

    // Call the per-material data fetch callable to fill parameters
    CallShader(
        ptConstants.materialFetchCallableBaseIndex +
            material_header.material_type_id,
        fetchData);

    // Step 2: Call shared evaluation callable based on shader type
    EvalCallableData evalData;
    evalData.color = Spectrum(0.0);
    evalData.throughput = Spectrum(1.0);
    evalData.L = lightSample.direction;
    evalData.V = -WorldRayDirection();
    evalData.sampledDir = float3(0, 0, 0);
    evalData.pdf = 0.0;
    evalData.seed = seed;
    evalData.eta_flipped = eta_flipped;
    evalData.material_params_index = id;
    evalData.vertexInfo = vertex;

    CallShader(fetchData.shader_type_id, evalData);

    float pdf = evalData.pdf;

    if (pdf == 0.0) {
        payload.seed = evalData.seed;
        payload.terminated = true;
        return;
    }

    // Validate throughput before continuing
    if (any(isnan(evalData.throughput.toFloat3()))) {
        payload.color = Spectrum(float3(0, 1, 0));
        payload.terminated = true;
        return;
    }

    if (any(isinf(evalData.throughput.toFloat3()))) {
        payload.color = Spectrum(float3(0, 0, 1));
        payload.terminated = true;
        return;
    }

    // payload.seed = evalData.seed;
    // payload.color = div(evalData.throughput, evalData.pdf);
    // payload.throughput = Spectrum(float3(0.0f));
    // payload.terminated = true;
    // return;

    // Add direct lighting contribution: light_intensity * BRDF * NdotL / PDF
    if (lightSample.valid) {
        Spectrum directContrib =
            div(mul(Spectrum(
                        lightSample.intensity *
                        max(0.0, dot(vertex.normalW, lightSample.direction))),
                    evalData.color),
                lightSample.pdf);
        payload.color =
            add(payload.color, mul(payload.throughput, directContrib));
    }

    float3 ray_dir = evalData.sampledDir;
    bool isTransmission = dot(ray_dir, vertex.faceNormalW) < 0.0;

    payload.throughput = mul(payload.throughput, div(evalData.throughput, pdf));
    // Russian Roulette path termination after depth 5
    if (payload.depth >= 3) {
        float3 thr = payload.throughput.toFloat3();
        float rrProbability = min(0.95, max(thr.x, max(thr.y, thr.z)));
        if (random_float(evalData.seed) > rrProbability) {
            payload.terminated = true;
            payload.seed = evalData.seed;
            return;
        }
        // Boost throughput to maintain unbiased result
        payload.throughput = div(payload.throughput, rrProbability);
    }

    payload.hitPosition = vertex.posW;
    payload.depth++;
    payload.seed = evalData.seed;

    // Trace next ray (light comes from environment/miss shader)
    if (!payload.terminated) {
        RayDesc nextRay;

        // For transmission, offset against the normal; for reflection, offset
        // with the normal
        float3 offset_normal =
            isTransmission ? -vertex.faceNormalW : vertex.faceNormalW;

        // Additional offset along ray direction to avoid grazing angle issues
        float3 ray_offset = offset_normal * 1e-3f;

        nextRay.Origin = vertex.posW + ray_offset;
        nextRay.Direction = ray_dir;
        nextRay.TMin = 0.0;
        nextRay.TMax = 10000.0;

        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, nextRay, payload);
    }
}
[shader("miss")]
void Miss(inout RayPayload payload)
{
    float3 rayDir = WorldRayDirection();
    Spectrum envColor = Spectrum(0.0);

    // Check if we have a dome light
    bool hasDomeLight = false;
    LightData domeLight;

    for (uint i = 0; i < ptConstants.lightCount; i++) {
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

#if USE_DOME_LIGHT_CALLABLE
        // Use custom dome light callable if available
        DomeLightCallableData domeData;
        domeData.rayDirection = rayDir;
        domeData.localDirection = localDir;
        domeData.envColor = Spectrum(0.0);
        domeData.intensity = domeLight.intensity;
        domeData.textureIndex = domeLight.textureIndex;
        domeData.seed = payload.seed;

        // Call the custom dome light shader at dynamic callable index
        CallShader(ptConstants.domeLightCallableIndex, domeData);

        envColor = domeData.envColor;
        payload.seed = domeData.seed;
#else
        // Default dome light behavior
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
            envColor = Spectrum(texColor.rgb * domeLight.intensity);
        }
        else {
            // No texture, use solid color intensity
            envColor = Spectrum(domeLight.intensity);
        }
#endif  // USE_DOME_LIGHT_CALLABLE
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

        envColor = Spectrum(skyColor * 0.0);  // Fallback disabled by default
    }

    // Add contribution to path
    payload.color = add(payload.color, mul(payload.throughput, envColor));

    // Debug: Check for invalid values
    if (any(isnan(payload.color.toFloat3()))) {
        payload.color = Spectrum(float3(1, 0, 0));  // Red for debugging NaN/Inf
        payload.terminated = true;
    }

    if (any(isnan(payload.color.toFloat3()))) {
        payload.color =
            Spectrum(float3(0, 1, 1));  // Magenta for debugging NaN/Inf
        payload.terminated = true;
    }

    payload.terminated = true;
}

[shader("closesthit")]
void ShadowHit(
    inout ShadowPayload payload,
    in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.isVisible = true;
    // payload.isVisible = false;
    // return;
    // Limit recursion depth to prevent infinite loops
    const uint MAX_SHADOW_OPACITY = 2;
    if (payload.recursionDepth >= MAX_SHADOW_OPACITY) {
        payload.isVisible = false;
        return;
    }

    uint instance_id = InstanceID();
    uint prim_id = PrimitiveIndex();

    // Get vertex information
    VertexInfo vertex =
        get_interpolated_vertex(instance_id, prim_id, attribs.barycentrics);

    // Get material ID
    var material_id = instanceDescBuffer[instance_id].materialID;
    if (vertex.subsetMaterialID != -1) {
        material_id = vertex.subsetMaterialID;
    }

    var material_header = materialHeaderBuffer[material_id];

    // Optimized opacity fetch: Call the opacity-specific callable
    // This avoids full material parameter buffer access
    FetchCallableData opacityFetchData;
    opacityFetchData.vertexInfo = vertex;
    opacityFetchData.materialBlobID = material_header.material_blob_id;
    opacityFetchData.material_params_index =
        0;  // Will be filled with opacity value
    opacityFetchData.shader_type_id = 0;

    // Call opacity fetch callable (uses fetch_shader_opacity internally)
    CallShader(
        ptConstants.materialFetchCallableBaseIndex +
            material_header.material_type_id +
            ptConstants.materialOpacityCallableOffset,
        opacityFetchData);

    // Extract opacity from material_params_index (reinterpreted as float)
    float opacity = asfloat(opacityFetchData.material_params_index);
    // Simple stochastic opacity test
    float opacity_threshold = random_float(payload.seed);

    var normal = vertex.faceNormalW;

    if (opacity < 0.25) {
        // Material is transparent, continue tracing shadow ray
        float3 rayOrigin = vertex.posW + normal * 0.001;
        float remainingDist = payload.rayTMax - RayTCurrent();

        if (remainingDist > 0.001) {
            RayDesc continuationRay;
            continuationRay.Origin = rayOrigin;
            continuationRay.Direction = WorldRayDirection();
            continuationRay.TMin = 0.001;
            continuationRay.TMax = remainingDist;

            ShadowPayload continuationPayload;
            continuationPayload.isVisible = true;
            continuationPayload.rayTMax = remainingDist;
            continuationPayload.recursionDepth = payload.recursionDepth + 1;
            continuationPayload.seed = payload.seed;

            TraceRay(
                SceneBVH,
                RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                0xFF,
                1,  // Shadow hit group index
                0,
                1,  // Shadow miss shader index
                continuationRay,
                continuationPayload);
            payload.seed = continuationPayload.seed;
            payload.isVisible = continuationPayload.isVisible;
            return;
        }
    }
    // If we reach here, the surface is opaque
    payload.isVisible = false;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload)
{
    payload.isVisible = true;
}

// ============================================================================
// Material Evaluation Callables
// ============================================================================
// Note: Callable implementations are now in separate files:
// - callables/eval_fallback.slang
// - callables/eval_standard_surface.slang
// - callables/eval_preview_surface.slang
// They are linked via add_source_code() in the C++ code.
