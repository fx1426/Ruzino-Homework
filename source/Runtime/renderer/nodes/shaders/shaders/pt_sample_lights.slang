import pt_constants;
__exported import Scene.Lights.LightData;
StructuredBuffer<LightData> lightBuffer;

#include "utils/random.slangh"

struct LightSample {
    float3 direction;  // Normalized direction to light
    float3 intensity;  // Light intensity
    float pdf;         // PDF of selecting this light and sampling point
    bool valid;        // Whether the sample is valid
};

struct ShadowPayload {
    float3 hitPos;        // Hit position for continuation rays
    float3 rayDir;        // Ray direction for continuation
    float rayTMax;        // Maximum ray distance
    uint recursionDepth;  // Current recursion depth for opacity testing
    uint seed;
    bool isVisible;  // Whether light is visible or occluded
};

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
                RAY_FLAG_NONE,
                0xFF,
                1,
                1,
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
        lightDistance = 1e4;

        float NdotL = max(0.0, dot(hitNormal, lightDir));
        if (NdotL > 0.0) {
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitNormal * 0.001;
            shadowRay.Direction = lightDir;
            shadowRay.TMin = 0.001;
            shadowRay.TMax = lightDistance;

            ShadowPayload shadowPayload;
            shadowPayload.isVisible = true;
            shadowPayload.rayTMax = lightDistance;
            shadowPayload.recursionDepth = 0;
            shadowPayload.seed = seed;
            TraceRay(
                SceneBVH,
                RAY_FLAG_NONE,
                0xFF,
                1,
                1,
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
                RAY_FLAG_NONE,
                0xFF,
                1,
                1,
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
                RAY_FLAG_NONE,
                0xFF,
                1,
                1,
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
