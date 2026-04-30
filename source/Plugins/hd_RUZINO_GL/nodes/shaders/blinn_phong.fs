#version 430 core

struct Light {
    mat4 light_projection;
    mat4 light_view;
    vec3 position;
    float radius;
    vec3 color;
    int shadow_map_id;
};

layout(binding = 0) buffer lightsBuffer {
    Light lights[4];
};

uniform vec2 iResolution;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler;
uniform sampler2D metallicRoughnessSampler;
uniform sampler2DArray shadow_maps;
uniform sampler2D position;
uniform sampler2D aoSampler;

uniform vec3 camPos;
uniform int light_count;
uniform int shadowMode;
uniform bool useSSAO;
uniform float pcssLightSize;
uniform int pcssBlockerSamples;
uniform int pcssFilterSamples;
uniform int renderMode;
uniform int toonBands;
uniform float specularThreshold;
uniform float rimStrength;
uniform float rimPower;
uniform float outlineWidth;
uniform float normalEdgeThreshold;
uniform float depthEdgeThreshold;

layout(location = 0) out vec4 Color;

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 diskSample(int index, int count, float angle)
{
    float fi = float(index) + 0.5;
    float fc = max(float(count), 1.0);
    float radius = sqrt(fi / fc);
    float theta = fi * 2.39996323 + angle;
    return radius * vec2(cos(theta), sin(theta));
}

bool outsideShadowMap(vec2 uv)
{
    return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

float shadowDepthTexel(int lightId, ivec2 coord)
{
    ivec2 size = textureSize(shadow_maps, 0).xy;
    coord = clamp(coord, ivec2(0), size - ivec2(1));
    return texelFetch(
        shadow_maps, ivec3(coord, lights[lightId].shadow_map_id), 0).r;
}

float shadowDepthAt(int lightId, vec2 uv)
{
    ivec2 size = textureSize(shadow_maps, 0).xy;
    vec2 st = clamp(uv, vec2(0.0), vec2(1.0)) * vec2(size) - vec2(0.5);
    ivec2 base = ivec2(floor(st));
    vec2 f = fract(st);

    float d00 = shadowDepthTexel(lightId, base);
    float d10 = shadowDepthTexel(lightId, base + ivec2(1, 0));
    float d01 = shadowDepthTexel(lightId, base + ivec2(0, 1));
    float d11 = shadowDepthTexel(lightId, base + ivec2(1, 1));

    float dx0 = mix(d00, d10, f.x);
    float dx1 = mix(d01, d11, f.x);
    return mix(dx0, dx1, f.y);
}

vec2 receiverDepthGradient(vec3 projCoords)
{
    vec3 dx = dFdx(projCoords);
    vec3 dy = dFdy(projCoords);
    float det = dx.x * dy.y - dy.x * dx.y;

    if (abs(det) < 1e-6) {
        return vec2(0.0);
    }

    return vec2(
        (dx.z * dy.y - dy.z * dx.y) / det,
        (dy.z * dx.x - dx.z * dy.x) / det);
}

float receiverDepthAt(vec3 projCoords, vec2 depthGradient, vec2 uv)
{
    return projCoords.z + dot(depthGradient, uv - projCoords.xy);
}

float offsetBias(float baseBias, vec2 depthGradient, vec2 offset)
{
    float slopeBias = dot(abs(depthGradient), abs(offset)) * 0.6;
    return baseBias + min(slopeBias, 0.02);
}

float compareWidth(vec2 depthGradient, vec2 offset, float minWidth)
{
    float slopeWidth = dot(abs(depthGradient), abs(offset)) * 0.5;
    return min(max(minWidth, slopeWidth), 0.006);
}

float shadowCompare(
    int lightId,
    vec2 uv,
    float currentDepth,
    float bias,
    float width)
{
    if (outsideShadowMap(uv)) {
        return 0.0;
    }

    float closestDepth = shadowDepthAt(lightId, uv);
    if (closestDepth >= 0.9999) {
        return 0.0;
    }

    float depthDelta = currentDepth - closestDepth;
    return smoothstep(bias, bias + width, depthDelta);
}

float hardShadow(int lightId, vec3 projCoords, float bias)
{
    vec2 texelSize = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float width = max(max(texelSize.x, texelSize.y) * 0.25, 0.0006);
    return shadowCompare(
        lightId, projCoords.xy, projCoords.z, bias, width);
}

float averageBlockerDepth(
    int lightId,
    vec3 projCoords,
    vec2 depthGradient,
    float bias)
{
    vec2 texelSize = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float minRadius = max(texelSize.x, texelSize.y);
    float lightSize = clamp(pcssLightSize, minRadius, 0.08);
    float searchRadius = clamp(lightSize * 0.5, minRadius, 0.04);
    float angle =
        hash12(gl_FragCoord.xy + vec2(float(lightId) * 17.0, 3.1)) * 6.2831853;

    int sampleCount = clamp(pcssBlockerSamples, 1, 64);
    float blockerDepthSum = 0.0;
    int blockerCount = 0;

    for (int i = 0; i < 64; ++i) {
        if (i >= sampleCount) {
            break;
        }

        vec2 offset = diskSample(i, sampleCount, angle) * searchRadius;
        vec2 uv = projCoords.xy + offset;
        if (outsideShadowMap(uv)) {
            continue;
        }

        float sampleDepth = shadowDepthAt(lightId, uv);
        float receiverDepth = receiverDepthAt(projCoords, depthGradient, uv);
        float sampleBias = offsetBias(bias, depthGradient, offset);
        float width = compareWidth(depthGradient, offset, 0.001);

        if (sampleDepth < 0.9999 &&
            receiverDepth - sampleBias - width > sampleDepth) {
            blockerDepthSum += sampleDepth;
            blockerCount++;
        }
    }

    if (blockerCount == 0) {
        return -1.0;
    }
    return blockerDepthSum / float(blockerCount);
}

float filteredShadow(
    int lightId,
    vec3 projCoords,
    vec2 depthGradient,
    float bias,
    float filterRadius)
{
    float angle =
        hash12(gl_FragCoord.yx + vec2(float(lightId) * 5.7, 11.0)) * 6.2831853;
    int sampleCount = clamp(pcssFilterSamples, 1, 64);
    float shadow = 0.0;

    for (int i = 0; i < 64; ++i) {
        if (i >= sampleCount) {
            break;
        }

        vec2 offset = diskSample(i, sampleCount, angle) * filterRadius;
        vec2 uv = projCoords.xy + offset;
        float receiverDepth = receiverDepthAt(projCoords, depthGradient, uv);
        float sampleBias = offsetBias(bias, depthGradient, offset);
        float width = compareWidth(depthGradient, offset, 0.0012);

        shadow += shadowCompare(lightId, uv, receiverDepth, sampleBias, width);
    }

    return clamp(shadow / float(sampleCount), 0.0, 1.0);
}

float pcssShadow(int lightId, vec3 projCoords, vec2 depthGradient, float bias)
{
    float blockerDepth =
        averageBlockerDepth(lightId, projCoords, depthGradient, bias);
    if (blockerDepth < 0.0) {
        return 0.0;
    }

    vec2 texelSize = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float minRadius = max(texelSize.x, texelSize.y);
    float lightSize = clamp(pcssLightSize, minRadius, 0.08);
    float penumbra =
        clamp((projCoords.z - blockerDepth) / max(blockerDepth, 0.001), 0.0, 4.0);
    float filterRadius = clamp(lightSize * penumbra, minRadius, 0.08);

    return filteredShadow(
        lightId, projCoords, depthGradient, bias, filterRadius);
}

float computeShadow(int lightId, vec3 pos, vec3 N, vec3 L)
{
    float ndotl = max(dot(N, L), 0.0);
    float receiverOffset = mix(0.002, 0.008, 1.0 - ndotl);
    vec3 shadowPos = pos + L * receiverOffset;

    vec4 lightClipPos =
        lights[lightId].light_projection * lights[lightId].light_view *
        vec4(shadowPos, 1.0);

    if (lightClipPos.w <= 0.0) {
        return 0.0;
    }

    vec3 projCoords = lightClipPos.xyz / lightClipPos.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0) {
        return 0.0;
    }

    float bias = max(0.004 * (1.0 - ndotl), 0.001);

    if (shadowMode != 0) {
        vec2 depthGradient = receiverDepthGradient(projCoords);
        return pcssShadow(lightId, projCoords, depthGradient, bias);
    }
    return hardShadow(lightId, projCoords, bias);
}

float quantizeDiffuse(float diff)
{
    float bands = max(float(toonBands), 2.0);
    float level = floor(clamp(diff, 0.0, 1.0) * bands) / (bands - 1.0);
    return clamp(level, 0.0, 1.0);
}

float detectNprEdge(vec2 uv, vec3 pos, vec3 N)
{
    if (outlineWidth <= 0.0) {
        return 0.0;
    }

    vec2 pixelStep = vec2(outlineWidth) / iResolution;
    float centerDepth = max(length(camPos - pos), 1e-4);
    float maxNormalDelta = 0.0;
    float maxDepthDelta = 0.0;

    vec2 offsets[8] = vec2[](
        vec2(-1.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, -1.0),
        vec2(0.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0));

    for (int i = 0; i < 8; ++i) {
        vec2 sampleUv = uv + offsets[i] * pixelStep;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 ||
            sampleUv.y < 0.0 || sampleUv.y > 1.0) {
            continue;
        }

        vec3 neighborPos = texture(position, sampleUv).xyz;
        vec3 neighborNormal = normalize(texture(normalMapSampler, sampleUv).xyz);
        float neighborDepth = max(length(camPos - neighborPos), 1e-4);

        maxNormalDelta = max(
            maxNormalDelta,
            1.0 - clamp(dot(N, neighborNormal), -1.0, 1.0));
        maxDepthDelta = max(
            maxDepthDelta,
            abs(neighborDepth - centerDepth) / centerDepth);
    }

    float normalEdge = smoothstep(
        normalEdgeThreshold * 0.5, normalEdgeThreshold, maxNormalDelta);
    float depthEdge = smoothstep(
        depthEdgeThreshold * 0.5, depthEdgeThreshold, maxDepthDelta);
    return clamp(max(normalEdge, depthEdge), 0.0, 1.0);
}

void main()
{
    vec2 uv = gl_FragCoord.xy / iResolution;

    vec3 pos = texture(position, uv).xyz;
    vec3 normal = texture(normalMapSampler, uv).xyz;
    vec3 albedo = texture(diffuseColorSampler, uv).rgb;
    vec2 metalnessRoughness = texture(metallicRoughnessSampler, uv).xy;
    float roughness = clamp(metalnessRoughness.y, 0.0, 1.0);
    float shininess = mix(128.0, 8.0, roughness);
    float ambientStrength = 0.08;
    float specularStrength = 0.35;

    vec3 N = normalize(normal);
    vec3 V = normalize(camPos - pos);
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    float ao = useSSAO ? texture(aoSampler, uv).r : 1.0;
    vec3 finalColor = ambientStrength * albedo * clamp(ao, 0.0, 1.0);

    for (int i = 0; i < light_count; ++i) {
        vec3 L = normalize(lights[i].position - pos);
        vec3 H = normalize(L + V);
        vec3 lightColor = lights[i].color;

        float diff = max(dot(N, L), 0.0);
        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }

        float shadow = computeShadow(i, pos, N, L);

        if (renderMode == 0) {
            vec3 diffuse = diff * lightColor * albedo;
            vec3 specular = specularStrength * spec * lightColor;
            finalColor += (diffuse + specular) * (1.0 - shadow);
        }
        else {
            float diffuseBand = quantizeDiffuse(diff);
            float specularBand =
                diff > 0.0 && spec > specularThreshold ? 1.0 : 0.0;
            float rim = pow(
                max(1.0 - clamp(dot(N, V), 0.0, 1.0), 0.0),
                rimPower) * rimStrength;
            vec3 direct =
                (diffuseBand * albedo +
                 specularStrength * specularBand * vec3(1.0) +
                 rim * vec3(1.0)) *
                lightColor;

            finalColor += direct * (1.0 - shadow);
        }
    }

    if (renderMode != 0) {
        float edge = detectNprEdge(uv, pos, N);
        finalColor = mix(finalColor, vec3(0.02), edge);
    }

    Color = vec4(finalColor, 1.0);
}
