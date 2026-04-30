#version 430 core

uniform vec2 iResolution;
uniform sampler2D positionSampler;
uniform sampler2D normalSampler;
uniform sampler2D depthSampler;
uniform mat4 view;
uniform mat4 projection;
uniform float ssaoRadius;
uniform float ssaoStrength;
uniform int ssaoSamples;

layout(location = 0) out float AO;

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 fallbackTangent(vec3 normal)
{
    vec3 axis = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                      : vec3(0.0, 1.0, 0.0);
    return normalize(cross(axis, normal));
}

vec3 hemisphereSample(int index, int count, float angle)
{
    float fi = float(index) + 0.5;
    float fc = max(float(count), 1.0);
    float z = 1.0 - fi / fc;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float theta = fi * 2.39996323 + angle;
    float scale = fi / fc;
    scale = mix(0.15, 1.0, scale * scale);
    return vec3(cos(theta) * r, sin(theta) * r, z) * scale;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / iResolution;
    vec3 worldPos = texture(positionSampler, uv).xyz;
    vec3 normal = texture(normalSampler, uv).xyz;

    if (length(normal) < 1e-4) {
        AO = 1.0;
        return;
    }

    normal = normalize(normal);
    vec3 viewPos = (view * vec4(worldPos, 1.0)).xyz;

    float angle = hash12(gl_FragCoord.xy) * 6.2831853;
    vec3 randomVec = normalize(vec3(
        hash12(gl_FragCoord.xy + vec2(13.1, 7.7)) * 2.0 - 1.0,
        hash12(gl_FragCoord.xy + vec2(3.3, 19.9)) * 2.0 - 1.0,
        hash12(gl_FragCoord.xy + vec2(23.5, 2.1)) * 2.0 - 1.0));

    vec3 tangent = randomVec - normal * dot(randomVec, normal);
    if (length(tangent) < 1e-4) {
        tangent = fallbackTangent(normal);
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent));
    mat3 tbn = mat3(tangent, bitangent, normal);

    int sampleCount = clamp(ssaoSamples, 1, 64);
    float occlusion = 0.0;

    for (int i = 0; i < 64; ++i) {
        if (i >= sampleCount) {
            break;
        }

        vec3 sampleOffset = tbn * hemisphereSample(i, sampleCount, angle);
        vec3 sampleWorldPos = worldPos + sampleOffset * ssaoRadius;
        vec4 offset = projection * view * vec4(sampleWorldPos, 1.0);

        if (abs(offset.w) < 1e-5) {
            continue;
        }

        vec2 sampleUv = offset.xy / offset.w * 0.5 + 0.5;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 ||
            sampleUv.y < 0.0 || sampleUv.y > 1.0) {
            continue;
        }

        float sampledDepth = texture(depthSampler, sampleUv).r;
        vec3 sampledWorldPos = texture(positionSampler, sampleUv).xyz;
        if (abs(sampledDepth) < 1e-6 && length(sampledWorldPos) < 1e-6) {
            continue;
        }

        float sampledViewZ = (view * vec4(sampledWorldPos, 1.0)).z;
        float targetViewZ = (view * vec4(sampleWorldPos, 1.0)).z;
        float rangeCheck = smoothstep(
            0.0,
            1.0,
            ssaoRadius / max(abs(viewPos.z - sampledViewZ), 1e-4));

        if (sampledViewZ >= targetViewZ + 0.025) {
            occlusion += rangeCheck;
        }
    }

    float ao = 1.0 - occlusion / float(sampleCount);
    AO = pow(clamp(ao, 0.0, 1.0), ssaoStrength);
}
