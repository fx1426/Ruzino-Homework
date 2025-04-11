import pbrlib.genslang.lib.mx_microfacet_specular;

struct EnvironmentPrefilterData {
    float4x4 u_envMatrix;
    float3 u_envLightIntensity;
    int u_envRadianceMips;
    int u_envIrradianceMips;
};
Texture2D u_envRadiance : register(t0);
SamplerState u_envRadiance_sampler : register(s0);
Texture2D u_envIrradiance : register(t1);
SamplerState u_envIrradiance_sampler : register(s1);
ConstantBuffer<EnvironmentPrefilterData> u_envPrefilterData : register(b0);

// Return the mip level associated with the given alpha in a prefiltered
// environment.
float mx_latlong_alpha_to_lod(float alpha)
{
    float lodBias = (alpha < 0.25) ? sqrt(alpha) : 0.5 * alpha + 0.375;
    return lodBias * float(u_envPrefilterData.u_envRadianceMips - 1);
}

float3 mx_environment_radiance(
    float3 N,
    float3 V,
    float3 X,
    float2 alpha,
    int distribution,
    FresnelData fd)
{
    N = mx_forward_facing_normal(N, V);
    float3 L = fd.refraction ? mx_refraction_solid_sphere(-V, N, fd.ior.x)
                             : -reflect(V, N);

    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float avgAlpha = mx_average_alpha(alpha);
    float3 F = mx_compute_fresnel(NdotV, fd);
    float G = mx_ggx_smith_G2(NdotV, NdotV, avgAlpha);
    float3 FG = fd.refraction ? float3(1.0) - (F * G) : F * G;

    float3 Li = mx_latlong_map_lookup(
        L,
        u_envPrefilterData.u_envMatrix,
        mx_latlong_alpha_to_lod(avgAlpha),
        u_envRadiance,
        u_envRadiance_sampler);
    return Li * FG * u_envPrefilterData.u_envLightIntensity;
}

float3 mx_environment_irradiance(float3 N)
{
    float3 Li = mx_latlong_map_lookup(
        N,
        u_envPrefilterData.u_envMatrix,
        0.0,
        u_envIrradiance,
        u_envIrradiance_sampler);
    return Li * u_envPrefilterData.u_envLightIntensity;
}
