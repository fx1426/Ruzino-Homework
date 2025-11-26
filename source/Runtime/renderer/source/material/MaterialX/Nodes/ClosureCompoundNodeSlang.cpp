//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "ClosureCompoundNodeSlang.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN
ShaderNodeImplPtr ClosureCompoundNodeSlang::create()
{
    return std::make_shared<ClosureCompoundNodeSlang>();
}

void ClosureCompoundNodeSlang::addClassification(ShaderNode& node) const
{
    // Add classification from the graph implementation.
    node.addClassification(_rootGraph->getClassification());
}

// TODO: add real tangent information
static std::string sample_source_code_fallback = R"(
import Utils.Math.MathHelpers;
#include "utils/random.slangh"

float3 sample_fallback(
    inout uint seed,
    out float pdf,
    in MaterialDataBlob blob_data, 
    in VertexInfo vertexInfo
)
{
    // Sample the direction
    float3 sampledDir = sample_cosine_hemisphere_concentric(random_float2(seed), pdf);

    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(vertexInfo.normalW, float4(1, 0, 0, 1), valid);
    sampledDir = sf.fromLocal(sampledDir);

    return sampledDir;
}

)";

static std::string sample_source_code_standard_surface = R"(
import pbrlib.genslang.mx_roughness_anisotropy;
import pbrlib.genslang.lib.mx_microfacet;
import pbrlib.genslang.lib.mx_microfacet_specular;
import pbrlib.genslang.lib.mx_microfacet_diffuse;
import utils.Math.MathHelpers;
#include "utils/Math/MathConstants.slangh"
#include "utils/random.slangh"

// Calculate luminance of a color
float luminance(float3 color)
{
    return dot(color, float3(0.212671, 0.715160, 0.072169));
}

// Exact Fresnel reflectance
float fresnel_dielectric(float cosTheta, float eta)
{
    float sinThetaT2 = (1.0 - cosTheta * cosTheta) / (eta * eta);
    
    // Check for total internal reflection
    if (sinThetaT2 >= 1.0) {
        return 1.0;
    }
    
    float cosThetaT = sqrt(1.0 - sinThetaT2);
    
    float Rs = (cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT);
    float Rp = (eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT);
    
    return 0.5 * (Rs * Rs + Rp * Rp);
}

// Sample diffuse component using cosine hemisphere sampling
float3 sample_diffuse_lobe(float2 u, out float pdf)
{
    float3 L = sample_cosine_hemisphere_concentric(u, pdf);
    return L;
}

// Sample specular reflection using GGX distribution
float3 sample_specular_reflection(float2 u, float3 V, float2 alpha, out float pdf)
{
    // Sample microfacet normal using GGX VNDF distribution
    // VNDF already ensures H is in the correct hemisphere for V
    float3 H = mx_ggx_importance_sample_VNDF(u, V, alpha);
    
    // Reflect view direction around microfacet normal
    float3 L = reflect(-V, H);
    
    // Check if the reflection is valid (above surface)
    // Use a small epsilon to tolerate numerical errors, especially for grazing angles
    if (L.z <= -M_FLOAT_EPS) {
        pdf = 0.0;
        return float3(0.0);
    }
    
    // Compute PDF using correct VNDF formulation
    float VdotH = max(dot(V, H), M_FLOAT_EPS);
    float NdotV = max(V.z, M_FLOAT_EPS);
    
    // VNDF PDF: D(H) * G1(V) * max(0, V·H) / NdotV
    float D = mx_ggx_NDF(H, alpha);
    float G1 = mx_ggx_smith_G1_aniso(NdotV, V.x, V.y, alpha.x, alpha.y);
    float vndf_pdf = D * G1 * VdotH / NdotV;
    
    // Transform to reflection direction: PDF_L = PDF_H / (4 * V·H)
    pdf = vndf_pdf / (4.0 * VdotH);
    return L;
}

// Sample transmission using GGX distribution and Snell's law
float3 sample_transmission(float2 u, float3 V, float2 alpha, float eta, out float pdf)
{
    // Sample microfacet normal using GGX VNDF distribution
    float3 H = mx_ggx_importance_sample_VNDF(u, V, alpha);
    
    // For transmission, ensure half vector points toward the incident side
    // This is necessary because refraction uses a different half-vector convention
    if (dot(V, H) < 0.0) {
        H = -H;
    }
    
    float VdotH = max(dot(V, H), M_FLOAT_EPS);
    float NdotV = max(V.z, M_FLOAT_EPS);
    
    // Check for total internal reflection
    float discriminant = 1.0 - (1.0 - VdotH * VdotH) / (eta * eta);
    if (discriminant < 0.0) {
        pdf = M_FLOAT_EPS;
        return reflect(-V, H); // Return reflection direction if TIR occurs
    }
    
    // Compute refracted direction using Snell's law
    float sqrt_discriminant = sqrt(discriminant);
    float3 L = -V / eta + H * (VdotH / eta - sqrt_discriminant);
    
    // Check if transmission is valid (below surface)
    if (L.z >= 0.0) {
        pdf = 0.0;
        return float3(0.0);
    }
    
    float LdotH = max(abs(dot(L, H)), M_FLOAT_EPS);
    
    // Compute PDF using GGX VNDF for transmission
    float D = mx_ggx_NDF(H, alpha);
    float G1 = mx_ggx_smith_G1_aniso(NdotV, V.x, V.y, alpha.x, alpha.y);
    float vndf_pdf = D * G1 * VdotH / NdotV;
    
    // Transform to transmission direction with Jacobian
    float denom = VdotH + LdotH / eta;
    float jacobian = (eta * eta * LdotH) / (denom * denom);
    pdf = vndf_pdf * jacobian;
    
    return L;
}

float3 sample_standard_surface(
    VertexData vd, 
    float3 V, 
    float base, 
    float3 base_color, 
    float diffuse_roughness, 
    float metalness, 
    float specular, 
    float3 specular_color, 
    float specular_roughness, 
    float specular_IOR, 
    float specular_anisotropy, 
    float specular_rotation, 
    float transmission, 
    float3 transmission_color, 
    float transmission_depth, 
    float3 transmission_scatter, 
    float transmission_scatter_anisotropy, 
    float transmission_dispersion, 
    float transmission_extra_roughness, 
    float subsurface, 
    float3 subsurface_color, 
    float3 subsurface_radius, 
    float subsurface_scale, 
    float subsurface_anisotropy, 
    float sheen, 
    float3 sheen_color, 
    float sheen_roughness, 
    float coat, 
    float3 coat_color, 
    float coat_roughness, 
    float coat_anisotropy, 
    float coat_rotation, 
    float coat_IOR, 
    float3 coat_normal, 
    float coat_affect_color, 
    float coat_affect_roughness, 
    float thin_film_thickness, 
    float thin_film_IOR, 
    float emission, 
    float3 emission_color, 
    float3 opacity, 
    bool thin_walled, 
    float3 normal, 
    float3 tangent,
    uint eta_flipped, // 0 for normal, 1 for flipped
    inout uint seed,
    out float pdf
)
{
    // Create base layer shading frame
    float3 rotated_tangent;
    float rotation_angle = specular_rotation * 360.0;
    mx_rotate_vector3(tangent, rotation_angle, normal, rotated_tangent);
    
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(normal, float4(rotated_tangent, 1.0), valid);
    
    // Transform view direction to local space
    float3 V_local = sf.toLocal(V);
    float NdotV = clamp(V_local.z, M_FLOAT_EPS, 1.0);

    // Compute base layer anisotropic roughness
    float2 alpha;
    mx_roughness_anisotropy(specular_roughness, specular_anisotropy, alpha);
    
    // Compute coat layer properties with its own shading frame
    float3 coat_rotated_tangent;
    float coat_rotation_angle = coat_rotation * 360.0;
    mx_rotate_vector3(tangent, coat_rotation_angle, coat_normal, coat_rotated_tangent);
    coat_rotated_tangent = normalize(coat_rotated_tangent);
    
    bool coat_valid;
    ShadingFrame coat_sf = ShadingFrame.createSafe(coat_normal, float4(coat_rotated_tangent, 1.0), coat_valid);
    float3 V_coat_local = coat_sf.toLocal(V);
    float coat_NdotV = clamp(V_coat_local.z, M_FLOAT_EPS, 1.0);
    
    float coat_weight = clamp(coat, 0.0, 1.0);
    float2 coat_alpha;
    mx_roughness_anisotropy(coat_roughness, coat_anisotropy, coat_alpha);
    float coat_fresnel = fresnel_dielectric(coat_NdotV, coat_IOR);
    
    // Compute base layer properties
    float eta = eta_flipped != 0 ? (1.0 / specular_IOR) : specular_IOR;
    float base_fresnel = fresnel_dielectric(NdotV, eta);
    
    // Layered sampling weights
    // Coat reflects: coat * F_coat
    // Base receives: (1 - coat * F_coat) of the incoming light
    float coat_reflection_prob = coat_weight * coat_fresnel;
    float coat_sample_weight = coat_reflection_prob;
    float base_sample_weight = 1.0 - coat_sample_weight;
    
    // Base layer weights (same as before, but scaled by base_sample_weight)
    float metal_weight = metalness;
    float nonmetal_weight = 1.0 - metalness;
    float diffuse_weight = base * (1.0 - base_fresnel) * luminance(base_color) * (1 - transmission);
    float reflection_weight = base_fresnel * specular;
    float transmission_weight = (1.0 - base_fresnel) * transmission * luminance(transmission_color);
    
    // Normalize base layer weights
    float total_nonmetal_weight = diffuse_weight + reflection_weight + transmission_weight;
    if (total_nonmetal_weight > M_FLOAT_EPS) {
        diffuse_weight /= total_nonmetal_weight;
        reflection_weight /= total_nonmetal_weight;
        transmission_weight /= total_nonmetal_weight;
    }
    
    // Sample component: coat vs base
    float component_sample = random_float(seed);
    float3 L_local;
    bool sampled_coat = (component_sample < coat_sample_weight);
    
    if (sampled_coat) {
        // Sample coat layer in its own coordinate frame
        float2 u = random_float2(seed);
        float3 L_coat_local = sample_specular_reflection(u, V_coat_local, coat_alpha, pdf);
        
        if (pdf <= 0.0) {
            pdf = 0.0;
            return float3(0.0);
        }
        
        // Transform from coat frame to world, then to base frame
        float3 L_world = coat_sf.fromLocal(L_coat_local);
        L_local = sf.toLocal(L_world);
    }
    else {
        // Sample base layer
        float base_sample = (component_sample - coat_sample_weight) / base_sample_weight;
        bool sampled_metal = (base_sample < metal_weight);
        
        if (sampled_metal) {
            // Sample metal reflection
            float2 u = random_float2(seed);
            L_local = sample_specular_reflection(u, V_local, alpha, pdf);
            
            if (pdf <= 0.0) {
                pdf = 0.0;
                return float3(0.0);
            }
        }
        else {
            // Sample non-metal component
            float nonmetal_sample = (base_sample - metal_weight) / nonmetal_weight;
            
            if (nonmetal_sample < diffuse_weight) {
                // Sample diffuse
                float2 u = random_float2(seed);
                L_local = sample_diffuse_lobe(u, pdf);
            }
            else if (nonmetal_sample < diffuse_weight + reflection_weight) {
                // Sample non-metal reflection
                float2 u = random_float2(seed);
                L_local = sample_specular_reflection(u, V_local, alpha, pdf);
            }
            else {
                // Sample transmission
                float2 u = random_float2(seed);
                L_local = sample_transmission(u, V_local, alpha, eta, pdf);
            }
            
            if (pdf <= 0.0) {
                pdf = 0.0;
                return float3(0.0);
            }
        }
    }
    
    // Calculate MIS PDF: compute PDF for coat and base layers
    float coat_pdf = 0.0;
    float base_pdf = 0.0;
    
    // Coat PDF (computed in coat's coordinate frame)
    // Transform L from base frame to world, then to coat frame
    float3 L_world = sf.fromLocal(L_local);
    float3 L_coat_local = coat_sf.toLocal(L_world);
    
    if (L_coat_local.z > -M_FLOAT_EPS) {
        float3 H_coat = normalize(V_coat_local + L_coat_local);
        float VdotH_coat = max(dot(V_coat_local, H_coat), M_FLOAT_EPS);
        
        float D = mx_ggx_NDF(H_coat, coat_alpha);
        float G1 = mx_ggx_smith_G1_aniso(coat_NdotV, V_coat_local.x, V_coat_local.y, coat_alpha.x, coat_alpha.y);
        float vndf_pdf = D * G1 * VdotH_coat / coat_NdotV;
        coat_pdf = vndf_pdf / (4.0 * VdotH_coat);
    }
    
    // Base layer PDF (metal + nonmetal)
    float metal_pdf = 0.0;
    float nonmetal_pdf = 0.0;
    
    // Metal path PDF (only has reflection component)
    if (L_local.z > -M_FLOAT_EPS) {
        float3 H = normalize(V_local + L_local);
        float VdotH = max(dot(V_local, H), M_FLOAT_EPS);
        
        float D = mx_ggx_NDF(H, alpha);
        float G1 = mx_ggx_smith_G1_aniso(NdotV, V_local.x, V_local.y, alpha.x, alpha.y);
        float vndf_pdf = D * G1 * VdotH / NdotV;
        metal_pdf = vndf_pdf / (4.0 * VdotH);
    }
    
    // Non-metal path PDF (compute all three components)
    if (total_nonmetal_weight > M_FLOAT_EPS) {
        float diffuse_pdf = 0.0;
        float reflection_pdf = 0.0;
        float transmission_pdf = 0.0;
        
        if (L_local.z > -M_FLOAT_EPS) {
            // Diffuse PDF (only valid for directions above surface)
            diffuse_pdf = max(L_local.z, 0.0) * M_1_PI;
            
            // Reflection PDF
            float3 H = normalize(V_local + L_local);
            float VdotH = max(dot(V_local, H), M_FLOAT_EPS);
            
            float D = mx_ggx_NDF(H, alpha);
            float G1 = mx_ggx_smith_G1_aniso(NdotV, V_local.x, V_local.y, alpha.x, alpha.y);
            float vndf_pdf = D * G1 * VdotH / NdotV;
            reflection_pdf = vndf_pdf / (4.0 * VdotH);    
        }
        else if (L_local.z < -M_FLOAT_EPS) {
            // Transmission PDF - compute for downward directions
            float3 H = normalize(V_local + L_local * eta);
            
            // Ensure half vector points toward the incident side
            if (H.z < 0.0) {
                H = -H;
            }
            
            float VdotH = max(dot(V_local, H), M_FLOAT_EPS);
            float LdotH = max(abs(dot(L_local, H)), M_FLOAT_EPS);
            
            float D = mx_ggx_NDF(H, alpha);
            float G1 = mx_ggx_smith_G1_aniso(NdotV, V_local.x, V_local.y, alpha.x, alpha.y);
            float vndf_pdf = D * G1 * VdotH / NdotV;
            
            // Transform to transmission direction with proper Jacobian
            float denom = VdotH + LdotH / eta;
            float jacobian = (eta * eta * LdotH) / (denom * denom);
            transmission_pdf = vndf_pdf * jacobian;
        }
        
        nonmetal_pdf = diffuse_pdf * diffuse_weight + reflection_pdf * reflection_weight + transmission_pdf * transmission_weight;
    }
    
    // Combine base layer PDFs
    base_pdf = metal_pdf * metal_weight + nonmetal_pdf * nonmetal_weight;
    
    // Final MIS PDF: coat + base
    pdf = coat_pdf * coat_sample_weight + base_pdf * base_sample_weight;
    pdf = max(pdf, M_FLOAT_EPS);

    // Transform to world space
    return sf.fromLocal(L_local);
}

)";

static std::string sample_source_code_usd_preview_surface = R"(
import pbrlib.genslang.mx_roughness_anisotropy;
import pbrlib.genslang.lib.mx_microfacet;
import pbrlib.genslang.lib.mx_microfacet_specular;
import pbrlib.genslang.lib.mx_microfacet_diffuse;
import utils.Math.MathHelpers;
#include "utils/Math/MathConstants.slangh"
#include "utils/random.slangh"

// Calculate luminance of a color
float luminance(float3 color)
{
    return dot(color, float3(0.212671, 0.715160, 0.072169));
}

// Exact Fresnel reflectance for USD Preview Surface
float fresnel_dielectric(float cosTheta, float eta)
{
    float sinThetaT2 = (1.0 - cosTheta * cosTheta) / (eta * eta);
    
    // Check for total internal reflection
    if (sinThetaT2 >= 1.0) {
        return 1.0;
    }
    
    float cosThetaT = sqrt(1.0 - sinThetaT2);
    
    float Rs = (cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT);
    float Rp = (eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT);
    
    return 0.5 * (Rs * Rs + Rp * Rp);
}

// Sample diffuse component using cosine hemisphere sampling
float3 sample_diffuse_lobe(float2 u, out float pdf)
{
    float3 L = sample_cosine_hemisphere_concentric(u, pdf);
    return L;
}

// Sample specular reflection using GGX distribution
float3 sample_specular_reflection(float2 u, float3 V, float roughness, out float pdf)
{
    float2 alpha = float2(roughness * roughness, roughness * roughness);
    
    // Sample microfacet normal using GGX VNDF distribution
    float3 H = mx_ggx_importance_sample_VNDF(u, V, alpha);
    
    // Reflect view direction around microfacet normal
    float3 L = reflect(-V, H);
    
    // Check if the reflection is valid (above surface)
    // Use a small epsilon to tolerate numerical errors, especially for grazing angles
    if (L.z <= -M_FLOAT_EPS) {
        pdf = 0.0;
        return float3(0.0);
    }
    
    // Compute PDF using correct VNDF formulation
    float NdotH = max(H.z, M_FLOAT_EPS);
    float VdotH = max(dot(V, H), M_FLOAT_EPS);
    float NdotV = max(V.z, M_FLOAT_EPS);
    
    // VNDF PDF: D(H) * G1(V) * max(0, V·H) / NdotV
    float D = mx_ggx_NDF(H, alpha);
    float G1 = mx_ggx_smith_G1_aniso(NdotV, V.x, V.y, alpha.x, alpha.y);
    float vndf_pdf = D * G1 * VdotH / NdotV;
    
    // Transform to reflection direction: PDF_L = PDF_H / (4 * V·H)
    pdf = vndf_pdf / (4.0 * VdotH);
    
    return L;
}

float3 sample_preview_surface(
    VertexData vd,
    float3 V,
    float3 diffuseColor,
    float3 emissiveColor,
    int useSpecularWorkflow,
    float3 specularColor,
    float metallic,
    float roughness,
    float clearcoat,
    float clearcoatRoughness,
    float opacity,
    int opacityMode,
    float opacityThreshold,
    float ior,
    float3 normal,
    float displacement,
    float occlusion,
    uint eta_flipped,
    inout uint seed,
    out float pdf
)
{
    // Create shading frame
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(vd.normalWorld, float4(1, 0, 0, 1), valid);

    // Transform view direction to local space
    float3 V_local = sf.toLocal(V);
    float NdotV = clamp(V_local.z, M_FLOAT_EPS, 1.0);

    // Compute material properties
    float eta = eta_flipped != 0 ? (1.0 / ior) : ior;
    float fresnel = fresnel_dielectric(NdotV, eta);

    // Determine workflow
    bool isSpecularWorkflow = false;

    // Compute final diffuse and specular colors based on workflow
    float3 final_diffuse_color;
    float3 final_specular_color;
    float final_metallic;

    if (isSpecularWorkflow) {
        final_diffuse_color = diffuseColor;
        final_specular_color = specularColor;
        final_metallic = 0.0; // Specular workflow doesn't use metallic
    } else {
        // Metallic workflow
        final_metallic = metallic;
        final_diffuse_color = diffuseColor * (1.0 - metallic);
        final_specular_color = lerp(float3(0.04), diffuseColor, metallic);
    }

    // Metal path weights
    float metal_weight = final_metallic;

    // Non-metal path weights
    float nonmetal_weight = 1.0 - final_metallic;
    float diffuse_weight = (1.0 - fresnel) * luminance(final_diffuse_color);
    float reflection_weight = fresnel;

    float total_nonmetal_weight = diffuse_weight + reflection_weight;
    if (total_nonmetal_weight > M_FLOAT_EPS) {
        diffuse_weight /= total_nonmetal_weight;
        reflection_weight /= total_nonmetal_weight;
    }

    // Sample component based on first level branching
    float component_sample = random_float(seed);
    float3 L_local;
    bool sampled_metal = (component_sample < metal_weight);

    if (sampled_metal) {
        // Sample metal reflection
        float2 u = random_float2(seed);
        L_local = sample_specular_reflection(u, V_local, roughness, pdf);
        
        if (pdf <= 0.0) {
            pdf = 0.0;
            return float3(0.0);
        }
    }
    else {
        // Sample non-metal component
        float nonmetal_sample = (component_sample - metal_weight) / nonmetal_weight;
        
        if (nonmetal_sample < diffuse_weight) {
            // Sample diffuse
            float2 u = random_float2(seed);
            L_local = sample_diffuse_lobe(u, pdf);
        }
        else {
            // Sample non-metal reflection
            float2 u = random_float2(seed);
            L_local = sample_specular_reflection(u, V_local, roughness, pdf);
        }
        
        if (pdf <= 0.0) {
            pdf = 0.0;
            return float3(0.0);
        }
    }

    // Calculate MIS PDF: both paths need to compute their probability for the sampled direction
    float metal_pdf = 0.0;
    float nonmetal_pdf = 0.0;

    // Metal path PDF (only has reflection component)
    // Use consistent threshold with sampling: allow directions very close to horizontal
    if (L_local.z > -M_FLOAT_EPS) {
        float3 H = normalize(V_local + L_local);
        float NdotH = max(H.z, M_FLOAT_EPS);
        float VdotH = max(dot(V_local, H), M_FLOAT_EPS);
        
        float2 alpha = float2(roughness * roughness, roughness * roughness);
        float D = mx_ggx_NDF(H, alpha);
        float G1 = mx_ggx_smith_G1_aniso(NdotV, V_local.x, V_local.y, alpha.x, alpha.y);
        float vndf_pdf = D * G1 * VdotH / NdotV;
        metal_pdf = vndf_pdf / (4.0 * VdotH);
    }

    // Non-metal path PDF (compute diffuse and reflection components)
    if (total_nonmetal_weight > M_FLOAT_EPS && L_local.z > -M_FLOAT_EPS) {
        // Diffuse PDF (only valid for directions above surface)
        float diffuse_pdf = max(L_local.z, 0.0) * M_1_PI;
        
        // Reflection PDF
        float3 H = normalize(V_local + L_local);
        float NdotH = max(H.z, M_FLOAT_EPS);
        float VdotH = max(dot(V_local, H), M_FLOAT_EPS);
        
        float2 alpha = float2(roughness * roughness, roughness * roughness);
        float D = mx_ggx_NDF(H, alpha);
        float G1 = mx_ggx_smith_G1_aniso(NdotV, V_local.x, V_local.y, alpha.x, alpha.y);
        float vndf_pdf = D * G1 * VdotH / NdotV;
        float reflection_pdf = vndf_pdf / (4.0 * VdotH);
        
        nonmetal_pdf = diffuse_pdf * diffuse_weight + reflection_pdf * reflection_weight;
    }

    // Final MIS PDF combining both paths
    pdf = metal_pdf * metal_weight + nonmetal_pdf * nonmetal_weight;
    pdf = max(pdf, M_FLOAT_EPS);

    // Transform to world space
    return sf.fromLocal(L_local);
 
}

)";

void ClosureCompoundNodeSlang::emitFunctionDefinition(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const Syntax& syntax = shadergen.getSyntax();

        bool isStandardSurface =
            _functionName.find("standard_surface") != string::npos;
        bool isUsdPreviewSurface =
            _functionName.find("UsdPreviewSurface") != string::npos;

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        string delim = "";

        // Begin function signature
        shadergen.emitLineBegin(stage);

        shadergen.emitString("void " + _functionName  + "(", stage);

        if (context.getShaderGenerator().nodeNeedsClosureData(node))
        {
            shadergen.emitString(delim + HW::CLOSURE_DATA_TYPE + " " + HW::CLOSURE_DATA_ARG + ", ", stage);
        }

        auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty()) {
            shadergen.emitString(
                delim + vertexData.getName() + " " + vertexData.getInstance(),
                stage);
            delim = ", ";
        }

        const string& type = syntax.getTypeName(Type::VECTOR3);
        shadergen.emitString(delim + type + " " + HW::DIR_L, stage);
        shadergen.emitString(", " + type + " " + HW::DIR_V, stage);
        // eta
        shadergen.emitString(delim + "uint eta_flipped", stage);

        // Add all inputs
        for (ShaderGraphInputSocket* inputSocket : _rootGraph->getInputSockets())
        {
            shadergen.emitString(delim + syntax.getTypeName(inputSocket->getType()) + " " + inputSocket->getVariable(), stage);
            delim = ", ";
        }

        // Add all outputs
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            shadergen.emitString(delim + syntax.getOutputTypeName(outputSocket->getType()) + " " + outputSocket->getVariable(), stage);
            delim = ", ";
        }

        // End function signature
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage, false);

        // Begin function body
        shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);

        if (isStandardSurface) {
            shadergen.emitLine("if (transmission > 0.0)", stage, false);
            shadergen.emitLine("if (eta_flipped > 0.0)", stage, false);
            shadergen.emitLine("specular_IOR = 1.0 / specular_IOR", stage);
        }

        // Emit all texturing nodes. These are inputs to the
        // closure nodes and need to be emitted first.
        shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

        // Emit function calls for internal closures nodes connected to the graph sockets.
        // These will in turn emit function calls for any dependent closure nodes upstream.
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            if (outputSocket->getConnection())
            {
                const ShaderNode* upstream = outputSocket->getConnection()->getNode();
                if (upstream->getParent() == _rootGraph.get() &&
                    (upstream->hasClassification(ShaderNode::Classification::CLOSURE) || upstream->hasClassification(ShaderNode::Classification::SHADER)))
                {
                    shadergen.emitFunctionCall(*upstream, context, stage);
                }
            }
        }

        // Emit final results
        for (ShaderGraphOutputSocket* outputSocket : _rootGraph->getOutputSockets())
        {
            const string result = shadergen.getUpstreamResult(outputSocket, context);
            shadergen.emitLine(outputSocket->getVariable() + " = " + result, stage);
        }

        // End function body
        shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);

        // Emit the sample fallback and standard surface sampling
        if (isStandardSurface) {
            shadergen.emitLine(
                sample_source_code_standard_surface, stage, false);
        }
        else if (isUsdPreviewSurface) {
            shadergen.emitLine(
                sample_source_code_usd_preview_surface, stage, false);
        }
        else
            shadergen.emitLine(sample_source_code_fallback, stage, false);
    }
}

void ClosureCompoundNodeSlang::emitFunctionCall(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        // Emit function calls for all child nodes to the vertex shader stage
        shadergen.emitFunctionCalls(*_rootGraph, context, stage);
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit calls for any closure dependencies upstream from this node.
        shadergen.emitDependentFunctionCalls(
            node, context, stage, ShaderNode::Classification::CLOSURE);

        // Declare the output variables
        emitOutputVariables(node, context, stage);

        shadergen.emitLineBegin(stage);
        string delim = "";

        // Emit function name.
        shadergen.emitString(_functionName + "(", stage);

        // Check if we have a closure context to modify the function call.
        if (context.getShaderGenerator().nodeNeedsClosureData(node))
        {
            shadergen.emitString(delim + HW::CLOSURE_DATA_ARG + ", ", stage);
        }

        auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty()) {
            shadergen.emitString(delim + vertexData.getInstance(), stage);
            delim = ", ";
        }

        shadergen.emitString(delim + HW::DIR_L + ", " + HW::DIR_V, stage);
        shadergen.emitString(delim + "eta_flipped", stage);
        
        // Emit all inputs.
        for (ShaderInput* input : node.getInputs()) {
            shadergen.emitString(delim, stage);
            shadergen.emitInput(input, context, stage);
            delim = ", ";
        }

        // Emit all outputs.
        for (size_t i = 0; i < node.numOutputs(); ++i) {
            shadergen.emitString(delim, stage);
            shadergen.emitOutput(
                node.getOutput(i), false, false, context, stage);
            delim = ", ";
        }
        
        // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);

        // Check if this is a standard surface material
        bool isStandardSurface =
            _functionName.find("standard_surface") != string::npos;
        bool isUsdPreviewSurface =
            _functionName.find("UsdPreviewSurface") != string::npos;
        if (isStandardSurface) {
            // Call the standard surface sampling function
            shadergen.emitLineBegin(stage);
            shadergen.emitString(
                "sampled_direction = sample_standard_surface(vd, V, ",
                stage);

            // Emit all the standard surface parameters
            string delim = "";
            for (ShaderInput* input : node.getInputs()) {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }
            shadergen.emitString(delim + "eta_flipped", stage);

            shadergen.emitString(", seed, pdf)", stage);
            shadergen.emitLineEnd(stage);
        }
        else if (isUsdPreviewSurface) {
            // Call the USD preview surface sampling function
            shadergen.emitLineBegin(stage);
            shadergen.emitString(
                "sampled_direction = sample_preview_surface(vd, V, ",
                stage);

            // Emit all the USD preview surface parameters
            string delim = "";
            for (ShaderInput* input : node.getInputs()) {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }
            shadergen.emitString(delim + "eta_flipped", stage);

            shadergen.emitString(", seed, pdf)", stage);
            shadergen.emitLineEnd(stage);
        }
        else {
            // Call the sample fallback, and eval the sampled direction
            shadergen.emitLine(
                "sampled_direction = sample_fallback(seed, pdf, data, "
                "vertexInfo)",
                stage);
        }
        
        // Use that direction to replace HW::DIR_L and re-evaluate the material
        shadergen.emitLine(
            "surfaceshader sampled_weight_out = "
            "surfaceshader(float3(0.0),float3(0.0));",
            stage);
        shadergen.emitLineBegin(stage);
        string delim2 = "";

        // Emit function name.
        shadergen.emitString(_functionName + "(", stage);

        if (context.getShaderGenerator().nodeNeedsClosureData(node))
        {
            shadergen.emitString(delim2 + HW::CLOSURE_DATA_ARG + ", ", stage);
        }

        auto& vertexData2 = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData2.empty()) {
            shadergen.emitString(delim2 + vertexData2.getInstance(), stage);
            delim2 = ", ";
        }

        shadergen.emitString(
            delim2 + "sampled_direction" + ", " + HW::DIR_V, stage);

        shadergen.emitString(delim2 + "eta_flipped", stage);

        // Emit all inputs.
        for (ShaderInput* input : node.getInputs()) {
            shadergen.emitString(delim2, stage);
            shadergen.emitInput(input, context, stage);
            delim2 = ", ";
        }

        // Emit all outputs.
        shadergen.emitString(delim2, stage);
        shadergen.emitString("sampled_weight_out", stage);

        // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
        shadergen.emitLine(
            "sampled_weight = sampled_weight_out.color", stage);
    }
}

MATERIALX_NAMESPACE_END
