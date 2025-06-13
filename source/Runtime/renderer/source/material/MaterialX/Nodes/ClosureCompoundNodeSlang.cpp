//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "ClosureCompoundNodeSlang.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/ShaderGenerator.h>

#include "Logger/Logger.h"
#include "MaterialXGenShader/Nodes/ClosureCompoundNode.h"

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
    ShadingFrame sf = ShadingFrame.createSafe(vertexInfo.normalW, float4(0, 0, 0, 1), valid);
    sampledDir = sf.fromLocal(sampledDir);

    return sampledDir;
}

)";

static std::string sample_source_code_standard_surface = R"(
import pbrlib.genslang.mx_roughness_anisotropy;
#include "utils/Math/MathConstants.slangh"

float3 sample_standard_surface(
    VertexData vd, 
    SamplerState sampler, 
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
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(normal, float4(1, 0, 0, 1), valid);

    float diffuse_weight = base * (1.0 - metalness) * (1.0 - transmission);
    // Specular reflection should always be present for dielectrics (Fresnel reflection)
    float specular_weight = specular * (1.0 - metalness);
    float metal_weight = metalness;
    float transmission_weight = transmission * (1.0 - metalness);
    
    // Normalize weights
    float total_weight = diffuse_weight + specular_weight + metal_weight + transmission_weight;
    if (total_weight <= 0.0) {
        // Fallback to cosine hemisphere sampling
        float3 L = sample_cosine_hemisphere_concentric(random_float2(seed), pdf);
        return sf.fromLocal(L);
    }
    
    diffuse_weight /= total_weight;
    specular_weight /= total_weight;
    metal_weight /= total_weight;
    transmission_weight /= total_weight;
    
    float3 L;
    pdf = 0.0;

    // Calculate layer weights for importance sampling
    
    // Choose sampling method based on weights
    float r = random_float(seed);
    
    if (r < metal_weight) {
        // Sample metallic reflection (same as specular but for metals)
        float2 roughness_vector;
        mx_roughness_anisotropy(specular_roughness, specular_anisotropy, roughness_vector);
        
        // Clamp roughness to avoid numerical issues
        float alpha = max(0.001, roughness_vector.x * roughness_vector.x);
        
        // Sample GGX distribution
        float2 xi = random_float2(seed);
        float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
        float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        float phi = 2.0 * M_PI * xi.x;
        
        // Create half vector in tangent space
        float3 H = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        H = sf.fromLocal(H);
        
        // Reflect view direction around half vector
        L = reflect(-V, H);
        
        // Calculate GGX PDF for this component
        float VdotH = max(0.001, dot(V, H));
        float NdotH = max(0.001, dot(normal, H));
        
        float alpha2 = alpha * alpha;
        float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
        float D = alpha2 / (M_PI * denom * denom);
        
        pdf += metal_weight * D * NdotH / (4.0 * VdotH);
        
    } else if (r < metal_weight + specular_weight) {
        // Sample dielectric reflection
        float2 roughness_vector;
        mx_roughness_anisotropy(specular_roughness, specular_anisotropy, roughness_vector);
        
        float alpha = max(0.001, roughness_vector.x * roughness_vector.x);
        
        float2 xi = random_float2(seed);
        float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
        float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        float phi = 2.0 * M_PI * xi.x;
        
        float3 H = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        H = sf.fromLocal(H);
        
        L = reflect(-V, H);
        
        float VdotH = max(0.001, dot(V, H));
        float NdotH = max(0.001, dot(normal, H));
        
        float alpha2 = alpha * alpha;
        float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
        float D = alpha2 / (M_PI * denom * denom);
        
        pdf += specular_weight * D * NdotH / (4.0 * VdotH);
          
    }   
         else if (r < metal_weight + specular_weight + transmission_weight) {
        // Sample transmission with roughness
        float eta;
        if (eta_flipped == 1) {
            // Going from material into air
            eta = specular_IOR;
        } else {
            // Going from air into material
            eta = 1.0 / specular_IOR;
        }

        // Use transmission roughness (with extra roughness added)
        float transmission_roughness = specular_roughness + transmission_extra_roughness;
        float2 roughness_vector;
        mx_roughness_anisotropy(transmission_roughness, specular_anisotropy, roughness_vector);
        
        float alpha = max(0.001, roughness_vector.x * roughness_vector.x);
        
        // Sample microfacet normal using GGX distribution
        float2 xi = random_float2(seed);
        float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
        float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        float phi = 2.0 * M_PI * xi.x;
        
        // Create microfacet normal in tangent space
        float3 H = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        H = sf.fromLocal(H);
        
        float VdotH = dot(V, H);
        
        // Compute refraction direction using microfacet normal
        float k = 1.0 - eta * eta * (1.0 - VdotH * VdotH);
        
        if (k < 0.0) {
            // Total internal reflection - fall back to specular reflection around microfacet
            L = reflect(-V, H);
            
            // Calculate reflection PDF
            float NdotH = max(0.001, dot(normal, H));
            float alpha2 = alpha * alpha;
            float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (M_PI * denom * denom);
            pdf += transmission_weight * D * NdotH / (4.0 * abs(VdotH));
        } else {
            // Refraction around microfacet using proper transmission formula
            // This matches the mx_surface_transmission implementation
            L = -eta * V + (eta * VdotH - sqrt(k)) * H;
            L = normalize(L);
            
            // Calculate transmission PDF using microfacet distribution
            float NdotH = max(0.001, dot(normal, H));
            float LdotH = abs(dot(L, H));
            
            float alpha2 = alpha * alpha;
            float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (M_PI * denom * denom);
            
            // Jacobian for transmission - must match the BSDF implementation
            float denom_jacobian = VdotH + LdotH / eta;
            float jacobian = (eta * eta * LdotH) / (denom_jacobian * denom_jacobian);
            pdf += transmission_weight * D * NdotH * jacobian;
        }
    }
        else {
        // Sample diffuse (cosine-weighted hemisphere)
        float cosine_pdf;
        L = sample_cosine_hemisphere_concentric(random_float2(seed), cosine_pdf);
        L = sf.fromLocal(L);
        
        // Add diffuse PDF contribution
        pdf += diffuse_weight * cosine_pdf;
    }
    
    // Now accumulate PDFs for all other components that could have generated this direction
    float NdotL = max(0.0, dot(normal, L));
    
    if (NdotL > 0.0) {
        // Add diffuse PDF if we didn't sample it
        if (r >= metal_weight + specular_weight + transmission_weight) {
            // We already added diffuse PDF above
        } else {
            pdf += diffuse_weight * NdotL / M_PI;
        }
        
        // Add specular reflection PDF contributions
        float3 H_refl = normalize(V + L);
        float VdotH_refl = max(0.001, dot(V, H_refl));
        float NdotH_refl = max(0.001, dot(normal, H_refl));
        
        // Metal reflection PDF
        if (r >= metal_weight) {
            float2 roughness_vector;
            mx_roughness_anisotropy(specular_roughness, specular_anisotropy, roughness_vector);
            float alpha = max(0.001, roughness_vector.x * roughness_vector.x);
            
            float alpha2 = alpha * alpha;
            float denom = NdotH_refl * NdotH_refl * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (M_PI * denom * denom);
            pdf += metal_weight * D * NdotH_refl / (4.0 * VdotH_refl);
        }
        
        // Specular reflection PDF
        if (r < metal_weight || r >= metal_weight + specular_weight) {
            float2 roughness_vector;
            mx_roughness_anisotropy(specular_roughness, specular_anisotropy, roughness_vector);
            float alpha = max(0.001, roughness_vector.x * roughness_vector.x);
            
            float alpha2 = alpha * alpha;
            float denom = NdotH_refl * NdotH_refl * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (M_PI * denom * denom);
            pdf += specular_weight * D * NdotH_refl / (4.0 * VdotH_refl);
        }
    }
    
    // Clamp PDF to avoid numerical issues
    pdf = max(pdf, 0.000001);
    
    return L;
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

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        // Find any closure contexts used by this node
        // and emit the function for each context.
        vector<ClosureContext*> ccts;
        shadergen.getClosureContexts(node, ccts);
        if (ccts.empty()) {
            emitFunctionDefinition(nullptr, context, stage);
        }
        else {
            for (ClosureContext* cct : ccts) {
                emitFunctionDefinition(cct, context, stage);
            }
        }  // Emit the sample fallback and standard surface sampling
        shadergen.emitLine(sample_source_code_fallback, stage, false);
        shadergen.emitLine(sample_source_code_standard_surface, stage, false);
    }
}

void ClosureCompoundNodeSlang::emitFunctionDefinition(
    ClosureContext* cct,
    GenContext& context,
    ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    const Syntax& syntax = shadergen.getSyntax();

    string delim = "";

    // Begin function signature
    shadergen.emitLineBegin(stage);
    if (cct) {
        // Use the first output for classifying node type for the closure
        // context. This is only relevent for closures, and they only have a
        // single output.
        const TypeDesc* closureType = _rootGraph->getOutputSocket()->getType();

        shadergen.emitString(
            "void " + _functionName + cct->getSuffix(closureType) + "(", stage);

        // Add any extra argument inputs first
        for (const ClosureContext::Argument& arg :
             cct->getArguments(closureType)) {
            const string& type = syntax.getTypeName(arg.first);
            shadergen.emitString(delim + type + " " + arg.second, stage);
            delim = ", ";
        }
    }
    else {
        shadergen.emitString("void " + _functionName + "(", stage);
    }

    auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
    if (!vertexData.empty()) {
        shadergen.emitString(
            delim + vertexData.getName() + " " + vertexData.getInstance(),
            stage);
        delim = ", ";
    }

    shadergen.emitString(delim + "SamplerState sampler", stage);

    const string& type = syntax.getTypeName(Type::VECTOR3);
    shadergen.emitString(delim + type + " " + HW::DIR_L, stage);
    shadergen.emitString(", " + type + " " + HW::DIR_V, stage);
    // eta
    shadergen.emitString(delim + "uint eta_flipped", stage);

    // Add all inputs
    for (ShaderGraphInputSocket* inputSocket : _rootGraph->getInputSockets()) {
        shadergen.emitString(
            delim + syntax.getTypeName(inputSocket->getType()) + " " +
                inputSocket->getVariable(),
            stage);
        delim = ", ";
    }

    // Add all outputs
    for (ShaderGraphOutputSocket* outputSocket :
         _rootGraph->getOutputSockets()) {
        shadergen.emitString(
            delim + syntax.getOutputTypeName(outputSocket->getType()) + " " +
                outputSocket->getVariable(),
            stage);
        delim = ", ";
    }

    // End function signature
    shadergen.emitString(")", stage);
    shadergen.emitLineEnd(stage, false);

    // Begin function body
    shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);

    if (cct) {
        context.pushClosureContext(cct);
    }

    // Emit all texturing nodes. These are inputs to the
    // closure nodes and need to be emitted first.
    shadergen.emitFunctionCalls(
        *_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

    // Emit function calls for internal closures nodes connected to the graph
    // sockets. These will in turn emit function calls for any dependent closure
    // nodes upstream.
    for (ShaderGraphOutputSocket* outputSocket :
         _rootGraph->getOutputSockets()) {
        if (outputSocket->getConnection()) {
            const ShaderNode* upstream =
                outputSocket->getConnection()->getNode();
            if (upstream->getParent() == _rootGraph.get() &&
                (upstream->hasClassification(
                     ShaderNode::Classification::CLOSURE) ||
                 upstream->hasClassification(
                     ShaderNode::Classification::SHADER))) {
                shadergen.emitFunctionCall(*upstream, context, stage);
            }
        }
    }

    if (cct) {
        context.popClosureContext();
    }

    // Emit final results
    for (ShaderGraphOutputSocket* outputSocket :
         _rootGraph->getOutputSockets()) {
        const string result =
            shadergen.getUpstreamResult(outputSocket, context);
        shadergen.emitLine(outputSocket->getVariable() + " = " + result, stage);
    }

    // End function body
    shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);
}

void ClosureCompoundNodeSlang::emitFunctionCall(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    USTC_CG::log::info(
        "Emitting closure compound function call for node: " + node.getName());

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

        // Check if we have a closure context to modify the function call.
        ClosureContext* cct = context.getClosureContext();
        if (cct) {
            // Use the first output for classifying node type for the closure
            // context. This is only relevent for closures, and they only have a
            // single output.
            const ShaderGraphOutputSocket* outputSocket =
                _rootGraph->getOutputSocket();
            const TypeDesc* closureType = outputSocket->getType();

            // Check if extra parameters has been added for this node.
            const ClosureContext::ClosureParams* params =
                cct->getClosureParams(&node);
            if (*closureType == *Type::BSDF && params) {
                // Assign the parameters to the BSDF.
                for (auto it : *params) {
                    shadergen.emitLine(
                        outputSocket->getVariable() + "." + it.first + " = " +
                            shadergen.getUpstreamResult(it.second, context),
                        stage);
                }
            }

            // Emit function name.
            shadergen.emitString(
                _functionName + cct->getSuffix(closureType) + "(", stage);

            // Emit extra argument.
            for (const ClosureContext::Argument& arg :
                 cct->getArguments(closureType)) {
                shadergen.emitString(delim + arg.second, stage);
                delim = ", ";
            }
        }
        else {
            // Emit function name.
            shadergen.emitString(_functionName + "(", stage);
        }

        auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty()) {
            shadergen.emitString(delim + vertexData.getInstance(), stage);
            delim = ", ";
        }

        shadergen.emitString(delim + "sampler", stage);

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
        }  // End function call
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);

        // Check if this is a standard surface material
        bool isStandardSurface =
            _functionName.find("standard_surface") != string::npos;
        if (isStandardSurface) {
            // Call the standard surface sampling function
            shadergen.emitLineBegin(stage);
            shadergen.emitString(
                "sampled_direction = sample_standard_surface(vd, sampler, V, ",
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
        else {
            // Call the sample fallback, and eval the sampled direction
            shadergen.emitLine(
                "sampled_direction = sample_fallback(seed, pdf, data, "
                "vertexInfo)",
                stage);
        }
        {
            // Use that direction to replace HW::DIR_L and re-evaluate the
            // material
            shadergen.emitLine(
                "surfaceshader sampled_weight_out = "
                "surfaceshader(float3(0.0),float3(0.0));",
                stage);
            shadergen.emitLineBegin(stage);
            string delim = "";

            // Check if we have a closure context to modify the function call.
            ClosureContext* cct = context.getClosureContext();
            if (cct) {
                // Use the first output for classifying node type for the
                // closure context. This is only relevent for closures, and they
                // only have a single output.
                const ShaderGraphOutputSocket* outputSocket =
                    _rootGraph->getOutputSocket();
                const TypeDesc* closureType = outputSocket->getType();

                // Check if extra parameters has been added for this node.
                const ClosureContext::ClosureParams* params =
                    cct->getClosureParams(&node);
                if (*closureType == *Type::BSDF && params) {
                    // Assign the parameters to the BSDF.
                    for (auto it : *params) {
                        shadergen.emitLine(
                            outputSocket->getVariable() + "." + it.first +
                                " = " +
                                shadergen.getUpstreamResult(it.second, context),
                            stage);
                    }
                }

                // Emit function name.
                shadergen.emitString(
                    _functionName + cct->getSuffix(closureType) + "(", stage);

                // Emit extra argument.
                for (const ClosureContext::Argument& arg :
                     cct->getArguments(closureType)) {
                    shadergen.emitString(delim + arg.second, stage);
                    delim = ", ";
                }
            }
            else {
                // Emit function name.
                shadergen.emitString(_functionName + "(", stage);
            }

            auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
            if (!vertexData.empty()) {
                shadergen.emitString(delim + vertexData.getInstance(), stage);
                delim = ", ";
            }

            shadergen.emitString(delim + "sampler", stage);

            shadergen.emitString(
                delim + "sampled_direction" + ", " + HW::DIR_V, stage);

            shadergen.emitString(delim + "eta_flipped", stage);

            // Emit all inputs.
            for (ShaderInput* input : node.getInputs()) {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }

            // Emit all outputs.
            shadergen.emitString(delim, stage);
            shadergen.emitString("sampled_weight_out", stage);
            delim = ", ";

            // End function call
            shadergen.emitString(")", stage);
            shadergen.emitLineEnd(stage);
            shadergen.emitLine(
                "sampled_weight = sampled_weight_out.color", stage);
        }
    }
}

MATERIALX_NAMESPACE_END
