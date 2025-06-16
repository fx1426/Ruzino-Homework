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
import pbrlib.genslang.lib.mx_microfacet;
import pbrlib.genslang.lib.mx_microfacet_specular;
#include "utils/Math/MathConstants.slangh"
#include "utils/random.slangh"

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
    // Create shading frame
    bool valid;
    ShadingFrame sf = ShadingFrame.createSafe(normal, float4(tangent, 1.0), valid);
    
    // Transform view direction to local space
    float3 V_local = sf.toLocal(V);
    float NdotV = clamp(V_local.z, M_FLOAT_EPS, 1.0);
    
    // Calculate material properties
    float3 actualBaseColor = base_color * base;
    float actualMetalness = clamp(metalness, 0.0, 1.0);
    float actualSpecular = clamp(specular, 0.0, 1.0);
    float actualSpecularRoughness = clamp(specular_roughness, M_FLOAT_EPS, 1.0);
    
    // Convert roughness and anisotropy to alpha values
    float2 alpha;
    mx_roughness_anisotropy(actualSpecularRoughness, specular_anisotropy, alpha);
    
    // Calculate Fresnel at normal incidence
    float F0_dielectric = mx_ior_to_f0(specular_IOR);
    float3 F0 = lerp(float3(F0_dielectric * actualSpecular), actualBaseColor, actualMetalness);
    
    // Sample random values
    float2 Xi = random_float2(seed);
    float xi_lobe = random_float(seed);
    
    // Calculate lobe weights (simplified - ignoring coat and sheen for now)
    float diffuse_weight = (1.0 - actualMetalness) * base;
    float specular_weight = 1.0;
    float total_weight = diffuse_weight + specular_weight;
    
    // Normalize weights
    if (total_weight > M_FLOAT_EPS) {
        diffuse_weight /= total_weight;
        specular_weight /= total_weight;
    } else {
        // Fallback to uniform hemisphere sampling
        float3 localDir = mx_uniform_sample_hemisphere(Xi);
        float3 sampledDir = sf.fromLocal(localDir);
        pdf = 1.0 / (2.0 * M_PI);
        return sampledDir;
    }
      // Choose which lobe to sample and calculate direction
    float3 L_local;
    float lobe_pdf = 0.0;
    
    if (xi_lobe < diffuse_weight) {
        // Sample diffuse lobe (cosine-weighted hemisphere)
        float cosTheta = sqrt(Xi.x);
        float sinTheta = sqrt(1.0 - Xi.x);
        float phi = 2.0 * M_PI * Xi.y;
        
        L_local = float3(
            sinTheta * cos(phi),
            sinTheta * sin(phi),
            cosTheta
        );
        
        lobe_pdf = cosTheta * M_PI_INV;
    } else {
        // Sample specular lobe using VNDF (Visible Normal Distribution Function)
        float3 H_local = mx_ggx_importance_sample_VNDF(Xi, V_local, alpha);
        
        // Reflect view direction around the sampled microfacet normal
        L_local = reflect(-V_local, H_local);
        
        // Check if the reflected direction is above the surface
        if (L_local.z <= 0.0) {
            // Fallback to cosine-weighted hemisphere sampling
            float cosTheta = sqrt(Xi.x);
            float sinTheta = sqrt(1.0 - Xi.x);
            float phi = 2.0 * M_PI * Xi.y;
            
            L_local = float3(
                sinTheta * cos(phi),
                sinTheta * sin(phi),
                cosTheta
            );
            
            lobe_pdf = cosTheta * M_PI_INV;
        } else {
            // Calculate PDF for GGX VNDF sampling
            float NdotH = clamp(H_local.z, M_FLOAT_EPS, 1.0);
            float VdotH = clamp(dot(V_local, H_local), M_FLOAT_EPS, 1.0);
            
            float D = mx_ggx_NDF(H_local, alpha);
            float G1 = mx_ggx_smith_G1(NdotV, mx_average_alpha(alpha));
            
            // PDF for VNDF sampling = D * G1 / (4 * NdotV)
            lobe_pdf = max((D * G1) / (4.0 * NdotV), M_FLOAT_EPS);
        }
    }
    
    // Ensure PDF is never zero
    pdf = max(lobe_pdf, M_FLOAT_EPS);
    
    // Transform to world space
    float3 sampledDir = sf.fromLocal(L_local);
    
    return sampledDir;
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
