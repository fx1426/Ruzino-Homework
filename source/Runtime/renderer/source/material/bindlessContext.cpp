#include "bindlessContext.h"

#include <MaterialXGenShader/TypeDesc.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hio/image.h>

#include <filesystem>

#include "MaterialX/SlangShaderGenerator.h"
#include "MaterialXGenShader/Shader.h"
#include "RHI/Hgi/format_conversion.hpp"
#include "hdMtlxFast.h"
#include "material.h"
#include "materialFilter.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/usd/ar/resolver.h"
USTC_CG_NAMESPACE_OPEN_SCOPE
//
// BindlessContext
//
BindlessContext::BindlessContext(
    size_t uniformBindingLocation,
    size_t samplerBindingLocation)
{
}

void BindlessContext::emitResourceBindings(
    GenContext& context,
    const VariableBlock& resources,
    ShaderStage& stage)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // First, emit all value uniforms in a block with single layout binding
    bool hasValueUniforms = false;
    for (auto uniform : resources.getVariableOrder()) {
        if (uniform->getType() != Type::FILENAME) {
            hasValueUniforms = true;
            break;
        }
    }
    if (hasValueUniforms && resources.getName() == HW::PUBLIC_UNIFORMS) {
        for (auto uniform : resources.getVariableOrder()) {
            auto type = uniform->getType();

            auto& syntax = generator.getSyntax();

            if (type != Type::FILENAME) {
                std::string dataFetch;
                size_t numComponents = 0;

                if (type == Type::FLOAT) {
                    auto val = uniform->getValue()->asA<float>();

                    log ::info(
                        "setting %s to %f",
                        uniform->getVariable().c_str(),
                        val);

                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(float));

                    dataFetch = "asfloat(data.data[" +
                                std::to_string(data_location++) + "])";

                    numComponents = 1;
                }
                else if (
                    type == Type::INTEGER || type == Type::STRING ||
                    type == Type::BOOLEAN) {
                    if (type == Type::INTEGER) {
                        auto val = uniform->getValue()->asA<int>();

                        log ::info(
                            "setting %s to %d",
                            uniform->getVariable().c_str(),
                            val);

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(int));
                    }
                    else if (type == Type::BOOLEAN) {
                        auto val = uniform->getValue()->asA<bool>();
                        int intVal = val ? 1 : 0;

                        log ::info(
                            "setting %s to %d",
                            uniform->getVariable().c_str(),
                            intVal);

                        memcpy(
                            &material_data.data[data_location],
                            &intVal,
                            sizeof(int));
                    }
                    dataFetch = "asint(data.data[" +
                                std::to_string(data_location++) + "])";
                    numComponents = 1;
                }
                else if (type == Type::VECTOR2) {
                    auto val = uniform->getValue()->asA<Vector2>();
                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(Vector2));

                    log::info(
                        "setting %s to %f, %f",
                        uniform->getVariable().c_str(),
                        val[0],
                        val[1]);

                    dataFetch = "float2(asfloat(data.data[" +
                                std::to_string(data_location) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 1) + "]))";

                    data_location += 2;
                    numComponents = 2;
                }
                else if (type == Type::VECTOR3 || type == Type::COLOR3) {
                    if (type == Type::COLOR3) {
                        auto val = uniform->getValue()->asA<Color3>();

                        log::info(
                            "setting %s to %f, %f, %f",
                            uniform->getVariable().c_str(),
                            val[0],
                            val[1],
                            val[2]);
                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Color3));
                    }
                    else {
                        auto val = uniform->getValue()->asA<Vector3>();

                        log ::info(
                            "setting %s to %f, %f, %f",
                            uniform->getVariable().c_str(),
                            val[0],
                            val[1],
                            val[2]);

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Vector3));
                    }

                    dataFetch = "float3(asfloat(data.data[" +
                                std::to_string(data_location) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 1) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 2) + "]))";
                    data_location += 3;
                    numComponents = 3;
                }
                else if (type == Type::COLOR4) {
                    if (uniform->getValue()->isA<Color4>()) {
                        auto val = uniform->getValue()->asA<Color4>();

                        log ::info(
                            "setting %s to %f, %f, %f, %f",
                            uniform->getVariable().c_str(),
                            val[0],
                            val[1],
                            val[2],
                            val[3]);

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Color4));
                    }
                    else if (uniform->getValue()->isA<Vector4>()) {
                        auto val = uniform->getValue()->asA<Vector4>();

                        log ::info(
                            "setting %s to %f, %f, %f, %f",
                            uniform->getVariable().c_str(),
                            val[0],
                            val[1],
                            val[2],
                            val[3]);

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Vector4));
                    }
                    else {
                        log::warning(
                            ("Unsupported uniform type: " + type.getName())
                                .c_str());
                        assert(false);
                    }
                    dataFetch = "float4(asfloat(data.data[" +
                                std::to_string(data_location) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 1) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 2) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 3) + "]))";
                    data_location += 4;
                    numComponents = 4;
                }
                else if (type == Type::MATRIX44) {
                    auto val = uniform->getValue()->asA<Matrix44>();
                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(Matrix44));
                    dataFetch = "float4x4(";
                    for (int i = 0; i < 16; i++) {
                        dataFetch += "asfloat(data.data[" +
                                     std::to_string(data_location++) + "])";
                        if (i < 15)
                            dataFetch += ", ";
                    }
                    dataFetch += ")";
                    numComponents = 16;
                }
                else if (type == Type::DISPLACEMENTSHADER) {
                    auto val = uniform->getValue();
                    // Load vector3 and float for displacement shader
                    std::string vectorPart = "float3(asfloat(data.data[" +
                                             std::to_string(data_location) +
                                             "]), asfloat(data.data[" +
                                             std::to_string(data_location + 1) +
                                             "]), asfloat(data.data[" +
                                             std::to_string(data_location + 2) +
                                             "]))";
                    std::string floatPart = "asfloat(data.data[" +
                                            std::to_string(data_location + 3) +
                                            "])";
                    dataFetch = "displacementshader(" + vectorPart + ", " +
                                floatPart + ")";
                    data_location += 4;
                    numComponents = 4;
                }
                else {
                    log::warning(("Unsupported uniform type: " + type.getName())
                                     .c_str());
                }

                if (uniform->getVariable() == "Surface_opacityThreshold") {
                    log::info(
                        "Surface_opacityThreshold: %s, value: %f",
                        dataFetch.c_str(),
                        uniform->getValue()->asA<float>());
                }

                if (numComponents > 0) {
                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getVariable() + " = " + dataFetch +
                                  ";\n";
                }
            }

            // generator.emitLineBegin(stage);
            // generator.emitVariableDeclaration(
            //     uniform, EMPTY_STRING, context, stage, true);
            // generator.emitString(Syntax::SEMICOLON, stage);
            // generator.emitLineEnd(stage, false);
        }

        // Second, emit all sampler uniforms as separate uniforms with separate
        // layout bindings
        for (auto uniform : resources.getVariableOrder()) {
            if (uniform->getType() == Type::FILENAME) {
                // generator.emitString(
                //     "layout (binding=" +
                //         std::to_string(
                //             _separateBindingLocation ?
                //             _hwUniformBindLocation++
                //                                      :
                //                                      _hwSamplerBindLocation++)
                //                                      +
                //         ") " + syntax.getUniformQualifier() + " ",
                //     stage);
                // generator.emitVariableDeclaration(
                //    uniform, EMPTY_STRING, context, stage, true);
                // generator.emitLineEnd(stage, true);

                fetch_data += "Texture2D " + uniform->getVariable() + " = " +
                              " t_BindlessTextures[$" + uniform->getName() +
                              "_id];\n";
            }
        }
    }

    if (resources.getName() == HW::VERTEX_DATA) {
        for (auto vertexdata_member : resources.getVariableOrder()) {
            if (vertexdata_member->getName() == HW::T_POSITION_WORLD) {
                fetch_data += "vd." + vertexdata_member->getName() +
                              " = vertexInfo.posW;\n";
            }
            else if (vertexdata_member->getName() == HW::T_NORMAL_WORLD) {
                fetch_data += "vd." + vertexdata_member->getName() +
                              " = vertexInfo.normalW;\n";
            }
            else if (vertexdata_member->getName() == HW::T_TANGENT_WORLD) {
                fetch_data +=
                    "vd." + vertexdata_member->getName() +
                    " = vertexInfo.tangentW.xyz * vertexInfo.tangentW.w;\n";
            }
            else {
                if (vertexdata_member->getType() == Type::VECTOR2) {
                    fetch_data += "vd." + vertexdata_member->getName() +
                                  " = vertexInfo.texC;\n";
                }
            }
        }
    }

    fetch_data =
        mx::replaceSubstrings(fetch_data, generator.getTokenSubstitutions());

    generator.emitLineBreak(stage);
}

void BindlessContext::emitStructuredResourceBindings(
    GenContext& context,
    const VariableBlock& uniforms,
    ShaderStage& stage,
    const std::string& structInstanceName,
    const std::string& arraySuffix)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // Slang structures need to be aligned. We make a best effort to base align
    // struct members and add padding if required.
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt

    const size_t baseAlignment = 16;
    std::unordered_map<TypeDesc, size_t, TypeDesc::Hasher> alignmentMap(
        { { Type::FLOAT, baseAlignment / 4 },
          { Type::INTEGER, baseAlignment / 4 },
          { Type::BOOLEAN, baseAlignment / 4 },
          { Type::COLOR3, baseAlignment },
          { Type::COLOR4, baseAlignment },
          { Type::VECTOR2, baseAlignment },
          { Type::VECTOR3, baseAlignment },
          { Type::VECTOR4, baseAlignment },
          { Type::MATRIX33, baseAlignment * 4 },
          { Type::MATRIX44, baseAlignment * 4 } });

    // Get struct alignment and size
    // alignment, uniform member index
    vector<std::pair<size_t, size_t>> memberOrder;
    size_t structSize = 0;
    for (size_t i = 0; i < uniforms.size(); ++i) {
        auto it = alignmentMap.find(uniforms[i]->getType());
        if (it == alignmentMap.end()) {
            structSize += baseAlignment;
            memberOrder.push_back(std::make_pair(baseAlignment, i));
        }
        else {
            structSize += it->second;
            memberOrder.push_back(std::make_pair(it->second, i));
        }
    }

    // Align up and determine number of padding floats to add
    const size_t numPaddingfloats =
        (((structSize + (baseAlignment - 1)) & ~(baseAlignment - 1)) -
         structSize) /
        4;

    // Sort order from largest to smallest
    std::sort(
        memberOrder.begin(),
        memberOrder.end(),
        [](const std::pair<size_t, size_t>& a,
           const std::pair<size_t, size_t>& b) { return a.first > b.first; });

    // Emit the struct
    generator.emitLine("struct " + uniforms.getName(), stage, false);
    generator.emitScopeBegin(stage);

    for (size_t i = 0; i < uniforms.size(); ++i) {
        size_t variableIndex = memberOrder[i].second;
        generator.emitLineBegin(stage);
        generator.emitVariableDeclaration(
            uniforms[variableIndex], EMPTY_STRING, context, stage, false);
        generator.emitString(Syntax::SEMICOLON, stage);
        generator.emitLineEnd(stage, false);
    }

    // Emit padding
    for (size_t i = 0; i < numPaddingfloats; ++i) {
        generator.emitLine("float pad" + std::to_string(i), stage, true);
    }
    generator.emitScopeEnd(stage, true);

    // Emit binding information
    generator.emitLineBreak(stage);
    generator.emitLine(
        syntax.getUniformQualifier() + " " + uniforms.getName() + "_" +
            stage.getName(),
        stage,
        false);
    generator.emitScopeBegin(stage);
    generator.emitLine(
        uniforms.getName() + " " + structInstanceName + arraySuffix, stage);
    generator.emitScopeEnd(stage, true);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
