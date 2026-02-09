#include "bindlessContext.h"

#include <MaterialXGenShader/TypeDesc.h>
#include <pxr/imaging/hio/image.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "MaterialX/SlangShaderGenerator.h"
#include "MaterialXGenShader/Shader.h"
#include "RHI/Hgi/format_conversion.hpp"
#include "hdMtlxFast.h"
#include "material.h"
#include "materialFilter.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/usd/ar/resolver.h"
RUZINO_NAMESPACE_OPEN_SCOPE
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
                unsigned int startLocation =
                    data_location;  // Record start location for mapping

                if (type == Type::FLOAT) {
                    auto val = uniform->getValue()->asA<float>();

                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(float));

                    dataFetch = "asfloat(data.data[" +
                                std::to_string(data_location++) + "])";

                    numComponents = 1;
                    // Record parameter mapping
                    record_parameter_mapping(
                        uniform->getVariable(),
                        startLocation,
                        type,
                        sizeof(float));
                }
                else if (
                    type == Type::INTEGER || type == Type::STRING ||
                    type == Type::BOOLEAN) {
                    if (type == Type::INTEGER) {
                        int val = 0;
                        try {
                            val = uniform->getValue()->asA<int>();
                        }
                        catch (const std::exception& e) {
                            // Try long as fallback (it's also mapped to
                            // "integer")
                            try {
                                val = static_cast<int>(
                                    uniform->getValue()->asA<long>());
                                spdlog::info(
                                    "Uniform '{}' is long type, converted to "
                                    "int: {}",
                                    uniform->getVariable(),
                                    val);
                            }
                            catch (const std::exception& e2) {
                                spdlog::warn(
                                    "Failed to convert INTEGER type for '{}': "
                                    "{}. Using default value 0. Inner error: "
                                    "{}",
                                    uniform->getVariable(),
                                    e.what(),
                                    e2.what());
                                val = 0;
                            }
                        }

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(int));
                    }
                    else if (type == Type::BOOLEAN) {
                        auto val = uniform->getValue()->asA<bool>();
                        int intVal = val ? 1 : 0;

                        memcpy(
                            &material_data.data[data_location],
                            &intVal,
                            sizeof(int));
                    }
                    dataFetch = "asint(data.data[" +
                                std::to_string(data_location++) + "])";
                    numComponents = 1;
                    // Record parameter mapping for INTEGER/BOOLEAN
                    record_parameter_mapping(
                        uniform->getVariable(),
                        startLocation,
                        type,
                        sizeof(int));
                }
                else if (type == Type::VECTOR2) {
                    auto val = uniform->getValue()->asA<Vector2>();
                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(Vector2));

                    dataFetch = "float2(asfloat(data.data[" +
                                std::to_string(data_location) +
                                "]), asfloat(data.data[" +
                                std::to_string(data_location + 1) + "]))";

                    data_location += 2;
                    numComponents = 2;
                    // Record parameter mapping for VECTOR2
                    record_parameter_mapping(
                        uniform->getVariable(),
                        startLocation,
                        type,
                        sizeof(Vector2));
                }
                else if (type == Type::VECTOR3 || type == Type::COLOR3) {
                    if (type == Type::COLOR3) {
                        auto val = uniform->getValue()->asA<Color3>();

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Color3));
                    }
                    else {
                        auto val = uniform->getValue()->asA<Vector3>();

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
                    // Record parameter mapping for VECTOR3/COLOR3
                    record_parameter_mapping(
                        uniform->getVariable(),
                        startLocation,
                        type,
                        sizeof(Vector3));
                }
                else if (type == Type::COLOR4 || type == Type::VECTOR4) {
                    // Check if uniform has a value
                    if (!uniform->getValue()) {
                        // No value provided, use default zero
                        Vector4 val(0.0f, 0.0f, 0.0f, 0.0f);

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Vector4));
                    }
                    else if (uniform->getValue()->isA<Color4>()) {
                        auto val = uniform->getValue()->asA<Color4>();

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Color4));
                    }
                    else if (uniform->getValue()->isA<Vector4>()) {
                        auto val = uniform->getValue()->asA<Vector4>();

                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Vector4));
                    }
                    else {
                        spdlog::warn(
                            "Unsupported uniform value type for {}: {}",
                            uniform->getVariable(),
                            type.getName());
                        // Use default zero value instead of asserting
                        Vector4 val(0.0f, 0.0f, 0.0f, 0.0f);
                        memcpy(
                            &material_data.data[data_location],
                            &val,
                            sizeof(Vector4));
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
                    // Record parameter mapping for VECTOR4/COLOR4
                    record_parameter_mapping(
                        uniform->getVariable(),
                        startLocation,
                        type,
                        sizeof(Vector4));
                }
                else if (type == Type::MATRIX44) {
                    auto val = uniform->getValue()->asA<Matrix44>();
                    memcpy(
                        &material_data.data[data_location],
                        &val,
                        sizeof(Matrix44));
                    unsigned int matrix_start = data_location;
                    dataFetch = "float4x4(";
                    for (int i = 0; i < 16; i++) {
                        dataFetch += "asfloat(data.data[" +
                                     std::to_string(data_location++) + "])";
                        if (i < 15)
                            dataFetch += ", ";
                    }
                    dataFetch += ")";
                    numComponents = 16;
                    // Record parameter mapping for MATRIX44
                    record_parameter_mapping(
                        uniform->getVariable(),
                        matrix_start,
                        type,
                        sizeof(Matrix44));
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
                else if (type == Type::SURFACESHADER) {
                    auto val = uniform->getValue();
                    // Load vector3 and float for surface shader
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
                    dataFetch =
                        "surfaceshader(" + vectorPart + ", " + floatPart + ")";
                    data_location += 4;
                    numComponents = 4;
                }
                else {
                    spdlog::warn(("Unsupported uniform type: " + type.getName())
                                     .c_str());
                }

                if (uniform->getVariable() == "Surface_opacityThreshold") {
                    spdlog::info(
                        "Surface_opacityThreshold: {}, value: {}",
                        dataFetch,
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
                // Reserve a location in data buffer for texture ID
                unsigned int texture_id_location = data_location++;

                // Store the mapping using uniform's name WITHOUT the "_file"
                // suffix to match the MaterialX document node names
                std::string uniformName = uniform->getName();
                // Remove "_file" suffix if present
                const std::string fileSuffix = "_file";
                if (uniformName.size() >= fileSuffix.size() &&
                    uniformName.compare(
                        uniformName.size() - fileSuffix.size(),
                        fileSuffix.size(),
                        fileSuffix) == 0) {
                    uniformName = uniformName.substr(
                        0, uniformName.size() - fileSuffix.size());
                }

                texture_id_locations[uniformName] = texture_id_location;

                spdlog::info(
                    "BindlessContext: Registering texture '{}' (variable: "
                    "'{}') at data location {}",
                    uniformName,
                    uniform->getVariable(),
                    texture_id_location);

                // Initialize with 0 (will be updated when texture is loaded)
                unsigned int default_id = 0;
                memcpy(
                    &material_data.data[texture_id_location],
                    &default_id,
                    sizeof(unsigned int));

                // Generate code to read texture ID from data buffer
                fetch_data += "Texture2D " + uniform->getVariable() + " = " +
                              " t_BindlessTextures[NonUniformResourceIndex(" +
                              "asuint(data.data[" +
                              std::to_string(texture_id_location) + "]))];\n";
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
            else if (vertexdata_member->getName() == HW::T_BITANGENT_WORLD) {
                fetch_data +=
                    "vd." + vertexdata_member->getName() +
                    " = cross(vertexInfo.normalW, vertexInfo.tangentW.xyz) * "
                    "vertexInfo.tangentW.w;\n";
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

RUZINO_NAMESPACE_CLOSE_SCOPE
