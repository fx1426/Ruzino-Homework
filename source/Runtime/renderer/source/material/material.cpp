#include "material.h"

#include <MaterialXGenShader/TypeDesc.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/usdImaging/usdImaging/tokens.h>

#include <filesystem>
#include <fstream>

#include "MaterialX/SlangShaderGenerator.h"
#include "MaterialXCore/Document.h"
#include "MaterialXFormat/Util.h"
#include "MaterialXGenShader/Shader.h"
#include "MaterialXGenShader/Util.h"
#include "RHI/Hgi/format_conversion.hpp"
#include "api.h"
#include "hdMtlxFast.h"
#include "materialFilter.h"
#include "nvrhi/nvrhi.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/base/arch/hash.h"
#include "pxr/base/arch/library.h"
#include "pxr/imaging/hd/changeTracker.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/sdr/registry.h"
#include "pxr/usd/sdr/shaderNode.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace MaterialX;
/// Shared pointer to a BindlessContext
using BindlessContextPtr = std::shared_ptr<class BindlessContext>;

/// @class BindlessContext
/// Class representing a resource binding for Slang shader resources.
class BindlessContext : public HwResourceBindingContext {
   public:
    BindlessContext(
        size_t uniformBindingLocation,
        size_t samplerBindingLocation);

    static BindlessContextPtr create(
        size_t uniformBindingLocation = 0,
        size_t samplerBindingLocation = 0)
    {
        return std::make_shared<BindlessContext>(
            uniformBindingLocation, samplerBindingLocation);
    }

    void initialize() override
    {
        fetch_data =
            "MaterialDataBlob data = materialBuffer[material_id];\n VertexData "
            "vd; \n";
        data_location = 0;
    }

    void emitDirectives(GenContext& context, ShaderStage& stage) override
    {
        const ShaderGenerator& generator = context.getShaderGenerator();
        generator.emitLine("import Scene.BindlessMaterial", stage);
        generator.emitLine("import Scene.VertexInfo", stage);
    }

    // Emit uniforms with binding information
    void emitResourceBindings(
        GenContext& context,
        const VariableBlock& resources,
        ShaderStage& stage) override;

    // Emit structured uniforms with binding information and align members where
    // possible
    void emitStructuredResourceBindings(
        GenContext& context,
        const VariableBlock& uniforms,
        ShaderStage& stage,
        const std::string& structInstanceName,
        const std::string& arraySuffix) override;

    std::string get_data_code()
    {
        return fetch_data;
    }

    MaterialDataBlob& get_material_data()
    {
        return material_data;
    }

   private:
    std::string fetch_data = "";
    unsigned int data_location = 0;
    MaterialDataBlob material_data;
};

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
                else if (type == Type::INTEGER || type == Type::STRING) {
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
                            ("Unsupported uniform type: " + type->getName())
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
                    log::warning(
                        ("Unsupported uniform type: " + type->getName())
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
            if (*uniform->getType() == *Type::FILENAME) {
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
    std::unordered_map<const TypeDesc*, size_t> alignmentMap(
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

namespace mx = MaterialX;

MaterialX::GenContextPtr Hd_USTC_CG_Material::shader_gen_context_ =
    std::make_shared<mx::GenContext>(mx::SlangShaderGenerator::create());
MaterialX::DocumentPtr Hd_USTC_CG_Material::libraries = mx::createDocument();

std::mutex Hd_USTC_CG_Material::shadergen_mutex;
std::mutex Hd_USTC_CG_Material::texture_mutex;
std::mutex Hd_USTC_CG_Material::material_data_handle_mutex;

std::once_flag Hd_USTC_CG_Material::shader_gen_initialized_;

Hd_USTC_CG_Material::Hd_USTC_CG_Material(SdfPath const& id) : HdMaterial(id)
{
    std::call_once(shader_gen_initialized_, []() {
        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
        searchPath.append(mx::FileSearchPath("usd/hd_USTC_CG/resources"));

        loadLibraries({ "libraries" }, searchPath, libraries);
        mx::loadLibraries(
            { "usd/hd_USTC_CG/resources/libraries" }, searchPath, libraries);
        shader_gen_context_->registerSourceCodeSearchPath(searchPath);

        shader_gen_context_->pushUserData(
            mx::HW::USER_DATA_BINDING_CONTEXT, BindlessContext::create());
    });
}

TF_DEFINE_PRIVATE_TOKENS(_tokens, (file)(sourceColorSpace)(raw)(srgb));
void Hd_USTC_CG_Material::CollectTextures(
    HdMaterialNetwork2Interface netInterface,
    HdMtlxTexturePrimvarData hdMtlxData)
{
    // Collect texture names and paths into a vector.
    for (const SdfPath& texturePath : hdMtlxData.hdTextureNodes) {
        TfToken textureNodeName = texturePath.GetToken();
        // Get the file parameter from the node.
        VtValue vFile =
            netInterface.GetNodeParameterValue(textureNodeName, _tokens->file);
        std::string path;
        if (vFile.IsHolding<SdfAssetPath>()) {
            path = vFile.Get<SdfAssetPath>().GetResolvedPath();
            if (path.empty()) {
                path = vFile.Get<SdfAssetPath>().GetAssetPath();
            }
        }
        else if (vFile.IsHolding<std::string>()) {
            path = vFile.Get<std::string>();
        }

        VtValue sourceColorSpace = netInterface.GetNodeParameterValue(
            textureNodeName, _tokens->sourceColorSpace);

        if (sourceColorSpace.IsHolding<std::string>()) {
            std::string colorSpace = sourceColorSpace.Get<std::string>();
            if (colorSpace == "srgb_texture") {
                texturePaths[textureNodeName.GetString()] = path;
            }
        }

        texturePaths[textureNodeName.GetString()] = path;

        // Load the texture immediately
        if (!pxr::HioImage::IsSupportedImageFile(path)) {
            TF_WARN(
                "Texture '%s': unsupported file format '%s'.",
                textureNodeName.GetString().c_str(),
                path.c_str());
            continue;
        }

        HioImageSharedPtr image = pxr::HioImage::OpenForReading(path);
        if (!image) {
            TF_WARN(
                "Texture '%s': failed to load image from file '%s'.",
                textureNodeName.GetString().c_str(),
                path.c_str());
            continue;
        }

        textureResources[textureNodeName.GetString()].filePath = path;
        textureResources[textureNodeName.GetString()].image = image;
    }
}

void Hd_USTC_CG_Material::MtlxGenerateShader(
    MaterialX::DocumentPtr mtlx_document,
    HdMaterialNetwork2Interface netInterface,
    HdMtlxTexturePrimvarData& hdMtlxData)
{
    _UpdateTextureNodes(
        &netInterface, hdMtlxData.hdTextureNodes, mtlx_document);

    auto renderable = findRenderableElements(mtlx_document);

    _FixOmittedConnections(mtlx_document, renderable);
    using namespace mx;

    auto element = renderable[0];

    std::string elementName(element->getNamePath());
    material_name = mx::createValidName(elementName);

    ShaderGenerator& shader_generator_ =
        shader_gen_context_->getShaderGenerator();
    {
        std::lock_guard lock(shadergen_mutex);
        auto shader = shader_generator_.generate(
            material_name, element, *shader_gen_context_);
        eval_shader_source = shader->getSourceCode(mx::Stage::PIXEL);

        BindlessContextPtr context =
            shader_gen_context_->getUserData<BindlessContext>(
                mx::HW::USER_DATA_BINDING_CONTEXT);
        get_data_code = context->get_data_code();

        material_data = context->get_material_data();

        material_data_handle->write_data(&material_data);
    }
}

HdMaterialNetwork2Interface Hd_USTC_CG_Material::FetchMaterialNetwork(
    HdSceneDelegate* sceneDelegate,
    HdMaterialNetwork2& hdNetwork,
    SdfPath& materialPath,
    SdfPath& surfTerminalPath,
    HdMaterialNode2 const*& surfTerminal)
{
    VtValue material = sceneDelegate->GetMaterialResource(GetId());
    HdMaterialNetworkMap networkMap = material.Get<HdMaterialNetworkMap>();

    bool isVolume;
    hdNetwork = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);

    materialPath = GetId();

    auto netInterface = HdMaterialNetwork2Interface(materialPath, &hdNetwork);
    _FixNodeTypes(&netInterface);
    _FixNodeValues(&netInterface);

    const TfToken& terminalNodeName = HdMaterialTerminalTokens->surface;

    surfTerminal =
        _GetTerminalNode(hdNetwork, terminalNodeName, &surfTerminalPath);
    return netInterface;
}
void Hd_USTC_CG_Material::BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param)
{
    auto descriptor_table =
        render_param->InstanceCollection->get_texture_descriptor_table();

    for (auto& texture_resource : textureResources) {
        // Create a thread for asynchronous processing
        std::thread texture_thread(
            [&texture_resource, this, descriptor_table, render_param]() {
                auto device = RHI::get_device();

                auto image = texture_resource.second.image;

                nvrhi::TextureDesc desc;
                desc.width = image->GetWidth();
                desc.height = image->GetHeight();
                desc.format = RHI::ConvertFromHioFormat(image->GetFormat());

                desc.initialState = nvrhi::ResourceStates::ShaderResource;
                desc.isRenderTarget = false;

                texture_resource.second.texture = device->createTexture(desc);

                auto texture_name =
                    std::filesystem::path(texture_resource.first)
                        .filename()
                        .string();

                auto storage_byte_size = image->GetBytesPerPixel();

                std::vector<uint8_t> data(
                    image->GetWidth() * image->GetHeight() * storage_byte_size,
                    0);

                HioImage::StorageSpec storageSpec;
                storageSpec.width = image->GetWidth();
                storageSpec.height = image->GetHeight();
                storageSpec.format = image->GetFormat();
                storageSpec.flipped = true;
                storageSpec.data = data.data();

                // Read the image data asynchronously
                texture_resource.second.image->Read(storageSpec);

                {
                    std::lock_guard lock(texture_mutex);
                    if (image->GetFormat() == HioFormatUNorm8Vec3srgb) {
                        // rearrange the data to be RGBA
                        std::vector<uint8_t> rgba_data(
                            image->GetWidth() * image->GetHeight() * 4, 0);
                        for (size_t i = 0; i < data.size() / 3; i++) {
                            rgba_data[i * 4] = data[i * 3];
                            rgba_data[i * 4 + 1] = data[i * 3 + 1];
                            rgba_data[i * 4 + 2] = data[i * 3 + 2];
                            rgba_data[i * 4 + 3] = 255;
                        }
                        data = std::move(rgba_data);
                    }

                    auto [gpu_texture, staging] =
                        RHI::load_texture(desc, data.data());

                    texture_resource.second.texture = gpu_texture;
                }

                texture_resource.second.descriptor =
                    descriptor_table->CreateDescriptorHandle(
                        nvrhi::BindingSetItem::Texture_SRV(
                            0, texture_resource.second.texture, desc.format));

                if (texture_resource.second.texture) {
                    auto texture_id = texture_resource.second.descriptor.Get();

                    // Replace the "$"+ texture_name+"_id" with the actual
                    // texture_id in the get_data_code string
                    std::string to_replace = "$" + texture_name + "_file_id";
                    std::string replace_with = std::to_string(texture_id);

                    size_t pos = get_data_code.find(to_replace);
                    if (pos != std::string::npos) {
                        get_data_code.replace(
                            pos, to_replace.length(), replace_with);
                    }
                }
            });

        // Add the thread to the render_param for tracking
        render_param->texture_loading_threads.push_back(
            std::move(texture_thread));
    }
}

void Hd_USTC_CG_Material::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    auto param = static_cast<Hd_USTC_CG_RenderParam*>(renderParam);

    ensure_material_data_handle(param);

    HdMaterialNetwork2 hdNetwork;
    SdfPath materialPath;

    SdfPath surfTerminalPath;
    HdMaterialNode2 const* surfTerminal;
    HdMaterialNetwork2Interface netInterface = FetchMaterialNetwork(
        sceneDelegate, hdNetwork, materialPath, surfTerminalPath, surfTerminal);

    HdMtlxTexturePrimvarData hdMtlxData;

    DocumentPtr mtlx_document = HdMtlxCreateMtlxDocumentFromHdNetworkFast(
        hdNetwork,
        *surfTerminal,
        surfTerminalPath,
        materialPath,
        libraries,
        &hdMtlxData);
    assert(mtlx_document);
    CollectTextures(netInterface, hdMtlxData);

    MtlxGenerateShader(mtlx_document, netInterface, hdMtlxData);

    BuildGPUTextures(param);

    *dirtyBits = HdChangeTracker::Clean;
}

HdDirtyBits Hd_USTC_CG_Material::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::AllDirty;
}

void Hd_USTC_CG_Material::Finalize(HdRenderParam* renderParam)
{
    HdMaterial::Finalize(renderParam);
}

std::vector<TextureHandle> Hd_USTC_CG_Material::GetTextures() const
{
    std::vector<TextureHandle> textures;
    for (const auto& tex : textureResources) {
        textures.push_back(tex.second.texture);
    }
    return textures;
}

void Hd_USTC_CG_Material::ensure_material_data_handle(
    Hd_USTC_CG_RenderParam* render_param)
{
    std::lock_guard<std::mutex> lock(material_data_handle_mutex);
    if (!material_data_handle) {
        if (!render_param) {
            throw std::runtime_error("Render param is null.");
        }
        material_data_handle =
            render_param->InstanceCollection->material_pool.allocate(1);
    }
}

unsigned Hd_USTC_CG_Material::GetMaterialLocation() const
{
    if (!material_data_handle) {
        return -1;
    }
    return material_data_handle->index();
}

// HLSL callable shader
static std::string slang_source_code = R"(
import Scene.VertexInfo;

struct CallableData
{
    float4 color;
    float3 L;
    float3 V;
    uint materialID;
    VertexInfo vertexInfo;
};

[shader("callable")]
void $getColor(inout CallableData data)
{
    eval(data.color, data.L, data.V, data.materialID, data.vertexInfo);
}

)";

// HLSL callable shader
static std::string slang_source_code_fallback = R"(

struct VertexData
{
    float foo;
};

import Scene.VertexInfo;
// ConstantBuffer<float> cb;

struct CallableData
{
    float4 color;
    float3 L;
    float3 V;
    uint materialID;
    VertexInfo vertexInfo;
};

[shader("callable")]
void $getColor(inout CallableData data)
{
    data.color = float4(0, 1, 0, 1);
}

)";
void Hd_USTC_CG_Material::ensure_shader_ready(const ShaderFactory& factory)
{
    if (shader_ready) {
        return;
    }

    if (!eval_shader_source.empty()) {
        // Replace the data loading placeholder with actual data code
        constexpr char DATA_PLACEHOLDER[] = "$BindlessDataLoading";
        size_t pos = eval_shader_source.find(DATA_PLACEHOLDER);
        if (pos != std::string::npos) {
            eval_shader_source.replace(
                pos, strlen(DATA_PLACEHOLDER), get_data_code);
        }

        // #ifndef NDEBUG
        try {
            std::filesystem::create_directories("generated_shaders");
            std::ofstream out("generated_shaders/" + material_name + ".slang");
            if (out.is_open()) {
                out << eval_shader_source;
                out.close();
            }
        }
        catch (const std::exception& e) {
            TF_WARN("Failed to save generated shader: %s", e.what());
        }
        // #endif
        final_shader_source = eval_shader_source + slang_source_code;
    }
    else {
        // Use fallback shader if no source is available
        if (material_name.empty()) {
            material_name = "fallback";
        }
        final_shader_source = slang_source_code_fallback;
    }

    // Replace the callable function name with the material name in all code
    constexpr char FUNC_PLACEHOLDER[] = "$getColor";

    // Replace in local_slang_source_code
    auto pos = final_shader_source.find(FUNC_PLACEHOLDER);
    if (pos != std::string::npos) {
        final_shader_source.replace(
            pos, strlen(FUNC_PLACEHOLDER), material_name);
    }

    // Combine shader parts into final source

    //shader = factory.compile_shader()

    shader_ready = true;
}

std::string Hd_USTC_CG_Material::GetShader(const ShaderFactory& factory)
{
    ensure_shader_ready(factory);
    return final_shader_source;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE