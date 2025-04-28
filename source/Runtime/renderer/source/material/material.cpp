#include "material.h"

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
#include "bindlessContext.h"
#include "hdMtlxFast.h"
#include "materialFilter.h"
#include "nvrhi/nvrhi.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/imaging/hd/changeTracker.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/sdr/registry.h"
USTC_CG_NAMESPACE_OPEN_SCOPE
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
                desc.keepInitialState = true;

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

    ProgramDesc program_desc;
    program_desc.add_source_code(final_shader_source);
    program_desc.set_shader_type(nvrhi::ShaderType::Callable);
    program_desc.set_entry_name(material_name);

    final_program = factory.createProgram(program_desc);
    assert(final_program);

    shader_ready = true;
}

std::string Hd_USTC_CG_Material::GetShader(const ShaderFactory& factory)
{
    ensure_shader_ready(factory);
    return final_shader_source;
}

ProgramHandle Hd_USTC_CG_Material::GetProgram(const ShaderFactory& factory)
{
    ensure_shader_ready(factory);
    return final_program;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
