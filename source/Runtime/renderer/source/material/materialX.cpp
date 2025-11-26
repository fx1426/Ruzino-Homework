#include "materialX.h"

#include <pxr/imaging/hdMtlx/hdMtlx.h>

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
USTC_CG_NAMESPACE_OPEN_SCOPE

TF_DEFINE_PRIVATE_TOKENS(_tokens, (file)(sourceColorSpace)(raw)(srgb));

namespace mx = MaterialX;

MaterialX::GenContextPtr Hd_USTC_CG_MaterialX::shader_gen_context_ =
    std::make_shared<mx::GenContext>(mx::SlangShaderGenerator::create());
MaterialX::DocumentPtr Hd_USTC_CG_MaterialX::libraries = mx::createDocument();

std::mutex Hd_USTC_CG_MaterialX::shadergen_mutex;
std::once_flag Hd_USTC_CG_MaterialX::shader_gen_initialized_;

Hd_USTC_CG_MaterialX::Hd_USTC_CG_MaterialX(SdfPath const& id)
    : Hd_USTC_CG_Material(id)
{
    std::call_once(shader_gen_initialized_, []() {
        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
        
        // Add current working directory to search path for libraries
        searchPath.append(mx::FilePath(std::filesystem::current_path().string()));
        
        searchPath.append(mx::FileSearchPath("usd/hd_USTC_CG/resources"));

        loadLibraries({ "libraries" }, searchPath, libraries);
        mx::loadLibraries(
            { "usd/hd_USTC_CG/resources/libraries" }, searchPath, libraries);
        shader_gen_context_->registerSourceCodeSearchPath(searchPath);

        shader_gen_context_->pushUserData(
            mx::HW::USER_DATA_BINDING_CONTEXT, BindlessContext::create());
    });
}

void Hd_USTC_CG_MaterialX::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    spdlog::info("MaterialX::Sync called for material '{}'", GetId().GetText());

    auto param = static_cast<Hd_USTC_CG_RenderParam*>(renderParam);

    ensure_material_data_handle(param);

    HdMaterialNetwork2 hdNetwork;
    SdfPath materialPath;

    SdfPath surfTerminalPath;
    HdMaterialNode2 const* surfTerminal;
    HdMaterialNetwork2Interface netInterface = FetchMaterialNetwork(
        sceneDelegate, hdNetwork, materialPath, surfTerminalPath, surfTerminal);

    spdlog::info(
        "MaterialX: MaterialPath = '{}', SurfTerminalPath = '{}'",
        materialPath.GetText(),
        surfTerminalPath.GetText());

    HdMtlxTexturePrimvarData hdMtlxData;

    DocumentPtr mtlx_document = HdMtlxCreateMtlxDocumentFromHdNetworkFast(
        hdNetwork,
        *surfTerminal,
        surfTerminalPath,
        materialPath,
        libraries,
        &hdMtlxData);

    if (!mtlx_document) {
        spdlog::error(
            "MaterialX: Failed to create MaterialX document for '{}'",
            GetId().GetText());
        *dirtyBits = HdChangeTracker::Clean;
        return;
    }

    spdlog::info("MaterialX: Created MaterialX document successfully");

    CollectTextures(netInterface, hdMtlxData);

    MtlxGenerateShader(mtlx_document, netInterface, hdMtlxData);

    BuildGPUTextures(param);

    *dirtyBits = HdChangeTracker::Clean;
}

void Hd_USTC_CG_MaterialX::ensure_shader_ready(const ShaderFactory& factory)
{
    if (shader_ready) {
        return;
    }

    Hd_USTC_CG_Material::ensure_shader_ready(factory);

    if (!eval_shader_source.empty()) {
        spdlog::info(
            "MaterialX: Processing shader source ({} bytes)",
            eval_shader_source.size());

        // Replace the data loading placeholder with actual data code
        constexpr char DATA_PLACEHOLDER[] = "$BindlessDataLoading";
        size_t pos = eval_shader_source.find(DATA_PLACEHOLDER);
        if (pos != std::string::npos) {
            eval_shader_source.replace(
                pos, strlen(DATA_PLACEHOLDER), get_data_code);
        }

        try {
            std::filesystem::create_directories("generated_shaders");
            std::ofstream out("generated_shaders/" + material_name + ".slang");
            if (out.is_open()) {
                out << eval_shader_source;
                out.close();
                spdlog::info(
                    "MaterialX: Saved shader to generated_shaders/{}.slang",
                    material_name);
            }
        }
        catch (const std::exception& e) {
            TF_WARN("Failed to save generated shader: %s", e.what());
        }
        final_shader_source = eval_shader_source + slang_source_code_main;
    }
    else {
        spdlog::warn(
            "MaterialX: eval_shader_source is empty for material '{}'",
            GetId().GetText());
    }

    // Combine shader parts into final source

    ProgramDesc program_desc;
    program_desc.add_source_code(final_shader_source);
    program_desc.set_shader_type(nvrhi::ShaderType::Callable);
    program_desc.set_entry_name(material_name);

    spdlog::info("MaterialX: Creating shader program for '{}'", material_name);
    final_program = factory.createProgram(program_desc);

    if (!final_program) {
        spdlog::error(
            "MaterialX: Failed to create shader program for '{}'",
            material_name);
    }
    else {
        spdlog::info(
            "MaterialX: Shader program created successfully for '{}'",
            material_name);
    }

    assert(final_program);

    shader_ready = true;
}

void Hd_USTC_CG_MaterialX::BuildGPUTextures(
    Hd_USTC_CG_RenderParam* render_param)
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
                
                // Force linear format for non-sRGB textures (like normal maps)
                if (!texture_resource.second.isSRGB) {
                    if (desc.format == nvrhi::Format::SRGBA8_UNORM) {
                        desc.format = nvrhi::Format::RGBA8_UNORM;
                    }
                }

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

void Hd_USTC_CG_MaterialX::CollectTextures(
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

        bool isSRGB = false;
        if (sourceColorSpace.IsHolding<std::string>()) {
            std::string colorSpace = sourceColorSpace.Get<std::string>();
            if (colorSpace == "srgb_texture") {
                isSRGB = true;
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
        textureResources[textureNodeName.GetString()].isSRGB = isSRGB;
    }
}

HdMaterialNetwork2Interface Hd_USTC_CG_MaterialX::FetchMaterialNetwork(
    HdSceneDelegate* sceneDelegate,
    HdMaterialNetwork2& hdNetwork,
    SdfPath& materialPath,
    SdfPath& surfTerminalPath,
    HdMaterialNode2 const*& surfTerminal)
{
    HdMaterialNetwork2Interface netInterface =
        FetchNetInterface(sceneDelegate, hdNetwork, materialPath);
    _FixNodeTypes(&netInterface);
    _FixNodeValues(&netInterface);

    const TfToken& terminalNodeName = HdMaterialTerminalTokens->surface;

    surfTerminal =
        _GetTerminalNode(hdNetwork, terminalNodeName, &surfTerminalPath);
    return netInterface;
}

void Hd_USTC_CG_MaterialX::MtlxGenerateShader(
    MaterialX::DocumentPtr mtlx_document,
    HdMaterialNetwork2Interface netInterface,
    HdMtlxTexturePrimvarData& hdMtlxData)
{
    _UpdateTextureNodes(
        &netInterface, hdMtlxData.hdTextureNodes, mtlx_document);

    auto renderable = findRenderableElements(mtlx_document);

    if (renderable.empty()) {
        TF_RUNTIME_ERROR("MaterialX: No renderable elements found in document");
        return;
    }

    _FixOmittedConnections(mtlx_document, renderable);
    using namespace mx;

    auto element = renderable[0];

    std::string elementName(element->getNamePath());
    material_name = mx::createValidName(elementName);

    spdlog::info(
        "MaterialX: Generating shader for material '{}'", material_name);

    ShaderGenerator& shader_generator_ =
        shader_gen_context_->getShaderGenerator();
    {
        std::lock_guard lock(shadergen_mutex);
        try {
            auto shader = shader_generator_.generate(
                material_name, element, *shader_gen_context_);

            if (!shader) {
                TF_RUNTIME_ERROR(
                    "MaterialX: Shader generation failed for material '%s'",
                    material_name.c_str());
                return;
            }

            eval_shader_source = shader->getSourceCode(mx::Stage::PIXEL);

            if (eval_shader_source.empty()) {
                TF_RUNTIME_ERROR(
                    "MaterialX: Empty shader source generated for material "
                    "'%s'",
                    material_name.c_str());
                return;
            }

            spdlog::info(
                "MaterialX: Generated {} bytes of shader code",
                eval_shader_source.size());

            BindlessContextPtr context =
                shader_gen_context_->getUserData<BindlessContext>(
                    mx::HW::USER_DATA_BINDING_CONTEXT);

            if (!context) {
                TF_RUNTIME_ERROR("MaterialX: Failed to get BindlessContext");
                return;
            }

            get_data_code = context->get_data_code();
            material_data = context->get_material_data();
            material_data_handle->write_data(&material_data);

            spdlog::info(
                "MaterialX: Shader generation complete for '{}'",
                material_name);
        }
        catch (const std::exception& e) {
            TF_RUNTIME_ERROR(
                "MaterialX: Exception during shader generation for '%s': %s",
                material_name.c_str(),
                e.what());
            return;
        }
    }
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
