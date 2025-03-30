#pragma once
#include <complex.h>

#include "Logger/Logger.h"
#include "MaterialX/SlangShaderGenerator.h"
#include "api.h"
#include "map.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hd/materialNetwork2Interface.h"
#include "pxr/imaging/hdMtlx/hdMtlx.h"
#include "pxr/imaging/hio/image.h"
#include "renderParam.h"
#include "renderer/program_vars.hpp"

namespace pxr {
class Hio_OpenEXRImage;

}

USTC_CG_NAMESPACE_OPEN_SCOPE
class Shader;
using namespace pxr;

class Hio_StbImage;
class HD_USTC_CG_API Hd_USTC_CG_Material : public HdMaterial {
   public:
    explicit Hd_USTC_CG_Material(SdfPath const& id);

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Finalize(HdRenderParam* renderParam) override;

    std::vector<TextureHandle> GetTextures() const;

    void ensure_material_data_handle(Hd_USTC_CG_RenderParam* render_param);

    void ensure_shader_ready(const ShaderFactory& factory);

    unsigned GetMaterialLocation() const;

    std::string GetShader(const ShaderFactory& factory);

    std::string GetMaterialName() const
    {
        return material_name;
    }

   private:
    HdMaterialNetwork2 surfaceNetwork;

    std::string eval_shader_source;
    std::string get_data_code;
    std::string material_name;
    std::string final_shader_source;

    bool shader_ready = false;

    std::unordered_map<std::string, std::string> texturePaths;

    struct TextureResource {
        std::string filePath;
        HioImageSharedPtr image;
        nvrhi::TextureHandle texture;
        DescriptorHandle descriptor;
    };

    std::unordered_map<std::string, TextureResource> textureResources;

    void CollectTextures(
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData hdMtlxData);

    void BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param);

    void MtlxGenerateShader(
        MaterialX::DocumentPtr mtlx_document,
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData& hdMtlxData);

    HdMaterialNetwork2Interface FetchMaterialNetwork(
        HdSceneDelegate* sceneDelegate,
        HdMaterialNetwork2& hdNetwork,
        SdfPath& materialPath,
        SdfPath& surfTerminalPath,
        HdMaterialNode2 const*& surfTerminal);

    DeviceMemoryPool<MaterialDataBlob>::MemoryHandle material_data_handle;
    MaterialDataBlob material_data;

    static MaterialX::GenContextPtr shader_gen_context_;
    static MaterialX::DocumentPtr libraries;

    static std::once_flag shader_gen_initialized_;
    static std::mutex texture_mutex;
    static std::mutex shadergen_mutex;
    static std::mutex material_data_handle_mutex;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE