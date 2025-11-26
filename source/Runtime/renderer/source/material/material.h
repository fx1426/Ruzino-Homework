#pragma once

#include "GPUContext/program_vars.hpp"
#include "api.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hd/materialNetwork2Interface.h"
#include "pxr/imaging/hdMtlx/hdMtlx.h"
#include "pxr/imaging/hio/image.h"
#include "renderParam.h"

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

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Finalize(HdRenderParam* renderParam) override;

    void ensure_material_data_handle(Hd_USTC_CG_RenderParam* render_param);

    virtual void ensure_shader_ready(const ShaderFactory& factory);

    unsigned GetMaterialLocation() const;

    std::string GetShader(const ShaderFactory& factory);

    std::string GetMaterialName() const
    {
        return material_name;
    }

    virtual void update_data_loader(
        DescriptorIndex descriptor_index,
        const std::string& texture_name)
    {
    }

   protected:
    HdMaterialNetwork2 surfaceNetwork;

    std::string eval_shader_source;
    std::string material_name;
    std::string final_shader_source;

    bool shader_ready = false;

    std::unordered_map<std::string, std::string> texturePaths;
    ProgramHandle final_program;

    struct TextureResource {
        std::string filePath;
        HioImageSharedPtr image;
        nvrhi::TextureHandle texture;
        DescriptorHandle descriptor;
        bool isSRGB = false;  // Track whether this texture should use sRGB format
    };

    std::unordered_map<std::string, TextureResource> textureResources;

    HdMaterialNetwork2Interface FetchNetInterface(
        HdSceneDelegate* sceneDelegate,
        HdMaterialNetwork2& hdNetwork,
        SdfPath& materialPath);

    DeviceMemoryPool<MaterialDataBlob>::MemoryHandle material_data_handle;
    DeviceMemoryPool<MaterialHeader>::MemoryHandle material_header_handle;
    MaterialDataBlob material_data;

    std::string slang_source_code_main;
    static std::string slang_source_code_template;
    static std::string eval_source_code_fallback;
    static std::mutex texture_mutex;
    static std::mutex material_data_handle_mutex;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
