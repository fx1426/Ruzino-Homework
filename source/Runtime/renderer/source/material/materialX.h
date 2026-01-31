#pragma once
#include "MaterialXCore/Document.h"
#include "MaterialXGenShader/Library.h"
#include "material.h"

namespace pxr {
class Hio_OpenEXRImage;
}

RUZINO_NAMESPACE_OPEN_SCOPE

class Shader;
using namespace pxr;

class Hio_StbImage;
class HD_RUZINO_API Hd_RUZINO_MaterialX : public Hd_RUZINO_Material {
   public:
    explicit Hd_RUZINO_MaterialX(SdfPath const& id);

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    void ensure_shader_ready(const ShaderFactory& factory) override;

    // Upload material data to GPU after texture loading is complete
    void upload_material_data();

   protected:
    void BuildGPUTextures(Hd_RUZINO_RenderParam* render_param);
    void CollectTextures(
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData hdMtlxData);
    HdMaterialNetwork2Interface FetchMaterialNetwork(
        HdSceneDelegate* sceneDelegate,
        HdMaterialNetwork2& hdNetwork,
        SdfPath& materialPath,
        SdfPath& surfTerminalPath,
        HdMaterialNode2 const*& surfTerminal);

    std::string get_data_code;
    // Mapping from texture variable name to data location for texture IDs
    std::unordered_map<std::string, unsigned int> texture_id_locations;
    // Flag to track if material data needs to be uploaded to GPU
    bool material_data_dirty = false;

   private:
    void MtlxGenerateShader(
        MaterialX::ElementPtr mtlx_element,
        HdMaterialNetwork2Interface netInterface,
        HdMtlxTexturePrimvarData& hdMtlxData);

    static MaterialX::GenContextPtr shader_gen_context_;
    static MaterialX::DocumentPtr libraries;
    static MaterialX::DocumentPtr
        shared_document;  // Shared document for all materials (avoids
                          // copy/import)
    static std::once_flag shader_gen_initialized_;
    static std::mutex shadergen_mutex;
    static std::mutex document_mutex;  // Protects shared_document

    // Cache for NodeDef lookups to avoid expensive searches
    static std::unordered_map<std::string, MaterialX::NodeDefPtr>
        nodedef_cache_;
    static std::mutex nodedef_cache_mutex_;
};

RUZINO_NAMESPACE_CLOSE_SCOPE
