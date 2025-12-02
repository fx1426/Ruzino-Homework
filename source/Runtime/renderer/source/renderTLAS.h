#pragma once
#include "api.h"
#include "geometries/mesh.h"
#include "nvrhi/nvrhi.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/pxr.h"

// SceneTypes
#include "../nodes/shaders/shaders/Scene/BindlessMaterial.slang"
#include "../nodes/shaders/shaders/Scene/SceneTypes.slang"
#include "../nodes/shaders/shaders/Scene/Lights/LightData.slang"
#include "internal/memory/DeviceMemoryPool.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

class HD_USTC_CG_API Hd_USTC_CG_RenderInstanceCollection {
   public:
    explicit Hd_USTC_CG_RenderInstanceCollection();
    ~Hd_USTC_CG_RenderInstanceCollection();

    nvrhi::rt::IAccelStruct *get_tlas();
    DescriptorTableManager *get_buffer_descriptor_table() const
    {
        return bindlessData.bufferDescriptorTableManager.get();
    }
    DescriptorTableManager *get_texture_descriptor_table() const
    {
        return bindlessData.textureDescriptorTableManager.get();
    }

    // DeviceMemoryPool<unsigned> index_pool;
    // DeviceMemoryPool<float> vertex_pool;
    DeviceMemoryPool<GeometryInstanceData> instance_pool;
    DeviceMemoryPool<nvrhi::rt::InstanceDesc> rt_instance_pool;
    DeviceMemoryPool<MeshDesc> mesh_pool;
    DeviceMemoryPool<MaterialDataBlob> material_pool;
    DeviceMemoryPool<MaterialHeader> material_header_pool;
    DeviceMemoryPool<nvrhi::DrawIndirectArguments> draw_indirect_pool;
    DeviceMemoryPool<LightData> light_pool;

    struct BindlessData {
        BindlessData();
        std::shared_ptr<DescriptorTableManager> bufferDescriptorTableManager;
        std::shared_ptr<DescriptorTableManager> textureDescriptorTableManager;

        nvrhi::BindingLayoutHandle bufferBindlessLayout;
        nvrhi::BindingLayoutHandle textureBindlessLayout;
    };
    BindlessData bindlessData;

    void set_require_rebuild_tlas()
    {
        require_rebuild_tlas = true;
    }
    
    bool get_require_rebuild_tlas() const
    {
        return require_rebuild_tlas;
    }
    
    // Track material changes
    uint32_t material_version = 0;
    void mark_materials_dirty() { material_version++; }
    uint32_t get_material_version() const { return material_version; }
    
    // Track geometry/buffer changes
    uint32_t geometry_version = 0;
    void mark_geometry_dirty() { geometry_version++; }
    uint32_t get_geometry_version() const { return geometry_version; }

   private:
    nvrhi::rt::AccelStructHandle TLAS;

    bool require_rebuild_tlas = true;
    void rebuild_tlas();
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
