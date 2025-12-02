#pragma once
#include "RHI/ResourceManager/resource_allocator.hpp"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/hash.h"
#include "pxr/usd/sdf/path.h"

namespace USTC_CG {
class Hd_USTC_CG_RenderInstanceCollection;
class LensSystem;
class Hd_USTC_CG_Light;
class Hd_USTC_CG_Camera;
class Hd_USTC_CG_Mesh;
class Hd_USTC_CG_Material;
}  // namespace USTC_CG

namespace USTC_CG {

struct RenderGlobalPayload {
    RenderGlobalPayload()
    {
    }
    RenderGlobalPayload(
        pxr::VtArray<Hd_USTC_CG_Camera*>* cameras,
        pxr::VtArray<Hd_USTC_CG_Light*>* lights,
        pxr::TfHashMap<pxr::SdfPath, Hd_USTC_CG_Material*, pxr::TfHash>*
            materials,
        nvrhi::IDevice* nvrhi_device)
        : cameras(cameras),
          lights(lights),
          materials(materials),
          nvrhi_device(nvrhi_device),
          shader_factory(&resource_allocator)
    {
        shader_factory.set_search_path(RENDERER_SHADER_DIR);
        shader_factory.add_search_path("usd/hd_USTC_CG/resources/libraries");
        resource_allocator.device = nvrhi_device;
        resource_allocator.shader_factory = &shader_factory;
    }

    RenderGlobalPayload(const RenderGlobalPayload& rhs)
        : cameras(rhs.cameras),
          lights(rhs.lights),
          materials(rhs.materials),
          nvrhi_device(rhs.nvrhi_device),
          shader_factory(&resource_allocator)
    {
        shader_factory.set_search_path(RENDERER_SHADER_DIR);
        shader_factory.add_search_path("usd/hd_USTC_CG/resources/libraries");

        resource_allocator.device = nvrhi_device;
        resource_allocator.shader_factory = &shader_factory;
    }

    RenderGlobalPayload& operator=(const RenderGlobalPayload& rhs)
    {
        cameras = rhs.cameras;
        lights = rhs.lights;
        materials = rhs.materials;
        nvrhi_device = rhs.nvrhi_device;
        shader_factory = ShaderFactory(&resource_allocator);
        shader_factory.set_search_path(RENDERER_SHADER_DIR);
        shader_factory.add_search_path("usd/hd_USTC_CG/resources/libraries");

        resource_allocator.device = nvrhi_device;
        resource_allocator.shader_factory = &shader_factory;
        return *this;
    }

    ResourceAllocator resource_allocator;
    ShaderFactory shader_factory;
    nvrhi::IDevice* nvrhi_device;
    Hd_USTC_CG_RenderInstanceCollection* InstanceCollection;
    bool reset_accumulation = false;

    // Scene dirty flags for tracking changes
    enum class SceneDirtyBits : uint32_t {
        Clean = 0,
        DirtyMaterials = 1 << 0,      // Material shaders changed
        DirtyGeometry = 1 << 1,        // Mesh topology/vertices changed
        DirtyTransforms = 1 << 2,      // Instance transforms changed
        DirtyLights = 1 << 3,          // Light count or properties changed
        DirtyCamera = 1 << 4,          // Camera changed
        DirtyTLAS = 1 << 5,            // TLAS needs rebuild
        DirtyAll = 0xFFFFFFFF
    };
    
    uint32_t scene_dirty_flags = static_cast<uint32_t>(SceneDirtyBits::DirtyAll);
    
    void mark_dirty(SceneDirtyBits bits) {
        scene_dirty_flags |= static_cast<uint32_t>(bits);
    }
    
    void clear_dirty(SceneDirtyBits bits) {
        scene_dirty_flags &= ~static_cast<uint32_t>(bits);
    }
    
    bool is_dirty(SceneDirtyBits bits) const {
        return (scene_dirty_flags & static_cast<uint32_t>(bits)) != 0;
    }
    
    void clear_all_dirty() {
        scene_dirty_flags = static_cast<uint32_t>(SceneDirtyBits::Clean);
    }

    auto& get_cameras() const
    {
        return *cameras;
    }

    auto& get_lights() const
    {
        return *lights;
    }

    auto& get_materials() const
    {
        return *materials;
    }

    LensSystem* lens_system;

   private:
    pxr::VtArray<Hd_USTC_CG_Camera*>* cameras;
    pxr::VtArray<Hd_USTC_CG_Light*>* lights;
    pxr::TfHashMap<pxr::SdfPath, Hd_USTC_CG_Material*, pxr::TfHash>* materials;
};

}  // namespace USTC_CG
