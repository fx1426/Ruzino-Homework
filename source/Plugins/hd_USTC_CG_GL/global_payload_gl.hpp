#pragma once

#include "RHI/ResourceManager/resource_allocator.hpp"
#include "pxr/usd/sdf/path.h"

namespace USTC_CG {
class Hd_USTC_CG_RenderInstanceCollection;
class Hd_USTC_CG_Light;
class Hd_USTC_CG_Camera;
class Hd_USTC_CG_Mesh;
class Hd_USTC_CG_Material;
}  // namespace USTC_CG

namespace USTC_CG {

struct RenderGlobalPayloadGL {
    RenderGlobalPayloadGL()
    {
    }
    RenderGlobalPayloadGL(
        pxr::VtArray<Hd_USTC_CG_Camera*>* cameras,
        pxr::VtArray<Hd_USTC_CG_Light*>* lights,
        pxr::VtArray<Hd_USTC_CG_Mesh*>* meshes,
        pxr::TfHashMap<pxr::SdfPath, Hd_USTC_CG_Material*, pxr::TfHash>*
            materials)
        : cameras(cameras),
          lights(lights),
          meshes(meshes),
          materials(materials)
    {
    }

    RenderGlobalPayloadGL(const RenderGlobalPayloadGL& rhs)
        : cameras(rhs.cameras),
          lights(rhs.lights),
          meshes(rhs.meshes),
          materials(rhs.materials)
    {
    }

    RenderGlobalPayloadGL& operator=(const RenderGlobalPayloadGL& rhs)
    {
        cameras = rhs.cameras;
        lights = rhs.lights;
        meshes = rhs.meshes;
        materials = rhs.materials;
        return *this;
    }

    ResourceAllocator resource_allocator;

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

    auto& get_meshes() const
    {
        return *meshes;
    }

   private:
    pxr::VtArray<Hd_USTC_CG_Camera*>* cameras;
    pxr::VtArray<Hd_USTC_CG_Light*>* lights;
    pxr::TfHashMap<pxr::SdfPath, Hd_USTC_CG_Material*, pxr::TfHash>* materials;
    pxr::VtArray<Hd_USTC_CG_Mesh*>* meshes;
};

}  // namespace USTC_CG
