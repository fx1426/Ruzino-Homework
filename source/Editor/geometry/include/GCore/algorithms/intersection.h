#pragma once

#include <pxr/base/gf/ray.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include "GCore/GOP.h"
#include "GCore/api.h"
#include "RHI/ResourceManager/resource_allocator.hpp"
USTC_CG_NAMESPACE_OPEN_SCOPE
struct MeshDesc;
GEOMETRY_API void init_gpu_geometry_algorithms();
GEOMETRY_API void deinit_gpu_geometry_algorithms();

GEOMETRY_API ResourceAllocator& get_resource_allocator();

struct GEOMETRY_API PointSample {
    pxr::GfVec3f position;
    pxr::GfVec3f normal;
    pxr::GfVec2f uv;
    unsigned valid;
};

// Remember to destroy the geometry explicitly with the resource allocator after
// use.
GEOMETRY_API nvrhi::rt::AccelStructHandle get_geomtry_tlas(
    const Geometry& geometry,
    MeshDesc* out_mesh_desc = nullptr,
    nvrhi::BufferHandle* out_vertex_buffer = nullptr);

GEOMETRY_API pxr::VtArray<PointSample> IntersectWithBuffer(
    const nvrhi::BufferHandle& ray_buffer,
    size_t ray_count,
    const Geometry& BaseMesh);

GEOMETRY_API nvrhi::BufferHandle IntersectToBuffer(
    const nvrhi::BufferHandle& ray_buffer,
    size_t ray_count,
    const Geometry& BaseMesh);

GEOMETRY_API pxr::VtArray<PointSample> Intersect(
    const pxr::VtArray<pxr::GfRay>& rays,
    const Geometry& BaseMesh);

GEOMETRY_API pxr::VtArray<PointSample> Intersect(
    const pxr::VtArray<pxr::GfVec3f>& start_point,
    const pxr::VtArray<pxr::GfVec3f>& next_point,
    const Geometry& BaseMesh);

// result should be of size start_point.size() * next_point.size()
GEOMETRY_API pxr::VtArray<PointSample> IntersectInterweaved(
    const pxr::VtArray<pxr::GfVec3f>& start_point,
    const pxr::VtArray<pxr::GfVec3f>& next_point,
    const Geometry& BaseMesh);

USTC_CG_NAMESPACE_CLOSE_SCOPE
