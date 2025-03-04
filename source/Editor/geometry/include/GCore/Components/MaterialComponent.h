#pragma once

#include <pxr/usd/usdMtlx/materialXConfigAPI.h>
#include <pxr/usd/usdShade/material.h>

#include <string>

#include "GCore/Components.h"
#include "GCore/GOP.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
struct GEOMETRY_API MaterialComponent : public GeometryComponent {
    explicit MaterialComponent(Geometry* attached_operand)
        : GeometryComponent(attached_operand)
    {
    }

    void apply_transform(const pxr::GfMatrix4d& transform) override
    {
    }

    GeometryComponentHandle copy(Geometry* operand) const override
    {
        auto ret = std::make_shared<MaterialComponent>(operand);

        // This is fast because the VtArray has the copy on write mechanism
        ret->textures = this->textures;
        return ret;
    }

    std::string to_string() const override
    {
        return {};
    }

    pxr::UsdShadeMaterial define_material(
        pxr::UsdStageRefPtr stage,
        pxr::SdfPath path);

    void set_materialx_path(pxr::SdfPath path)
    {
        mtlx_material_path = path;
    }

    pxr::SdfPath get_material_path();

    std::vector<std::string> textures;

    pxr::SdfPath mtlx_material_path;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
