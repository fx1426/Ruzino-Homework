#ifdef GEOM_USD_EXTENSION

#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <string>

#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/InstancerComponent.h"
#include "GCore/Components/VolumeComponent.h"
#include "GCore/geom_payload.hpp"
#include "GCore/usd_extension.h"
#include "geom_node_base.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/usd/usd/payloads.h"

NODE_DEF_OPEN_SCOPE

bool legal(const std::string& string)
{
    if (string.empty()) {
        return false;
    }
    if (std::find_if(string.begin(), string.end(), [](char val) {
            return val == '(' || val == ')' || val == ',';
        }) == string.end()) {
        return true;
    }
    return false;
}

NODE_DECLARATION_FUNCTION(write_usd)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Sub Path").optional(true);
}

NODE_EXECUTION_FUNCTION(write_usd)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();

    auto geometry = params.get_input<Geometry>("Geometry");

    pxr::UsdTimeCode time = global_payload.current_time;

    pxr::UsdStageRefPtr stage = global_payload.stage;
    auto sdf_path = global_payload.prim_path;

    auto sub_path = params.get_input<std::string>("Sub Path");
    if (!std::string(sub_path.c_str()).empty()) {
        if (!legal(sub_path)) {
            spdlog::error("Illegal sub path");
            return false;
        }
        sdf_path = sdf_path.AppendPath(pxr::SdfPath(sub_path.c_str()));
    }

    if (!write_geometry_to_usd(geometry, stage, sdf_path, time)) {
        return false;
    }

    if (global_payload.has_simulation) {
        pxr::UsdPrim prim = stage->GetPrimAtPath(sdf_path);
        prim.CreateAttribute(
                pxr::TfToken("Animatable"), pxr::SdfValueTypeNames->Bool)
            .Set(true);
    }
    else {
        pxr::UsdPrim prim = stage->GetPrimAtPath(sdf_path);
        prim.CreateAttribute(
                pxr::TfToken("Animatable"), pxr::SdfValueTypeNames->Bool)
            .Set(false);
    }

    pxr::UsdGeomImageable(stage->GetPrimAtPath(sdf_path)).MakeVisible();
    return true;
}

NODE_DECLARATION_REQUIRED(write_usd);

NODE_DECLARATION_UI(write_usd);
NODE_DEF_CLOSE_SCOPE
#endif
