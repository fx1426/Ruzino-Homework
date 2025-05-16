#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdVol/openVDBAsset.h>

#include <string>

#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/InstancerComponent.h"
#include "GCore/Components/MaterialComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/Components/VolumeComponent.h"
#include "GCore/Components/XformComponent.h"
#include "GCore/geom_payload.hpp"
#include "geom_node_base.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/usd/usd/payloads.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(write_usd)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Sub Path").optional(true);
}

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

NODE_EXECUTION_FUNCTION(write_usd)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();

    auto geometry = params.get_input<Geometry>("Geometry");

    auto mesh = geometry.get_component<MeshComponent>();
    auto points = geometry.get_component<PointsComponent>();
    auto curve = geometry.get_component<CurveComponent>();
    auto volume = geometry.get_component<VolumeComponent>();
    auto instancer = geometry.get_component<InstancerComponent>();

    assert(!(points && mesh));

    pxr::UsdTimeCode time = global_payload.current_time;

    pxr::UsdStageRefPtr stage = global_payload.stage;
    auto sdf_path = global_payload.prim_path;

    auto sub_path = params.get_input<std::string>("Sub Path");
    if (!std::string(sub_path.c_str()).empty()) {
        if (!legal(sub_path)) {
            log::error("Illegal sub path");
            return false;
        }
        sdf_path = sdf_path.AppendPath(pxr::SdfPath(sub_path.c_str()));
    }

    if (instancer) {
        sdf_path =
            global_payload.prim_path.AppendPath(pxr::SdfPath("Prototype"));
    }

    if (mesh) {
        pxr::UsdGeomMesh usdgeom = pxr::UsdGeomMesh::Define(stage, sdf_path);

        if (usdgeom) {
#if USE_USD_SCRATCH_BUFFER
            copy_prim(mesh->get_usd_mesh().GetPrim(), usdgeom.GetPrim());
#else
            usdgeom.CreatePointsAttr().Set(mesh->get_vertices(), time);
            usdgeom.CreateFaceVertexCountsAttr().Set(
                mesh->get_face_vertex_counts(), time);
            usdgeom.CreateFaceVertexIndicesAttr().Set(
                mesh->get_face_vertex_indices(), time);

            auto primVarAPI = pxr::UsdGeomPrimvarsAPI(usdgeom);

            if (!mesh->get_normals().empty()) {
                usdgeom.CreateNormalsAttr().Set(mesh->get_normals(), time);
                if (mesh->get_normals().size() == mesh->get_vertices().size()) {
                    usdgeom.SetNormalsInterpolation(pxr::UsdGeomTokens->vertex);
                }
                else {
                    usdgeom.SetNormalsInterpolation(
                        pxr::UsdGeomTokens->faceVarying);
                }
                usdgeom.CreateSubdivisionSchemeAttr().Set(
                    pxr::UsdGeomTokens->none);
            }
            else {
                usdgeom.CreateNormalsAttr().Block();
            }
            if (!mesh->get_display_color().empty()) {
                auto colorPrimvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken("displayColor"),
                    pxr::SdfValueTypeNames->Color3fArray);
                colorPrimvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                colorPrimvar.Set(mesh->get_display_color(), time);
            }
            if (!mesh->get_texcoords_array().empty()) {
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken("UVMap"),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.Set(mesh->get_texcoords_array(), time);
                if (mesh->get_texcoords_array().size() ==
                    mesh->get_vertices().size()) {
                    primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                }
                else {
                    primvar.SetInterpolation(pxr::UsdGeomTokens->faceVarying);
                }
            }

#endif
            usdgeom.CreateDoubleSidedAttr().Set(true);

            // It's invalid to use pxr::TfToken("some_prefix" + some_string)
            // directly, so we need to create a new string first.

            for (const std::string& name :
                 mesh->get_vertex_scalar_quantity_names()) {
                auto values = mesh->get_vertex_scalar_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:scalar:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->FloatArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_face_scalar_quantity_names()) {
                auto values = mesh->get_face_scalar_quantity(name);
                const std::string primvar_name =
                    "polyscope:face:scalar:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->FloatArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values, time);
            }

            for (std::string& name : mesh->get_vertex_color_quantity_names()) {
                auto values = mesh->get_vertex_color_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:color:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Color3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_face_color_quantity_names()) {
                auto values = mesh->get_face_color_quantity(name);
                const std::string primvar_name = "polyscope:face:color:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Color3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_vertex_vector_quantity_names()) {
                auto values = mesh->get_vertex_vector_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:vector:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Vector3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_face_vector_quantity_names()) {
                auto values = mesh->get_face_vector_quantity(name);
                const std::string primvar_name =
                    "polyscope:face:vector:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Vector3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_face_corner_parameterization_quantity_names()) {
                auto values =
                    mesh->get_face_corner_parameterization_quantity(name);
                const std::string primvar_name =
                    "polyscope:face_corner:parameterization:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->faceVarying);
                primvar.Set(values, time);
            }

            for (const std::string& name :
                 mesh->get_vertex_parameterization_quantity_names()) {
                auto values = mesh->get_vertex_parameterization_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:parameterization:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values, time);
            }
        }
    }
    else if (points) {
        pxr::UsdGeomPoints usdpoints =
            pxr::UsdGeomPoints::Define(stage, sdf_path);

        usdpoints.CreatePointsAttr().Set(points->get_vertices(), time);

        if (points->get_width().size() > 0) {
            usdpoints.CreateWidthsAttr().Set(points->get_width(), time);
        }

        auto PrimVarAPI = pxr::UsdGeomPrimvarsAPI(usdpoints);
        if (points->get_display_color().size() > 0) {
            pxr::UsdGeomPrimvar colorPrimvar = PrimVarAPI.CreatePrimvar(
                pxr::TfToken("displayColor"),
                pxr::SdfValueTypeNames->Color3fArray);
            colorPrimvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
            colorPrimvar.Set(points->get_display_color(), time);
        }
    }
    else if (curve) {
        pxr::UsdGeomBasisCurves usd_curve =
            pxr::UsdGeomBasisCurves::Define(stage, sdf_path);
        if (usd_curve) {
#if USE_USD_SCRATCH_BUFFER
            copy_prim(curve->get_usd_curve().GetPrim(), usd_curve.GetPrim());
#else
            usd_curve.CreatePointsAttr().Set(curve->get_vertices(), time);
            usd_curve.CreateWidthsAttr().Set(curve->get_width(), time);
            usd_curve.CreateCurveVertexCountsAttr().Set(
                curve->get_vert_count(), time);
            usd_curve.CreateNormalsAttr().Set(curve->get_curve_normals(), time);
            usd_curve.CreateDisplayColorAttr().Set(
                curve->get_display_color(), time);
            usd_curve.CreateWrapAttr().Set(
                curve->get_periodic() ? pxr::UsdGeomTokens->periodic
                                      : pxr::UsdGeomTokens->nonperiodic,
                time);
            usd_curve.CreateTypeAttr().Set(
                curve->get_type() == CurveComponent::CurveType::Linear
                    ? pxr::UsdGeomTokens->linear
                    : pxr::UsdGeomTokens->cubic,
                time);
#endif
        }
    }
    else if (volume) {
        auto openvdb_asset = pxr::UsdVolOpenVDBAsset::Define(stage, sdf_path);

        // Save the volume grid onto the disk
        auto file_name = "volume" + sdf_path.GetName() +
                         std::to_string(time.GetValue()) + ".vdb";
        volume->write_disk(file_name);
        openvdb_asset.CreateFilePathAttr().Set(pxr::SdfAssetPath(file_name));
    }
    else {
        params.set_error("No valid geometry component found");
        return false;
    }

    if (instancer) {
        auto instancer_component =
            pxr::UsdGeomPointInstancer::Define(stage, global_payload.prim_path);
        instancer_component.CreatePrototypesRel().SetTargets({ sdf_path });

        auto transforms = instancer->get_instances();

        pxr::VtVec3fArray positions = pxr::VtVec3fArray(transforms.size());
        pxr::VtQuathArray orientations = pxr::VtQuathArray(transforms.size());
        pxr::VtVec3fArray scales = pxr::VtVec3fArray(transforms.size());

        for (size_t i = 0; i < transforms.size(); ++i) {
            pxr::GfVec3f translation;
            pxr::GfQuath rotation;
            // pxr::GfVec3f scale;
            translation = pxr::GfVec3f(transforms[i].ExtractTranslation());
            rotation = pxr::GfQuath(transforms[i].ExtractRotationQuat());
            // scale = pxr::GfVec3f(transforms[i].ExtractScale());
            positions[i] = translation;
            orientations[i] = rotation;
            // scales[i] = scale;
        }
        instancer_component.CreateProtoIndicesAttr().Set(
            pxr::VtIntArray(instancer->get_proto_indices()));
        instancer_component.CreatePositionsAttr().Set(positions);
        instancer_component.CreateOrientationsAttr().Set(orientations);
        // instancer_component.CreateScalesAttr().Set(scales);
    }

    // Material and Texture
    auto material_component = geometry.get_component<MaterialComponent>();
    if (material_component) {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        if (legal(std::string(material_component->textures[0].c_str()))) {
            auto material_path = material_component->get_material_path();

            auto material =
                material_component->define_material(stage, material_path);
            usdgeom.GetPrim().ApplyAPI(pxr::UsdShadeTokens->MaterialBindingAPI);
            pxr::UsdShadeMaterialBindingAPI(usdgeom).Bind(material);
        }
        else {
            // TODO: Throw something
        }
    }

    auto xform_component = geometry.get_component<XformComponent>();
    if (xform_component) {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        // Transform
        assert(
            xform_component->translation.size() ==
            xform_component->rotation.size());

        pxr::GfMatrix4d final_transform = xform_component->get_transform();

        auto xform_op = usdgeom.GetTransformOp();
        if (!xform_op) {
            xform_op = usdgeom.AddTransformOp();
        }
        xform_op.Set(final_transform, time);
    }
    else {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        auto xform_op = usdgeom.GetTransformOp();
        if (!xform_op) {
            xform_op = usdgeom.AddTransformOp();
        }
        xform_op.Set(pxr::GfMatrix4d(1), time);
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
