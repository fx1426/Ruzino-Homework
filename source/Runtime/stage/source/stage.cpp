#include "stage/stage.hpp"

#include <pxr/base/gf/rotation.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/payloads.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdMtlx/reader.h>
#include <pxr/usd/usdMtlx/utils.h>
#include <pxr/usd/usdShade/material.h>

#include <filesystem>

#include "MaterialXFormat/File.h"
#include "MaterialXFormat/Util.h"
#include "animation.h"

RUZINO_NAMESPACE_OPEN_SCOPE
#define SAVE_ALL_THE_TIME 0

Stage::Stage()
{
    std::string stage_path = "../../Assets/demo_stroke.usdc";
    stage_path = "../../Assets/stage.usdc";

    std::filesystem::path executable_path;

#ifdef _WIN32
    char p[MAX_PATH];
    GetModuleFileNameA(NULL, p, MAX_PATH);
    executable_path = std::filesystem::path(p).parent_path();
#else
    char p[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", p, PATH_MAX);
    if (count != -1) {
        p[count] = '\0';
        executable_path = std::filesystem::path(path).parent_path();
    }
    else {
        throw std::runtime_error("Failed to get executable path.");
    }
#endif

    std::filesystem::path abs_path;
    if (!stage_path.empty()) {
        abs_path = std::filesystem::path(stage_path);
    }
    else {
        spdlog::error("Path is empty.");
        return;
    }
    if (!abs_path.is_absolute()) {
        abs_path = executable_path / abs_path;
    }
    abs_path = abs_path.lexically_normal();
    m_stage_path = abs_path.string();

    // if stage.usda exists, load it
    stage = pxr::UsdStage::Open(abs_path.string());
    if (stage) {
        return;
    }

    stage = pxr::UsdStage::CreateNew(abs_path.string());
    stage->SetMetadata(pxr::UsdGeomTokens->metersPerUnit, 1.0);
    stage->SetMetadata(pxr::UsdGeomTokens->upAxis, pxr::TfToken("Z"));
}

Stage::Stage(const std::string& stage_path)
{
    std::filesystem::path abs_path;
    if (!stage_path.empty()) {
        abs_path = std::filesystem::path(stage_path);
    }
    else {
        spdlog::error("Path is empty.");
        return;
    }
    abs_path = abs_path.lexically_normal();
    m_stage_path = abs_path.string();
    // if stage.usda exists, load it
    stage = pxr::UsdStage::Open(abs_path.string());
    if (stage) {
        return;
    }
    stage = pxr::UsdStage::CreateNew(abs_path.string());
    stage->SetMetadata(pxr::UsdGeomTokens->metersPerUnit, 1.0);
    stage->SetMetadata(pxr::UsdGeomTokens->upAxis, pxr::TfToken("Z"));
}

Stage::~Stage()
{
    remove_prim(pxr::SdfPath("/scratch_buffer"));
    if (stage && !m_stage_path.empty()) {
        stage->Export(m_stage_path);
    }
    animatable_prims.clear();
}

void Stage::tick(float ellapsed_time)
{
    // Stage的全局时间码用于追踪整体状态
    // 但实际的仿真时间由每个prim独立管理
    // if (should_simulate()) 
    {
        // for each prim, if it is animatable, update it
        // 每个prim都会独立判断自己是否应该进行仿真
        for (auto&& prim : stage->Traverse()) {
            if (animation::WithDynamicLogicPrim::is_animatable(prim)) {
                if (animatable_prims.find(prim.GetPath()) ==
                    animatable_prims.end()) {
                    animatable_prims.emplace(
                        prim.GetPath(),
                        std::move(animation::WithDynamicLogicPrim(prim, this)));
                }

                animatable_prims.at(prim.GetPath()).update(ellapsed_time);
            }
        }

        auto current = current_time_code.GetValue();
        current += ellapsed_time;
        current_time_code = pxr::UsdTimeCode(current);
    }
}

void Stage::finish_tick()
{
}

pxr::UsdTimeCode Stage::get_current_time()
{
    return current_time_code;
}

void Stage::set_current_time(pxr::UsdTimeCode time)
{
    current_time_code = time;
}

pxr::UsdTimeCode Stage::get_render_time()
{
    return render_time_code;
}

void Stage::set_render_time(pxr::UsdTimeCode time)
{
    render_time_code = time;
}

template<typename T>
T Stage::create_prim(const pxr::SdfPath& path, const std::string& baseName)
    const
{
    int id = 0;
    while (stage->GetPrimAtPath(
        path.AppendPath(pxr::SdfPath(baseName + "_" + std::to_string(id))))) {
        id++;
    }
    auto a = T::Define(
        stage,
        path.AppendPath(pxr::SdfPath(baseName + "_" + std::to_string(id))));
#if SAVE_ALL_THE_TIME
    stage->Save();
#endif
    return a;
}

pxr::UsdPrim Stage::add_prim(const pxr::SdfPath& path)
{
    return stage->DefinePrim(path);
}

pxr::UsdShadeMaterial Stage::create_material(const pxr::SdfPath& path)
{
    auto material = create_prim<pxr::UsdShadeMaterial>(path, "material");
    
    // Add custom shader_path attribute for material callable shader
    auto shader_path_attr = material.GetPrim().CreateAttribute(
        pxr::TfToken("shader_path"),
        pxr::SdfValueTypeNames->String,
        false);
    shader_path_attr.Set(std::string(""));  // Empty by default
    
    return material;
}

pxr::UsdGeomSphere Stage::create_sphere(const pxr::SdfPath& path) const
{
    return create_prim<pxr::UsdGeomSphere>(path, "sphere");
}

pxr::UsdGeomCylinder Stage::create_cylinder(const pxr::SdfPath& path) const
{
    return create_prim<pxr::UsdGeomCylinder>(path, "cylinder");
}

pxr::UsdGeomCube Stage::create_cube(const pxr::SdfPath& path) const
{
    return create_prim<pxr::UsdGeomCube>(path, "cube");
}

pxr::UsdGeomXform Stage::create_xform(const pxr::SdfPath& path) const
{
    return create_prim<pxr::UsdGeomXform>(path, "xform");
}

pxr::UsdGeomMesh Stage::create_mesh(const pxr::SdfPath& path) const
{
    return create_prim<pxr::UsdGeomMesh>(path, "mesh");
}

pxr::UsdLuxRectLight Stage::create_rect_light(const pxr::SdfPath& path) const
{
    auto light = create_prim<pxr::UsdLuxRectLight>(path, "rect_light");
    light.GetIntensityAttr().Set(1.0f);
    light.GetWidthAttr().Set(2.0f);
    light.GetHeightAttr().Set(2.0f);

    auto xform = pxr::UsdGeomXformable(light);
    pxr::GfMatrix4d matrix;
    matrix.SetTranslate(pxr::GfVec3d(0.0, 0.0, 1.0));
    xform.MakeMatrixXform().Set(matrix);

    return light;
}

pxr::UsdLuxDistantLight Stage::create_distant_light(
    const pxr::SdfPath& path) const
{
    auto light = create_prim<pxr::UsdLuxDistantLight>(path, "distant_light");
    light.GetIntensityAttr().Set(1.0f);
    light.GetAngleAttr().Set(0.5f);

    auto xform = pxr::UsdGeomXformable(light);
    pxr::GfMatrix4d matrix;
    matrix.SetRotate(pxr::GfRotation(pxr::GfVec3d(1.0, 0.0, 0.0), 180.0));
    xform.MakeMatrixXform().Set(matrix);

    return light;
}

pxr::UsdLuxDiskLight Stage::create_disk_light(const pxr::SdfPath& path) const
{
    auto light = create_prim<pxr::UsdLuxDiskLight>(path, "disk_light");
    light.GetIntensityAttr().Set(1.0f);
    light.GetRadiusAttr().Set(1.0f);

    auto xform = pxr::UsdGeomXformable(light);
    pxr::GfMatrix4d matrix;
    matrix.SetTranslate(pxr::GfVec3d(0.0, 0.0, 2.0));
    xform.MakeMatrixXform().Set(matrix);

    return light;
}

pxr::UsdLuxDomeLight Stage::create_dome_light(const pxr::SdfPath& path) const
{
    auto light = create_prim<pxr::UsdLuxDomeLight>(path, "dome_light");
    light.GetIntensityAttr().Set(1.0f);

    // Add custom shader_path attribute for dome light callable shader
    auto shader_path_attr = light.GetPrim().CreateAttribute(
        pxr::TfToken("shader_path"),
        pxr::SdfValueTypeNames->String,
        false);
    shader_path_attr.Set(std::string(""));  // Empty by default

    return light;
}

void Stage::remove_prim(const pxr::SdfPath& path)
{
    if (animatable_prims.find(path) != animatable_prims.end()) {
        animatable_prims.erase(path);
    }
    stage->RemovePrim(path);  // This operation is in fact not recommended! In
                              // Omniverse applications, they set the prim to
                              // invisible instead of removing it.

#if SAVE_ALL_THE_TIME
    stage->Save();
#endif
}

std::string Stage::stage_content() const
{
    std::string str;
    stage->GetRootLayer()->ExportToString(&str);
    return str;
}

pxr::UsdStageRefPtr Stage::get_usd_stage() const
{
    return stage;
}

void Stage::create_editor_at_path(const pxr::SdfPath& sdf_path)
{
    create_editor_pending_path = sdf_path;
}

bool Stage::consume_editor_creation(pxr::SdfPath& json_path, bool fully_consume)
{
    if (create_editor_pending_path.IsEmpty()) {
        return false;
    }

    json_path = create_editor_pending_path;
    if (fully_consume) {
        create_editor_pending_path = pxr::SdfPath::EmptyPath();
    }
    return true;
}

void Stage::save_string_to_usd(
    const pxr::SdfPath& path,
    const std::string& data)
{
    auto prim = stage->GetPrimAtPath(path);
    if (!prim) {
        return;
    }

    auto attr = prim.CreateAttribute(
        pxr::TfToken("node_json"), pxr::SdfValueTypeNames->String);
    attr.Set(data);
#if SAVE_ALL_THE_TIME
    stage->Save();
#endif
}

std::string Stage::load_string_from_usd(const pxr::SdfPath& path)
{
    auto prim = stage->GetPrimAtPath(path);
    if (!prim) {
        return "";
    }

    auto attr = prim.GetAttribute(pxr::TfToken("node_json"));
    if (!attr) {
        return "";
    }

    std::string data;
    attr.Get(&data);
    return data;
}

void Stage::import_usd_as_payload(
    const std::string& path_string,
    const pxr::SdfPath& sdf_path)
{
    auto prim = stage->GetPrimAtPath(sdf_path);
    if (!prim) {
        return;
    }

    // bring the usd file into the stage with payload

    auto paylaods = prim.GetPayloads();
    paylaods.AddPayload(pxr::SdfPayload(path_string));
#if SAVE_ALL_THE_TIME
    stage->Save();
#endif
}

void Stage::import_usd_as_reference(
    const std::string& path_string,
    const pxr::SdfPath& sdf_path)
{
    auto prim = stage->GetPrimAtPath(sdf_path);
    if (!prim) {
        return;
    }

    // bring the usd file into the stage with reference

    auto references = prim.GetReferences();
    references.AddReference(pxr::SdfReference(path_string));
}

void Stage::import_materialx(
    const std::string& path_string,
    const pxr::SdfPath& sdf_path)
{
    MaterialX::FilePath path(path_string);
}

std::unique_ptr<Stage> create_global_stage(const std::string& usd_name)
{
    return std::make_unique<Stage>(usd_name);
}

std::unique_ptr<Stage> create_custom_global_stage(const std::string& filename)
{
    return std::make_unique<Stage>(filename);
}

bool Stage::get_prim_time_info(
    const pxr::SdfPath& path,
    pxr::UsdTimeCode& current_time,
    pxr::UsdTimeCode& render_time) const
{
    auto it = animatable_prims.find(path);
    if (it == animatable_prims.end()) {
        return false;
    }

    current_time = it->second.get_prim_current_time();
    render_time = it->second.get_prim_render_time();
    return true;
}

void Stage::set_prim_render_time(
    const pxr::SdfPath& path,
    pxr::UsdTimeCode time)
{
    auto it = animatable_prims.find(path);
    if (it != animatable_prims.end()) {
        it->second.set_prim_render_time(time);
    }
}

void Stage::Save()
{
    if (stage && !m_stage_path.empty()) {
        stage->Export(m_stage_path);
        spdlog::info("Stage saved to: {}", m_stage_path);
    }
}

void Stage::SaveAs(const std::string& new_path)
{
    if (!stage) {
        spdlog::error("No stage to save");
        return;
    }

    std::filesystem::path abs_path = std::filesystem::path(new_path).lexically_normal();
    
    // Export 会自动清理冗余数据
    if (stage->Export(abs_path.string())) {
        m_stage_path = abs_path.string();
        // Reopen the stage at the new location
        stage = pxr::UsdStage::Open(m_stage_path);
        spdlog::info("Stage saved as: {}", m_stage_path);
    } else {
        spdlog::error("Failed to save stage to: {}", abs_path.string());
    }
}

bool Stage::OpenStage(const std::string& path)
{
    std::filesystem::path abs_path = std::filesystem::path(path).lexically_normal();
    
    if (!std::filesystem::exists(abs_path)) {
        spdlog::error("Stage file does not exist: {}", abs_path.string());
        return false;
    }

    // Clear existing animatable prims
    animatable_prims.clear();
    
    // Open the new stage
    auto new_stage = pxr::UsdStage::Open(abs_path.string());
    if (!new_stage) {
        spdlog::error("Failed to open stage: {}", abs_path.string());
        return false;
    }

    // Replace the current stage
    stage = new_stage;
    m_stage_path = abs_path.string();
    
    // Reset time codes
    current_time_code = pxr::UsdTimeCode(0.0f);
    render_time_code = pxr::UsdTimeCode(0.0f);
    
    spdlog::info("Opened stage: {}", m_stage_path);
    return true;
}

RUZINO_NAMESPACE_CLOSE_SCOPE
