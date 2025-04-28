#include "GCore/Components/MaterialComponent.h"

#include <filesystem>

USTC_CG_NAMESPACE_OPEN_SCOPE
pxr::UsdShadeMaterial MaterialComponent::define_material(
    pxr::UsdStageRefPtr stage,
    pxr::SdfPath path)
{
    auto material_path = get_material_path();
    auto texture_name = std::string(textures[0].c_str());

    auto material_shader_path =
        material_path.AppendPath(pxr::SdfPath("PBRShader"));
    auto material_stReader_path =
        material_path.AppendPath(pxr::SdfPath("stReader"));
    auto material_texture_path =
        material_path.AppendPath(pxr::SdfPath("diffuseTexture"));
    pxr::UsdShadeMaterial material =
        pxr::UsdShadeMaterial::Define(stage, material_path);
    auto pbrShader = pxr::UsdShadeShader::Define(stage, material_shader_path);

    pbrShader.CreateIdAttr(pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
    material.CreateSurfaceOutput().ConnectToSource(
        pbrShader.ConnectableAPI(), pxr::TfToken("surface"));

    auto stReader = pxr::UsdShadeShader::Define(stage, material_stReader_path);
    stReader.CreateIdAttr(
        pxr::VtValue(pxr::TfToken("UsdPrimvarReader_float2")));

    auto diffuseTextureSampler =
        pxr::UsdShadeShader::Define(stage, material_texture_path);

    diffuseTextureSampler.CreateIdAttr(
        pxr::VtValue(pxr::TfToken("UsdUVTexture")));
    diffuseTextureSampler
        .CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(texture_name));
    diffuseTextureSampler
        .CreateInput(pxr::TfToken("st"), pxr::SdfValueTypeNames->Float2)
        .ConnectToSource(stReader.ConnectableAPI(), pxr::TfToken("result"));
    diffuseTextureSampler.CreateOutput(
        pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);

    diffuseTextureSampler
        .CreateInput(pxr::TfToken("wrapS"), pxr::SdfValueTypeNames->Token)
        .Set(pxr::TfToken("mirror"));

    diffuseTextureSampler
        .CreateInput(pxr::TfToken("wrapT"), pxr::SdfValueTypeNames->Token)
        .Set(pxr::TfToken("mirror"));

    pbrShader
        .CreateInput(
            pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .ConnectToSource(
            diffuseTextureSampler.ConnectableAPI(), pxr::TfToken("rgb"));

    auto stInput = material.CreateInput(
        pxr::TfToken("frame:stPrimvarName"), pxr::SdfValueTypeNames->Token);
    stInput.Set(pxr::TfToken("UVMap"));

    stReader.CreateInput(pxr::TfToken("varname"), pxr::SdfValueTypeNames->Token)
        .ConnectToSource(stInput);

    return material;
}

pxr::SdfPath MaterialComponent::get_material_path()
{
    auto texture_name = std::string(textures[0].c_str());
    std::filesystem::path p =
        std::filesystem::path(texture_name).replace_extension();
    auto file_name = "texture" + p.filename().string();
    auto material_path_root = pxr::SdfPath("/TexModel");
    auto material_path =
        material_path_root.AppendPath(pxr::SdfPath(file_name + "Mat"));
    return material_path;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
