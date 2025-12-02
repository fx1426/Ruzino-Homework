#define IMGUI_DEFINE_MATH_OPERATORS

#include "widgets/usdtree/usd_fileviewer.h"
#include "macro_map.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <map>
#include <vector>

#include "GUI/ImGuiFileDialog.h"
#include "imgui.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/base/vt/typeHeaders.h"
#include "pxr/base/vt/visitValue.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/property.h"
#include "pxr/usd/usdGeom/xformOp.h"
#include "stage/stage.hpp"
USTC_CG_NAMESPACE_OPEN_SCOPE

// Macro for creating drag float component controls (X/Y/Z)
#define DRAG_FLOAT_COMPONENT(label, var, index, modified_flag) \
    ImGui::Text(label ":"); ImGui::SameLine(); \
    if (ImGui::DragFloat("##" label, &var, 0.1f, -1000.f, 1000.f, "%.2f")) { \
        modified_flag = true; \
    } \
    ImGui::SameLine();

// Macro for creating rotation component controls with custom range
#define DRAG_ROT_COMPONENT(label, var, index, eulerVec, modified_flag, min_val, max_val) \
    ImGui::Text(label ":"); ImGui::SameLine(); \
    if (ImGui::DragFloat("##rot_" label, &var, 1.0f, min_val, max_val, "%.1f°")) { \
        eulerVec[index] = var; \
        modified_flag = true; \
    } \
    ImGui::SameLine();

// Macro for creating scale component controls
#define DRAG_SCALE_COMPONENT(label, var, scaleVec, index, modified_flag) \
    ImGui::Text(label ":"); ImGui::SameLine(); \
    if (ImGui::DragFloat("##scale_" label, &var, 0.01f, 0.001f, 100.f, "%.3f")) { \
        scaleVec[index] = var; \
        modified_flag = true; \
    } \
    ImGui::SameLine();

// Macro for simple scalar attribute handling
#define HANDLE_SCALAR_ATTR(Type, ImGuiFunc, ...) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        if (ImGuiFunc("##value", &value, __VA_ARGS__)) { \
            attr.Set(value); \
        } \
    }

// Macro for DragScalar types
#define HANDLE_DRAGSCALAR_ATTR(Type, DataType) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        if (ImGui::DragScalar("##value", DataType, &value, 1.0f)) { \
            attr.Set(value); \
        } \
    }

// Macro for vector attribute handling with optional color editing
#define HANDLE_VEC_ATTR(Type, dim) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        if (ImGui::DragFloat##dim("##value", value.data(), 0.01f)) { \
            attr.Set(value); \
        } \
    }

// Macro for vector attribute with color support
#define HANDLE_VEC_ATTR_COLOR(Type, dim) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        if (attrName.find("color") != std::string::npos || \
            attrName.find("Color") != std::string::npos) { \
            if (ImGui::ColorEdit##dim("##value", value.data())) { \
                attr.Set(value); \
            } \
        } \
        else { \
            if (ImGui::DragFloat##dim("##value", value.data(), 0.01f)) { \
                attr.Set(value); \
            } \
        } \
    }

// Macro for double vector attribute handling with float conversion
#define HANDLE_VECD_ATTR(Type, dim) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        float tmp[dim]; \
        for (int i = 0; i < dim; ++i) tmp[i] = static_cast<float>(value[i]); \
        if (ImGui::DragFloat##dim("##value", tmp, 0.01f)) { \
            for (int i = 0; i < dim; ++i) value[i] = static_cast<double>(tmp[i]); \
            attr.Set(value); \
        } \
    }

// Macro for double vector attribute with color support
#define HANDLE_VECD_ATTR_COLOR(Type, dim) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        float tmp[dim]; \
        for (int i = 0; i < dim; ++i) tmp[i] = static_cast<float>(value[i]); \
        bool changed = false; \
        if (attrName.find("color") != std::string::npos || \
            attrName.find("Color") != std::string::npos) { \
            changed = ImGui::ColorEdit##dim("##value", tmp); \
        } \
        else { \
            changed = ImGui::DragFloat##dim("##value", tmp, 0.01f); \
        } \
        if (changed) { \
            for (int i = 0; i < dim; ++i) value[i] = static_cast<double>(tmp[i]); \
            attr.Set(value); \
        } \
    }

// Macro for integer vector attribute handling
#define HANDLE_VECI_ATTR(Type, dim) \
    else if (v.IsHolding<Type>()) { \
        Type value = v.Get<Type>(); \
        if (ImGui::DragInt##dim("##value", value.data())) { \
            attr.Set(value); \
        } \
    }

// Macro for array preview display - scalar types
#define PREVIEW_SCALAR_ARRAY(ArrType, format) \
    else if (v.IsHolding<ArrType>()) { \
        auto arr = v.Get<ArrType>(); \
        for (size_t i = 0; i < previewCount; ++i) { \
            ImGui::Text(format, i, arr[i]); \
        } \
        if (hasMore) { \
            ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6); \
            for (size_t i = arraySize - previewCount; i < arraySize; ++i) { \
                ImGui::Text(format, i, arr[i]); \
            } \
        } \
    }

// Macro for array preview display - Vec2 types
#define PREVIEW_VEC2_ARRAY(ArrType) \
    else if (v.IsHolding<ArrType>()) { \
        auto arr = v.Get<ArrType>(); \
        for (size_t i = 0; i < previewCount; ++i) { \
            ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]); \
        } \
        if (hasMore) { \
            ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6); \
            for (size_t i = arraySize - previewCount; i < arraySize; ++i) { \
                ImGui::Text("[%zu]: (%.3f, %.3f)", i, arr[i][0], arr[i][1]); \
            } \
        } \
    }

// Macro for array preview display - Vec3 types
#define PREVIEW_VEC3_ARRAY(ArrType) \
    else if (v.IsHolding<ArrType>()) { \
        auto arr = v.Get<ArrType>(); \
        for (size_t i = 0; i < previewCount; ++i) { \
            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]); \
        } \
        if (hasMore) { \
            ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6); \
            for (size_t i = arraySize - previewCount; i < arraySize; ++i) { \
                ImGui::Text("[%zu]: (%.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2]); \
            } \
        } \
    }

// Macro for array preview display - Vec4 types
#define PREVIEW_VEC4_ARRAY(ArrType) \
    else if (v.IsHolding<ArrType>()) { \
        auto arr = v.Get<ArrType>(); \
        for (size_t i = 0; i < previewCount; ++i) { \
            ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]); \
        } \
        if (hasMore) { \
            ImGui::TextDisabled("... (%zu elements omitted) ...", arraySize - 6); \
            for (size_t i = arraySize - previewCount; i < arraySize; ++i) { \
                ImGui::Text("[%zu]: (%.3f, %.3f, %.3f, %.3f)", i, arr[i][0], arr[i][1], arr[i][2], arr[i][3]); \
            } \
        } \
    }
void UsdFileViewer::ShowFileTree()
{
    auto root = stage->get_usd_stage()->GetPseudoRoot();
    ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                            ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("stage_table", 2, flags)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch);
        DrawChild(root, true);

        ImGui::EndTable();
    }
}

void UsdFileViewer::ShowPrimInfo()
{
    using namespace pxr;
    ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                            ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("table", 3, flags)) {
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn(
            "Property Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableHeadersRow();
        UsdPrim prim = stage->get_usd_stage()->GetPrimAtPath(selected);
        if (prim) {
            auto attributes = prim.GetAttributes();
            std::vector<std::future<std::string>> futures;

            for (auto&& attr : attributes) {
                futures.push_back(std::async(std::launch::async, [&attr]() {
                    VtValue v;
                    attr.Get(&v);
                    if (v.IsArrayValued()) {
                        std::string displayString;
                        auto formatArray = [&](auto array) {
                            size_t arraySize = array.size();
                            size_t displayCount = 3;
                            for (size_t i = 0;
                                 i < std::min(displayCount, arraySize);
                                 ++i) {
                                displayString += TfStringify(array[i]) + ", \n";
                            }
                            if (arraySize > 2 * displayCount) {
                                displayString += "... \n";
                            }
                            for (size_t i = std::max(
                                     displayCount, arraySize - displayCount);
                                 i < arraySize;
                                 ++i) {
                                displayString += TfStringify(array[i]) + ", \n";
                            }
                            if (!displayString.empty()) {
                                displayString.pop_back();
                                displayString.pop_back();
                                displayString.pop_back();
                            }
                        };
                        if (v.IsHolding<VtArray<double>>()) {
                            formatArray(v.Get<VtArray<double>>());
                        }
                        else if (v.IsHolding<VtArray<float>>()) {
                            formatArray(v.Get<VtArray<float>>());
                        }
                        else if (v.IsHolding<VtArray<int>>()) {
                            formatArray(v.Get<VtArray<int>>());
                        }
                        else if (v.IsHolding<VtArray<unsigned int>>()) {
                            formatArray(v.Get<VtArray<unsigned int>>());
                        }
                        else if (v.IsHolding<VtArray<int64_t>>()) {
                            formatArray(v.Get<VtArray<int64_t>>());
                        }
                        else if (v.IsHolding<VtArray<uint64_t>>()) {
                            formatArray(v.Get<VtArray<uint64_t>>());
                        }
                        else if (v.IsHolding<VtArray<GfMatrix4d>>()) {
                            formatArray(v.Get<VtArray<GfMatrix4d>>());
                        }
                        else if (v.IsHolding<VtArray<GfMatrix4f>>()) {
                            formatArray(v.Get<VtArray<GfMatrix4f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2d>>()) {
                            formatArray(v.Get<VtArray<GfVec2d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2f>>()) {
                            formatArray(v.Get<VtArray<GfVec2f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec2i>>()) {
                            formatArray(v.Get<VtArray<GfVec2i>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3d>>()) {
                            formatArray(v.Get<VtArray<GfVec3d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3f>>()) {
                            formatArray(v.Get<VtArray<GfVec3f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec3i>>()) {
                            formatArray(v.Get<VtArray<GfVec3i>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4d>>()) {
                            formatArray(v.Get<VtArray<GfVec4d>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4f>>()) {
                            formatArray(v.Get<VtArray<GfVec4f>>());
                        }
                        else if (v.IsHolding<VtArray<GfVec4i>>()) {
                            formatArray(v.Get<VtArray<GfVec4i>>());
                        }
                        else {
                            displayString = "Unsupported array type";
                        }
                        return displayString;
                    }
                    else {
                        return VtVisitValue(
                            v, [](auto&& v) { return TfStringify(v); });
                    }
                }));
            }

            auto relations = prim.GetRelationships();
            std::vector<std::future<std::string>> relation_futures;
            for (auto&& relation : relations) {
                relation_futures.push_back(
                    std::async(std::launch::async, [&relation]() {
                        std::string displayString;
                        SdfPathVector relation_targets;
                        relation.GetTargets(&relation_targets);
                        for (auto&& target : relation_targets) {
                            displayString += target.GetString() + ",\n";
                        }
                        if (!displayString.empty()) {
                            displayString.pop_back();
                            displayString.pop_back();
                        }
                        return displayString;
                    }));
            }
            auto displayRow = [](const char* type,
                                 const std::string& name,
                                 const std::string& value) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(type);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(name.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(value.c_str());
            };

            for (size_t i = 0; i < attributes.size(); ++i) {
                displayRow(
                    "A", attributes[i].GetName().GetString(), futures[i].get());
            }

            for (size_t i = 0; i < relations.size(); ++i) {
                displayRow(
                    "R",
                    relations[i].GetName().GetString(),
                    relation_futures[i].get());
            }
        }
        ImGui::EndTable();
    }
}

void UsdFileViewer::EditValue()
{
    using namespace pxr;
    UsdPrim prim = stage->get_usd_stage()->GetPrimAtPath(selected);
    if (!prim) {
        ImGui::TextDisabled("No prim selected");
        return;
    }

    // Prim Info Section
    if (ImGui::CollapsingHeader("Prim Info", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Path: %s", prim.GetPath().GetString().c_str());
        ImGui::Text("Type: %s", prim.GetTypeName().GetText());
        ImGui::Text("Active: %s", prim.IsActive() ? "Yes" : "No");
        ImGui::Separator();
    }

    // Transform controls in a collapsible section
    auto xformable = UsdGeomXformable::Get(stage->get_usd_stage(), selected);
    if (xformable) {
        if (ImGui::CollapsingHeader(
                "Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool rst_stack;
            auto xform_op = xformable.GetOrderedXformOps(&rst_stack);

            // Initialize transform if not exists
            if (xform_op.size() == 0) {
                GfMatrix4d mat = GfMatrix4d(1);
                auto trans = xformable.AddTransformOp();
                trans.Set(mat);
                xformable.SetXformOpOrder({ trans });
                xform_op = xformable.GetOrderedXformOps(&rst_stack);
            }

            if (xform_op.size() == 1 &&
                xform_op[0].GetOpType() == UsdGeomXformOp::TypeTransform) {
                auto trans = xform_op[0];
                GfMatrix4d mat;
                trans.Get(&mat);

                // Decompose matrix into translation, rotation, scale
                GfVec3d translation = mat.ExtractTranslation();

                // Check if we need to recompute transform values (new prim selected or first time)
                GfVec3d scaleVec;
                GfVec3d eulerXYZ;
                if (!has_cached_transform ||
                    cached_transform_path != selected) {
                    // Extract scale by getting the length of each basis vector
                    scaleVec = GfVec3d(
                        GfVec3d(mat[0][0], mat[0][1], mat[0][2]).GetLength(),
                        GfVec3d(mat[1][0], mat[1][1], mat[1][2]).GetLength(),
                        GfVec3d(mat[2][0], mat[2][1], mat[2][2]).GetLength());
                    // Extract rotation using GfRotation (more stable than Euler
                    // angles)
                    GfMatrix3d rotMat3 = mat.ExtractRotationMatrix();

                    // Calculate Euler angles with proper handling of edge cases
                    double sy = -rotMat3[2][0];

                    // Clamp to avoid numerical issues
                    sy = GfClamp(sy, -1.0, 1.0);

                    if (std::abs(sy) < 0.99999) {
                        // Normal case
                        eulerXYZ[0] = atan2(rotMat3[2][1], rotMat3[2][2]) *
                                      180.0 / M_PI;             // X
                        eulerXYZ[1] = asin(sy) * 180.0 / M_PI;  // Y
                        eulerXYZ[2] = atan2(rotMat3[1][0], rotMat3[0][0]) *
                                      180.0 / M_PI;  // Z
                    }
                    else {
                        // Gimbal lock case
                        eulerXYZ[0] =
                            atan2(-rotMat3[0][1], rotMat3[1][1]) * 180.0 / M_PI;
                        eulerXYZ[1] =
                            sy > 0 ? 89.9 : -89.9;  // Clamp to safe range
                        eulerXYZ[2] = 0.0;
                    }

                    // Cache the transform values
                    cached_euler_angles = eulerXYZ;
                    cached_scale = scaleVec;
                    cached_transform_path = selected;
                    has_cached_transform = true;
                }
                else {
                    // Use cached values to avoid jitter and drift
                    eulerXYZ = cached_euler_angles;
                    scaleVec = cached_scale;
                }

                bool modified = false;

                // Translation
                ImGui::Text("Translation");
                ImGui::Indent();
                float trans_tmp[3] = { static_cast<float>(translation[0]),
                                       static_cast<float>(translation[1]),
                                       static_cast<float>(translation[2]) };

                float availWidth = ImGui::GetContentRegionAvail().x;
                float itemWidth = (availWidth - 120.0f) / 3.0f;  // 120 for labels and spacing
                
                ImGui::PushItemWidth(itemWidth);
                bool transModified = false;
                DRAG_FLOAT_COMPONENT("X", trans_tmp[0], 0, transModified)
                DRAG_FLOAT_COMPONENT("Y", trans_tmp[1], 1, transModified)
                DRAG_FLOAT_COMPONENT("Z", trans_tmp[2], 2, transModified)
                ImGui::PopItemWidth();
                
                if (transModified) {
                    translation[0] = trans_tmp[0];
                    translation[1] = trans_tmp[1];
                    translation[2] = trans_tmp[2];
                    modified = true;
                }
                
                ImGui::Unindent();

                ImGui::Spacing();

                // Rotation (Euler angles in degrees)
                ImGui::Text("Rotation (degrees)");
                ImGui::Indent();
                float rot_tmp[3] = { static_cast<float>(eulerXYZ[0]),
                                     static_cast<float>(eulerXYZ[1]),
                                     static_cast<float>(eulerXYZ[2]) };

                ImGui::PushItemWidth(itemWidth);
                bool rotModified = false;

                DRAG_ROT_COMPONENT("X", rot_tmp[0], 0, eulerXYZ, rotModified, -180.f, 180.f)
                DRAG_ROT_COMPONENT("Y", rot_tmp[1], 1, eulerXYZ, rotModified, -89.9f, 89.9f)
                DRAG_ROT_COMPONENT("Z", rot_tmp[2], 2, eulerXYZ, rotModified, -180.f, 180.f)
                ImGui::PopItemWidth();
                ImGui::Unindent();

                if (rotModified) {
                    modified = true;
                    // Update cache with new values
                    cached_euler_angles = eulerXYZ;
                }

                ImGui::Spacing();

                // Scale
                ImGui::Text("Scale");
                ImGui::Indent();
                
                // Use separate variables to avoid cross-contamination
                float scale_x = static_cast<float>(scaleVec[0]);
                float scale_y = static_cast<float>(scaleVec[1]);
                float scale_z = static_cast<float>(scaleVec[2]);

                bool scaleModified = false;
                
                ImGui::PushItemWidth(itemWidth);
                DRAG_SCALE_COMPONENT("X", scale_x, scaleVec, 0, scaleModified)
                DRAG_SCALE_COMPONENT("Y", scale_y, scaleVec, 1, scaleModified)
                DRAG_SCALE_COMPONENT("Z", scale_z, scaleVec, 2, scaleModified)
                ImGui::PopItemWidth();

                // Uniform scale button
                if (ImGui::Button("Uniform Scale")) {
                    float uniformScale = (scale_x + scale_y + scale_z) / 3.0f;
                    scaleVec[0] = scaleVec[1] = scaleVec[2] = uniformScale;
                    scaleModified = true;
                }
                
                if (scaleModified) {
                    modified = true;
                    // Update cache with new scale values
                    cached_scale = scaleVec;
                }
                
                ImGui::Unindent();

                if (modified) {
                    // Reconstruct matrix from components
                    // First create scale matrix
                    GfMatrix4d scaleMat(1);
                    scaleMat[0][0] = scaleVec[0];
                    scaleMat[1][1] = scaleVec[1];
                    scaleMat[2][2] = scaleVec[2];

                    // Create rotation matrix from Euler angles using GfRotation
                    // This is more stable and avoids gimbal lock issues
                    GfRotation rotX_rotation(GfVec3d(1, 0, 0), eulerXYZ[0]);
                    GfRotation rotY_rotation(GfVec3d(0, 1, 0), eulerXYZ[1]);
                    GfRotation rotZ_rotation(GfVec3d(0, 0, 1), eulerXYZ[2]);

                    // Combine rotations: Z * Y * X (standard XYZ Euler order)
                    GfRotation combinedRotation =
                        rotZ_rotation * rotY_rotation * rotX_rotation;

                    // Convert to 3x3 matrix then to 4x4
                    GfMatrix3d rotMat3(combinedRotation.GetQuat());
                    GfMatrix4d rotMat4(1);
                    rotMat4.SetRotate(rotMat3);

                    // Combine: Scale * Rotation
                    GfMatrix4d newMat = rotMat4 * scaleMat;

                    // Set translation
                    newMat.SetTranslateOnly(translation);

                    // Verify matrix is valid before setting
                    if (!std::isnan(newMat[0][0]) &&
                        !std::isnan(newMat[3][3])) {
                        try {
                            trans.Set(newMat);
                        }
                        catch (...) {
                            spdlog::warn("Failed to set transform matrix");
                        }
                    }
                    else {
                        spdlog::warn(
                            "Invalid transform matrix detected, skipping "
                            "update");
                    }
                }
            }
            else if (xform_op.size() > 1) {
                ImGui::TextWrapped(
                    "Complex transform stack detected. Only simple transform "
                    "editing is supported.");
                ImGui::TextDisabled(
                    "Transform stack has %zu operations", xform_op.size());
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
        }
    }

    // Attributes in a collapsible section
    if (prim) {
        if (ImGui::CollapsingHeader(
                "Attributes", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto attributes = prim.GetAttributes();

            // Group attributes by category
            std::map<std::string, std::vector<UsdAttribute>> attributeGroups;
            for (auto&& attr : attributes) {
                std::string attrName = attr.GetName().GetString();

                // Skip xformOp attributes as they're handled in Transform
                // section
                if (attrName.find("xformOp") != std::string::npos) {
                    continue;
                }

                // Categorize by namespace
                std::string category = "General";
                size_t colonPos = attrName.find(':');
                if (colonPos != std::string::npos) {
                    category = attrName.substr(0, colonPos);
                }

                attributeGroups[category].push_back(attr);
            }

            // Display attributes by category
            for (auto& [category, attrs] : attributeGroups) {
                if (attrs.empty())
                    continue;

                if (ImGui::TreeNodeEx(
                        category.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (auto&& attr : attrs) {
                        VtValue v;
                        attr.Get(&v);
                        std::string attrName = attr.GetName().GetString();
                        std::string label = attrName + "##" + attrName;

                        ImGui::PushID(label.c_str());
                        ImGui::AlignTextToFramePadding();
                        ImGui::Text("%s", attrName.c_str());
                        ImGui::SameLine(200);
                        ImGui::SetNextItemWidth(-1);

                        if (v.IsHolding<double>()) {
                            double value = v.Get<double>();
                            if (ImGui::DragScalar("##value", ImGuiDataType_Double, &value, 0.01, nullptr, nullptr, "%.3f")) {
                                attr.Set(value);
                            }
                        }
                        HANDLE_SCALAR_ATTR(float, ImGui::DragFloat, 0.01f, 0.0f, 0.0f, "%.3f")
                        HANDLE_SCALAR_ATTR(int, ImGui::DragInt, 1.0f)
                        HANDLE_DRAGSCALAR_ATTR(unsigned int, ImGuiDataType_U32)
                        HANDLE_DRAGSCALAR_ATTR(int64_t, ImGuiDataType_S64)
                        HANDLE_VEC_ATTR(GfVec2f, 2)
                        HANDLE_VEC_ATTR_COLOR(GfVec3f, 3)
                        HANDLE_VEC_ATTR_COLOR(GfVec4f, 4)
                        HANDLE_VECI_ATTR(GfVec2i, 2)
                        HANDLE_VECI_ATTR(GfVec3i, 3)
                        HANDLE_VECI_ATTR(GfVec4i, 4)
                        HANDLE_VECD_ATTR(GfVec2d, 2)
                        HANDLE_VECD_ATTR_COLOR(GfVec3d, 3)
                        HANDLE_VECD_ATTR_COLOR(GfVec4d, 4)
                        else {
                            // Read-only display for unsupported types
                            if (v.IsArrayValued()) {
                                // For arrays, show type and size
                                size_t arraySize = v.GetArraySize();
                                ImGui::TextDisabled(
                                    "%s [%zu elements]",
                                    v.GetTypeName().c_str(),
                                    arraySize);

                                // Show preview of elements (first 3 and last 3)
                                size_t previewCount =
                                    std::min<size_t>(3, arraySize);
                                bool hasMore = arraySize > 6;

                                ImGui::Indent();

                                if (false) {}  // Start the else-if chain
                                PREVIEW_SCALAR_ARRAY(VtArray<float>, "[%zu]: %.3f")
                                PREVIEW_SCALAR_ARRAY(VtArray<double>, "[%zu]: %.3f")
                                PREVIEW_SCALAR_ARRAY(VtArray<int>, "[%zu]: %d")
                                PREVIEW_VEC2_ARRAY(VtArray<GfVec2f>)
                                PREVIEW_VEC2_ARRAY(VtArray<GfVec2d>)
                                PREVIEW_VEC3_ARRAY(VtArray<GfVec3f>)
                                PREVIEW_VEC3_ARRAY(VtArray<GfVec3d>)
                                PREVIEW_VEC4_ARRAY(VtArray<GfVec4f>)
                                PREVIEW_VEC4_ARRAY(VtArray<GfVec4d>)
                                else {
                                    // Unknown array type
                                    ImGui::TextDisabled(
                                        "(preview not available)");
                                }

                                ImGui::Unindent();
                            }
                            else {
                                // For scalar unsupported types, show type name
                                ImGui::TextDisabled(
                                    "[%s]", v.GetTypeName().c_str());
                            }
                        }

                        ImGui::PopID();
                    }
                    ImGui::TreePop();
                }
            }
        }
    }
}

void UsdFileViewer::select_file()
{
    auto instance = IGFD::FileDialog::Instance();
    if (instance->Display("SelectFile")) {
        auto selected = instance->GetFilePathName();
        spdlog::info(selected.c_str());

        is_selecting_file = false;

        stage->import_usd_as_payload(selected, selecting_file_base);
    }
}

int UsdFileViewer::delete_pass_id = 0;

void UsdFileViewer::remove_prim_logic()
{
    if (delete_pass_id == 3) {
        stage->remove_prim(to_delete);
    }

    if (delete_pass_id == 2) {
        stage->add_prim(to_delete);
    }

    if (delete_pass_id == 1) {
        stage->remove_prim(to_delete);
    }

    if (delete_pass_id > 0) {
        delete_pass_id--;
    }
}

void UsdFileViewer::show_right_click_menu()
{
    if (ImGui::BeginPopupContextWindow("Prim Operation")) {
        if (ImGui::BeginMenu("Create Geometry")) {
            if (ImGui::MenuItem("Mesh")) {
                stage->create_mesh(selected);
            }
            if (ImGui::MenuItem("Cylinder")) {
                stage->create_cylinder(selected);
            }
            if (ImGui::MenuItem("Sphere")) {
                stage->create_sphere(selected);
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Create Light")) {
            if (ImGui::MenuItem("Dome Light")) {
                stage->create_dome_light(selected);
            }
            if (ImGui::MenuItem("Disk Light")) {
                stage->create_disk_light(selected);
            }
            if (ImGui::MenuItem("Distant Light")) {
                stage->create_distant_light(selected);
            }
            if (ImGui::MenuItem("Rect Light")) {
                stage->create_rect_light(selected);
            }
            if (ImGui::MenuItem("Sphere Light")) {
                stage->create_sphere_light(selected);
            }
            ImGui::EndMenu();
        }

        // create_material
        if (ImGui::BeginMenu("Create Material")) {
            if (ImGui::MenuItem("Material")) {
                stage->create_material(selected);
            }
            if (ImGui::MenuItem("Scratch Material")) {
                auto material = stage->create_material(selected);
            }

            ImGui::EndMenu();
        }

        if (selected != pxr::SdfPath("/")) {
            if (ImGui::MenuItem("Import...")) {
                is_selecting_file = true;
                selecting_file_base = selected;
            }
            if (ImGui::MenuItem("Edit")) {
                stage->create_editor_at_path(selected);
            }

            if (ImGui::MenuItem("Delete")) {
                to_delete = selected;
                delete_pass_id = 3;
            }
        }

        ImGui::EndPopup();
    }
}

void UsdFileViewer::DrawChild(const pxr::UsdPrim& prim, bool is_root)
{
    auto flags =
        ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
    if (is_root) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    bool is_leaf = prim.GetChildren().empty();
    if (is_leaf) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet |
                 ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    if (prim.GetPath() == selected) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    bool open = ImGui::TreeNodeEx(prim.GetName().GetText(), flags);

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        selected = prim.GetPath();
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        selected = prim.GetPath();
        ImGui::OpenPopup("Prim Operation");
    }

    ImGui::TableNextColumn();
    ImGui::TextUnformatted(prim.GetTypeName().GetText());

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        selected = prim.GetPath();
    }

    if (!is_leaf) {
        if (open) {
            for (const pxr::UsdPrim& child : prim.GetChildren()) {
                DrawChild(child);
            }

            ImGui::TreePop();
        }
    }

    if (prim.GetPath() == selected) {
        show_right_click_menu();
    }
    if (is_selecting_file) {
        select_file();
    }
}

bool UsdFileViewer::BuildUI()
{
    ImGui::Begin("Stage Viewer", nullptr, ImGuiWindowFlags_None);
    ShowFileTree();
    ImGui::End();

    ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_None);
    EditValue();
    ImGui::End();

    remove_prim_logic();

    return true;
}

UsdFileViewer::UsdFileViewer(Stage* stage) : stage(stage)
{
}

UsdFileViewer::~UsdFileViewer()
{
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
