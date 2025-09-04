#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

using namespace USTC_CG;

// Colormap functions
glm::vec3 viridis_colormap(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);
    const glm::vec3 c0(0.2777273f, 0.005407344f, 0.3340998f);
    const glm::vec3 c1(0.1050930f, 1.404613f, 1.384590f);
    const glm::vec3 c2(-0.3308618f, 0.214847f, 0.09509516f);
    const glm::vec3 c3(-4.634230f, -5.799100f, -19.33244f);
    const glm::vec3 c4(6.228269f, 14.17993f, 56.69055f);
    const glm::vec3 c5(4.776384f, -13.74514f, -65.35303f);
    const glm::vec3 c6(-5.435455f, 4.645852f, 26.31280f);

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

glm::vec3 plasma_colormap(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);
    const glm::vec3 c0(
        0.05873234392399702f, 0.02333670892565664f, 0.5433401826748754f);
    const glm::vec3 c1(
        2.176514634195958f, 0.2383834171260182f, 0.7539604599784036f);
    const glm::vec3 c2(
        -2.689460476458034f, -7.455851135738909f, 3.110799939717086f);
    const glm::vec3 c3(
        6.130348345893603f, 42.3461881477227f, -28.51885465332158f);
    const glm::vec3 c4(
        -11.10743619062271f, -82.66631109428045f, 60.13984767418263f);
    const glm::vec3 c5(
        10.02306557647065f, 71.41361770095349f, -54.07218655560067f);
    const glm::vec3 c6(
        -3.658713842777788f, -22.93153465461149f, 18.19190778040903f);

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

glm::vec3 hot_colormap(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);
    if (t < 1.0f / 3.0f) {
        return glm::vec3(3.0f * t, 0.0f, 0.0f);
    }
    else if (t < 2.0f / 3.0f) {
        return glm::vec3(1.0f, 3.0f * (t - 1.0f / 3.0f), 0.0f);
    }
    else {
        return glm::vec3(1.0f, 1.0f, 3.0f * (t - 2.0f / 3.0f));
    }
}

glm::vec3 cool_colormap(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);
    return glm::vec3(t, 1.0f - t, 1.0f);
}

glm::vec3 jet_colormap(float t)
{
    t = glm::clamp(t, 0.0f, 1.0f);
    if (t < 0.25f) {
        return glm::vec3(0.0f, 4.0f * t, 1.0f);
    }
    else if (t < 0.5f) {
        return glm::vec3(0.0f, 1.0f, 1.0f - 4.0f * (t - 0.25f));
    }
    else if (t < 0.75f) {
        return glm::vec3(4.0f * (t - 0.5f), 1.0f, 0.0f);
    }
    else {
        return glm::vec3(1.0f, 1.0f - 4.0f * (t - 0.75f), 0.0f);
    }
}

glm::vec3 apply_colormap(float value, int colormap_type)
{
    switch (colormap_type) {
        case 0: return viridis_colormap(value);
        case 1: return plasma_colormap(value);
        case 2: return hot_colormap(value);
        case 3: return cool_colormap(value);
        case 4: return jet_colormap(value);
        default: return viridis_colormap(value);
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(colormap)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Scalar Name").default_val("result");
    b.add_input<int>("Colormap Type")
        .default_val(0)
        .min(0)
        .max(4);  // 0: viridis, 1: plasma, 2: hot, 3: cool, 4: jet
    b.add_input<bool>("Auto Range").default_val(true);
    b.add_input<float>("Min Value").default_val(0.0f);
    b.add_input<float>("Max Value").default_val(1.0f);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(colormap)
{
    // 获取输入参数
    Geometry input_geometry = params.get_input<Geometry>("Geometry");
    std::string scalar_name = params.get_input<std::string>("Scalar Name");
    int colormap_type = params.get_input<int>("Colormap Type");
    bool auto_range = params.get_input<bool>("Auto Range");
    float min_value = params.get_input<float>("Min Value");
    float max_value = params.get_input<float>("Max Value");

    // 获取网格组件
    auto mesh_component = input_geometry.get_component<MeshComponent>();
    if (!mesh_component) {
        return false;
    }

    // 获取标量数据
    std::vector<float> scalar_data =
        mesh_component->get_vertex_scalar_quantity(scalar_name);
    if (scalar_data.empty()) {
        return false;
    }

    // 计算数据范围
    if (auto_range) {
        auto minmax =
            std::minmax_element(scalar_data.begin(), scalar_data.end());
        min_value = *minmax.first;
        max_value = *minmax.second;
    }

    // 避免除零
    if (std::abs(max_value - min_value) < 1e-8f) {
        max_value = min_value + 1.0f;
    }

    // 生成颜色数据
    std::vector<glm::vec3> colors;
    colors.reserve(scalar_data.size());

    for (float value : scalar_data) {
        // 归一化到[0,1]
        float normalized = (value - min_value) / (max_value - min_value);
        glm::vec3 color = apply_colormap(normalized, colormap_type);
        colors.push_back(color);
    }

    // 添加颜色量到网格
    mesh_component->set_display_color(colors);

    // 输出结果
    params.set_output("Geometry", input_geometry);
    return true;
}

NODE_DECLARATION_UI(colormap);

NODE_DEF_CLOSE_SCOPE
