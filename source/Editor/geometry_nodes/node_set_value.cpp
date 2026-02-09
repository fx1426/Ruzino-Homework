#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "fem_bem/Expression.hpp"
#include "nodes/core/def/node_def.hpp"
#include "spdlog/spdlog.h"

using namespace Ruzino;
using namespace Ruzino::fem_bem;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(set_value)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Expression").default_val("x + y + z");
    b.add_input<std::string>("Result Name").default_val("result");
    b.add_input<int>("Attribute Domain")
        .default_val(0)
        .min(0)
        .max(1);  // 0: vertex, 1: face

    b.add_input_group<float>("Params").set_runtime_dynamic(true);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(set_value)
{
    // 获取输入参数
    Geometry input_geometry = params.get_input<Geometry>("Geometry");

    input_geometry.apply_transform();
    std::string expression_str =
        params.get_input<std::string>("Expression").c_str();
    std::string result_name = params.get_input<std::string>("Result Name");
    int attribute_domain = params.get_input<int>("Attribute Domain");

    // 获取网格组件
    auto mesh_component = input_geometry.get_component<MeshComponent>();
    if (!mesh_component) {
        return false;
    }

    // 创建表达式
    Expression expr;
    try {
        expr = Expression::from_string(expression_str);
    }
    catch (const std::exception& e) {
        spdlog::error("Error parsing expression: {}", e.what());
        return false;
    }

    // Get dynamic parameters from Params group
    auto param_group = params.get_input_group<float>("Params");

    ParameterMap<float> params_map;
    std::vector<float> values;

    if (attribute_domain == 0) {
        // Vertex domain
        std::vector<glm::vec3> vertices = mesh_component->get_vertices();
        if (vertices.empty()) {
            return false;
        }

        // Get all existing vertex scalar quantities
        auto scalar_names = mesh_component->get_vertex_scalar_quantity_names();
        std::map<std::string, std::vector<float>> existing_quantities;
        for (const auto& name : scalar_names) {
            existing_quantities[name] =
                mesh_component->get_vertex_scalar_quantity(name);
            spdlog::info("Existing vertex scalar quantity: {}", name);
        }

        values.reserve(vertices.size());
        for (size_t vertex_idx = 0; vertex_idx < vertices.size();
             ++vertex_idx) {
            const auto& vertex = vertices[vertex_idx];
            params_map.clear();
            params_map.insert_unchecked("x", vertex.x);
            params_map.insert_unchecked("y", vertex.y);
            params_map.insert_unchecked("z", vertex.z);

            // Add existing scalar quantities
            for (const auto& [name, quantity_values] : existing_quantities) {
                if (vertex_idx < quantity_values.size()) {
                    params_map.insert_unchecked(
                        name.c_str(), quantity_values[vertex_idx]);
                }
            }

            // Add dynamic parameters as p0, p1, p2...
            for (size_t i = 0; i < param_group.size(); ++i) {
                std::string param_name = "p" + std::to_string(i);
                params_map.insert_unchecked(param_name.c_str(), param_group[i]);
            }

            try {
                float value = expr.evaluate_at(params_map);
                values.push_back(value);
            }
            catch (const std::exception& e) {
                spdlog::error(
                    "Error evaluating expression at vertex: {}", e.what());
                return false;
            }
        }
        mesh_component->add_vertex_scalar_quantity(result_name, values);
    }
    else {
        // Face domain
        std::vector<glm::vec3> vertices = mesh_component->get_vertices();
        std::vector<int> indices = mesh_component->get_face_vertex_indices();
        if (vertices.empty() || indices.empty()) {
            return false;
        }

        size_t face_count = indices.size() / 3;

        // Get all existing face scalar quantities
        auto scalar_names = mesh_component->get_face_scalar_quantity_names();
        std::map<std::string, std::vector<float>> existing_quantities;
        for (const auto& name : scalar_names) {
            existing_quantities[name] =
                mesh_component->get_face_scalar_quantity(name);
        }

        values.reserve(face_count);

        for (size_t i = 0; i < face_count; ++i) {
            // Calculate face center
            glm::vec3 v0 = vertices[indices[i * 3 + 0]];
            glm::vec3 v1 = vertices[indices[i * 3 + 1]];
            glm::vec3 v2 = vertices[indices[i * 3 + 2]];
            glm::vec3 center = (v0 + v1 + v2) / 3.0f;

            params_map.clear();
            params_map.insert_unchecked("x", center.x);
            params_map.insert_unchecked("y", center.y);
            params_map.insert_unchecked("z", center.z);

            // Add existing scalar quantities
            for (const auto& [name, quantity_values] : existing_quantities) {
                if (i < quantity_values.size()) {
                    params_map.insert_unchecked(
                        name.c_str(), quantity_values[i]);
                }
            }

            // Add dynamic parameters as p0, p1, p2...
            for (size_t j = 0; j < param_group.size(); ++j) {
                std::string param_name = "p" + std::to_string(j);
                params_map.insert_unchecked(param_name.c_str(), param_group[j]);
            }

            try {
                float value = expr.evaluate_at(params_map);
                values.push_back(value);
            }
            catch (const std::exception& e) {
                spdlog::error(
                    "Error evaluating expression at face: {}", e.what());
                return false;
                values.push_back(0.0f);
            }
        }
        mesh_component->add_face_scalar_quantity(result_name, values);
    }

    // 输出结果
    params.set_output("Geometry", input_geometry);
    return true;
}

NODE_DECLARATION_UI(set_value);

NODE_DEF_CLOSE_SCOPE