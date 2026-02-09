#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "GCore/Components/MeshComponent.h"
#include "GCore/geom_payload.hpp"
#include "nodes/core/def/node_def.hpp"
#include "spdlog/spdlog.h"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

// Vertex to Face Average
NODE_DECLARATION_FUNCTION(vertex_to_face_average)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Source Attribute").default_val("eigenfunction");
    b.add_input<std::string>("Target Attribute")
        .default_val("face_eigenfunction");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(vertex_to_face_average)
{
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    std::string source_attr = params.get_input<std::string>("Source Attribute");
    std::string target_attr = params.get_input<std::string>("Target Attribute");

    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    std::vector<float> vertex_values =
        mesh_component->get_vertex_scalar_quantity(source_attr);

    if (vertex_values.empty()) {
        spdlog::error("Source vertex attribute '{}' not found", source_attr);
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return false;
    }

    std::vector<int> face_vertex_counts =
        mesh_component->get_face_vertex_counts();
    std::vector<int> face_vertex_indices =
        mesh_component->get_face_vertex_indices();

    size_t num_faces = face_vertex_counts.size();
    std::vector<float> face_values;
    face_values.reserve(num_faces);

    size_t index_offset = 0;
    for (size_t face_idx = 0; face_idx < num_faces; ++face_idx) {
        int vertex_count = face_vertex_counts[face_idx];

        float sum = 0.0f;
        for (int i = 0; i < vertex_count; ++i) {
            int vertex_idx = face_vertex_indices[index_offset + i];
            if (vertex_idx >= 0 && vertex_idx < vertex_values.size()) {
                sum += vertex_values[vertex_idx];
            }
        }
        face_values.push_back(sum / static_cast<float>(vertex_count));
        index_offset += vertex_count;
    }

    mesh_component->add_face_scalar_quantity(target_attr, face_values);
    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(vertex_to_face_average);

// Vertex to Face And
NODE_DECLARATION_FUNCTION(vertex_to_face_and)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Source Attribute").default_val("nn_dirichlet");
    b.add_input<std::string>("Target Attribute").default_val("face_dirichlet");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(vertex_to_face_and)
{
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    std::string source_attr = params.get_input<std::string>("Source Attribute");
    std::string target_attr = params.get_input<std::string>("Target Attribute");

    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    std::vector<float> vertex_values =
        mesh_component->get_vertex_scalar_quantity(source_attr);

    if (vertex_values.empty()) {
        spdlog::error("Source vertex attribute '{}' not found", source_attr);
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return false;
    }

    std::vector<int> face_vertex_counts =
        mesh_component->get_face_vertex_counts();
    std::vector<int> face_vertex_indices =
        mesh_component->get_face_vertex_indices();

    size_t num_faces = face_vertex_counts.size();
    std::vector<float> face_values;
    face_values.reserve(num_faces);

    size_t index_offset = 0;
    for (size_t face_idx = 0; face_idx < num_faces; ++face_idx) {
        int vertex_count = face_vertex_counts[face_idx];

        float face_value = 1.0f;
        for (int i = 0; i < vertex_count; ++i) {
            int vertex_idx = face_vertex_indices[index_offset + i];
            if (vertex_idx >= 0 && vertex_idx < vertex_values.size()) {
                if (vertex_values[vertex_idx] <= 0.5f) {
                    face_value = 0.0f;
                    break;
                }
            }
        }
        face_values.push_back(face_value);
        index_offset += vertex_count;
    }

    mesh_component->add_face_scalar_quantity(target_attr, face_values);
    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(vertex_to_face_and);

// Vertex to Face Or
NODE_DECLARATION_FUNCTION(vertex_to_face_or)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Source Attribute").default_val("nn_dirichlet");
    b.add_input<std::string>("Target Attribute").default_val("face_dirichlet");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(vertex_to_face_or)
{
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    std::string source_attr = params.get_input<std::string>("Source Attribute");
    std::string target_attr = params.get_input<std::string>("Target Attribute");

    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    std::vector<float> vertex_values =
        mesh_component->get_vertex_scalar_quantity(source_attr);

    if (vertex_values.empty()) {
        spdlog::error("Source vertex attribute '{}' not found", source_attr);
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return false;
    }

    std::vector<int> face_vertex_counts =
        mesh_component->get_face_vertex_counts();
    std::vector<int> face_vertex_indices =
        mesh_component->get_face_vertex_indices();

    size_t num_faces = face_vertex_counts.size();
    std::vector<float> face_values;
    face_values.reserve(num_faces);

    size_t index_offset = 0;
    for (size_t face_idx = 0; face_idx < num_faces; ++face_idx) {
        int vertex_count = face_vertex_counts[face_idx];

        float face_value = 0.0f;
        for (int i = 0; i < vertex_count; ++i) {
            int vertex_idx = face_vertex_indices[index_offset + i];
            if (vertex_idx >= 0 && vertex_idx < vertex_values.size()) {
                if (vertex_values[vertex_idx] > 0.5f) {
                    face_value = 1.0f;
                    break;
                }
            }
        }
        face_values.push_back(face_value);
        index_offset += vertex_count;
    }

    mesh_component->add_face_scalar_quantity(target_attr, face_values);
    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(vertex_to_face_or);

NODE_DEF_CLOSE_SCOPE
