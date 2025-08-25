#include <igl/boundary_loop.h>
#include <igl/topological_hole_fill.h>

#include <Eigen/Core>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(fill_hole)
{
    b.add_input<Geometry>("Mesh");
    b.add_input<bool>("Fill All Holes").default_val(true);
    b.add_output<Geometry>("Filled Mesh");
}

NODE_EXECUTION_FUNCTION(fill_hole)
{
    auto input_geom = params.get_input<Geometry>("Mesh");
    bool fill_all = params.get_input<bool>("Fill All Holes");

    input_geom.apply_transform();

    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        params.set_error("No mesh component found in input geometry");
        return false;
    }

    // Get mesh data
    const auto& vertices = mesh_component->get_vertices();
    const auto& indices = mesh_component->get_face_vertex_indices();
    const auto& face_counts = mesh_component->get_face_vertex_counts();

    if (vertices.empty() || indices.empty()) {
        params.set_error("Input mesh is empty");
        return false;
    }

    // Check if mesh is triangulated
    for (size_t i = 0; i < face_counts.size(); ++i) {
        if (face_counts[i] != 3) {
            params.set_error(
                "Hole filling only supports triangular meshes. Please "
                "triangulate first.");
            return false;
        }
    }

    try {
        // Convert to Eigen matrices for igl
        Eigen::MatrixXd V(vertices.size(), 3);
        Eigen::MatrixXi F(face_counts.size(), 3);

        // Fill vertex matrix
        for (size_t i = 0; i < vertices.size(); ++i) {
            V(i, 0) = vertices[i][0];
            V(i, 1) = vertices[i][1];
            V(i, 2) = vertices[i][2];
        }

        // Fill face matrix
        for (size_t i = 0; i < face_counts.size(); ++i) {
            F(i, 0) = indices[i * 3];
            F(i, 1) = indices[i * 3 + 1];
            F(i, 2) = indices[i * 3 + 2];
        }

        // Find boundary loops (holes)
        std::vector<std::vector<int>> boundary_loops;
        igl::boundary_loop(F, boundary_loops);

        if (boundary_loops.empty()) {
            // No holes found, return original mesh
            params.set_output("Filled Mesh", std::move(input_geom));
            return true;
        }  // Fill holes using topological hole fill
        Eigen::MatrixXi F_filled;
        if (fill_all) {
            // Fill all holes
            igl::topological_hole_fill(F, boundary_loops, F_filled);
        }
        else {
            // Fill only the first hole found
            std::vector<std::vector<int>> single_hole = { boundary_loops[0] };
            igl::topological_hole_fill(F, single_hole, F_filled);
        }

        // Create output geometry
        Geometry output_geometry;
        std::shared_ptr<MeshComponent> output_mesh =
            std::make_shared<MeshComponent>(&output_geometry);
        output_geometry.attach_component(output_mesh);

        // Find the maximum vertex index in F_filled to determine how many
        // vertices we need
        int max_vertex_index = 0;
        for (int i = 0; i < F_filled.rows(); ++i) {
            for (int j = 0; j < F_filled.cols(); ++j) {
                max_vertex_index = std::max(max_vertex_index, F_filled(i, j));
            }
        }
        // We need max_vertex_index + 1 vertices
        int required_vertices = max_vertex_index + 1;

        // Debug info
        // printf("Original vertices: %d, Required vertices: %d, Max vertex
        // index: %d\n",
        //        (int)V.rows(), required_vertices, max_vertex_index);

        // Convert back to pxr format
        std::vector<glm::vec3> output_vertices;
        output_vertices.reserve(required_vertices);

        // Copy original vertices
        for (int i = 0; i < V.rows(); ++i) {
            output_vertices.push_back(glm::vec3(V(i, 0), V(i, 1), V(i, 2)));
        }
        // Add new vertices for hole filling (these are the abstract vertices
        // mentioned in the igl doc)
        // According to igl documentation, new vertices are added with indices
        // F.maxCoeff() + (hole index)
        for (int i = V.rows(); i < required_vertices; ++i) {
            // Find which hole this vertex belongs to and calculate its position
            // For now, we'll place it at the centroid of the hole boundary
            int hole_index = i - V.rows();
            if (hole_index < static_cast<int>(boundary_loops.size())) {
                const auto& hole = boundary_loops[hole_index];
                glm::vec3 centroid(0.0f, 0.0f, 0.0f);
                for (int vertex_idx : hole) {
                    if (vertex_idx >= 0 &&
                        vertex_idx < static_cast<int>(V.rows())) {
                        centroid += glm::vec3(
                            V(vertex_idx, 0),
                            V(vertex_idx, 1),
                            V(vertex_idx, 2));
                    }
                }
                if (!hole.empty()) {
                    centroid /= static_cast<float>(hole.size());
                }
                output_vertices.push_back(centroid);
            }
            else {
                // Fallback: place at origin
                output_vertices.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
            }
        }
        std::vector<int> output_indices;
        std::vector<int> output_face_counts;

        output_indices.reserve(F_filled.rows() * 3);
        output_face_counts.reserve(F_filled.rows());

        // Verify all indices are valid before adding faces
        for (int i = 0; i < F_filled.rows(); ++i) {
            bool valid_face = true;
            for (int j = 0; j < F_filled.cols(); ++j) {
                if (F_filled(i, j) >=
                        static_cast<int>(output_vertices.size()) ||
                    F_filled(i, j) < 0) {
                    valid_face = false;
                    break;
                }
            }

            if (valid_face) {
                output_face_counts.push_back(3);
                output_indices.push_back(F_filled(i, 0));
                output_indices.push_back(F_filled(i, 1));
                output_indices.push_back(F_filled(i, 2));
            }
        }

        // Calculate normals
        std::vector<glm::vec3> normals;
        normals.reserve(output_indices.size());

        for (size_t i = 0; i < output_face_counts.size(); ++i) {
            size_t idx_start = i * 3;
            int i0 = output_indices[idx_start];
            int i1 = output_indices[idx_start + 1];
            int i2 = output_indices[idx_start + 2];

            glm::vec3 v0 = output_vertices[i0];
            glm::vec3 v1 = output_vertices[i1];
            glm::vec3 v2 = output_vertices[i2];

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 normal = normalize(glm::cross(edge1, edge2));

            normals.push_back(normal);
            normals.push_back(normal);
            normals.push_back(normal);
        }

        // Copy texture coordinates if they exist
        const auto& original_texcoords = mesh_component->get_texcoords_array();
        if (!original_texcoords.empty() &&
            original_texcoords.size() == vertices.size()) {
            std::vector<glm::vec2> output_texcoords;
            output_texcoords.reserve(output_vertices.size());

            // Copy existing texture coordinates
            for (size_t i = 0; i < vertices.size(); ++i) {
                output_texcoords.push_back(original_texcoords[i]);
            }

            // Add default texture coordinates for new vertices (if any)
            for (size_t i = vertices.size(); i < output_vertices.size(); ++i) {
                output_texcoords.push_back(glm::vec2(0.5f, 0.5f));
            }

            output_mesh->set_texcoords_array(output_texcoords);
        }

        // Set mesh data
        output_mesh->set_vertices(output_vertices);
        output_mesh->set_face_vertex_indices(output_indices);
        output_mesh->set_face_vertex_counts(output_face_counts);
        output_mesh->set_normals(normals);

        // Copy other components from original geometry
        for (const auto& component : input_geom.get_components()) {
            if (!std::dynamic_pointer_cast<MeshComponent>(component)) {
                output_geometry.attach_component(component);
            }
        }

        params.set_output("Filled Mesh", std::move(output_geometry));
    }
    catch (const std::exception& e) {
        params.set_error(
            (std::string("Hole filling error: ") + e.what()).c_str());
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(fill_hole);

NODE_DEF_CLOSE_SCOPE
