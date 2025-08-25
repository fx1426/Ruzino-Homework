
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"
#include "tetgen.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(tetgen_tetrahedralize)
{
    b.add_input<Geometry>("Surface Mesh");
    b.add_input<float>("Quality Ratio").min(1.0f).max(10.0f).default_val(2.0f);
    b.add_input<float>("Max Volume").min(0.001f).max(1000.0f).default_val(1.0f);
    b.add_input<bool>("Refine").default_val(true);
    b.add_input<bool>("Conforming Delaunay").default_val(true);
    b.add_output<Geometry>("Tetrahedral Mesh");
}

NODE_EXECUTION_FUNCTION(tetgen_tetrahedralize)
{
    auto geometry = params.get_input<Geometry>("Surface Mesh");
    float quality_ratio = params.get_input<float>("Quality Ratio");
    float max_volume = params.get_input<float>("Max Volume");
    bool refine = params.get_input<bool>("Refine");
    bool conforming = params.get_input<bool>("Conforming Delaunay");
    geometry.apply_transform();
    auto mesh_component = geometry.get_component<MeshComponent>();
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

    // Prepare TetGen input
    tetgenio input, output;
    tetgenbehavior behavior;

    // Set up points
    input.numberofpoints = vertices.size();
    input.pointlist = new REAL[input.numberofpoints * 3];

    for (size_t i = 0; i < vertices.size(); ++i) {
        input.pointlist[i * 3] = vertices[i][0];
        input.pointlist[i * 3 + 1] = vertices[i][1];
        input.pointlist[i * 3 + 2] = vertices[i][2];
    }

    // Set up facets (only triangles supported for now)
    input.numberoffacets = 0;
    for (size_t i = 0; i < face_counts.size(); ++i) {
        if (face_counts[i] == 3) {
            input.numberoffacets++;
        }
        else if (face_counts[i] != 3) {
            params.set_error(
                "TetGen only supports triangular faces. Please triangulate the "
                "mesh first.");
            delete[] input.pointlist;
            return false;
        }
    }

    input.facetlist = new tetgenio::facet[input.numberoffacets];
    input.facetmarkerlist = new int[input.numberoffacets];

    size_t face_idx = 0;
    size_t vertex_offset = 0;

    for (size_t i = 0; i < face_counts.size(); ++i) {
        if (face_counts[i] == 3) {
            tetgenio::facet& f = input.facetlist[face_idx];
            f.numberofpolygons = 1;
            f.polygonlist = new tetgenio::polygon[1];
            f.numberofholes = 0;
            f.holelist = nullptr;

            tetgenio::polygon& p = f.polygonlist[0];
            p.numberofvertices = 3;
            p.vertexlist = new int[3];

            p.vertexlist[0] = indices[vertex_offset];
            p.vertexlist[1] = indices[vertex_offset + 1];
            p.vertexlist[2] = indices[vertex_offset + 2];

            input.facetmarkerlist[face_idx] = 1;
            face_idx++;
        }
        vertex_offset += face_counts[i];
    }

    // Set TetGen behavior
    behavior.plc = 1;  // Piecewise Linear Complex
    behavior.quality = refine ? 1 : 0;
    behavior.minratio = quality_ratio;
    behavior.fixedvolume = 1;
    behavior.maxvolume = max_volume;
    behavior.quiet = 1;  // Suppress output

    if (conforming) {
        behavior.cdt = 1;  // Conforming Delaunay
    }

    try {
        // Run TetGen
        tetrahedralize(&behavior, &input, &output);

        if (output.numberoftetrahedra == 0) {
            params.set_error("TetGen failed to generate tetrahedra");
            return false;
        }

        // Create output geometry
        Geometry output_geometry;
        std::shared_ptr<MeshComponent> output_mesh =
            std::make_shared<MeshComponent>(&output_geometry);
        output_geometry.attach_component(output_mesh);

        // Convert output points
        std::vector<glm::vec3> output_points;
        output_points.reserve(output.numberofpoints);

        for (int i = 0; i < output.numberofpoints; ++i) {
            output_points.push_back(glm::vec3(
                output.pointlist[i * 3],
                output.pointlist[i * 3 + 1],
                output.pointlist[i * 3 + 2]));
        }

        // Convert tetrahedra to mesh faces (surface of tetrahedra)
        std::vector<int> output_indices;
        std::vector<int> output_face_counts;

        // Each tetrahedron has 4 triangular faces
        for (int i = 0; i < output.numberoftetrahedra; ++i) {
            int* tet = &output.tetrahedronlist[i * 4];

            // Face 0: vertices 1, 0, 2 (counter-clockwise when viewed from
            // outside)
            output_face_counts.push_back(3);
            output_indices.push_back(tet[1]);
            output_indices.push_back(tet[0]);
            output_indices.push_back(tet[2]);

            // Face 1: vertices 0, 1, 3
            output_face_counts.push_back(3);
            output_indices.push_back(tet[0]);
            output_indices.push_back(tet[1]);
            output_indices.push_back(tet[3]);

            // Face 2: vertices 2, 3, 1
            output_face_counts.push_back(3);
            output_indices.push_back(tet[2]);
            output_indices.push_back(tet[3]);
            output_indices.push_back(tet[1]);

            // Face 3: vertices 3, 2, 0
            output_face_counts.push_back(3);
            output_indices.push_back(tet[3]);
            output_indices.push_back(tet[2]);
            output_indices.push_back(tet[0]);
        }

        // Calculate normals for the faces
        std::vector<glm::vec3> normals;
        normals.reserve(output_indices.size());

        for (size_t i = 0; i < output_face_counts.size(); ++i) {
            size_t idx_start = i * 3;
            int i0 = output_indices[idx_start];
            int i1 = output_indices[idx_start + 1];
            int i2 = output_indices[idx_start + 2];

            glm::vec3 v0 = output_points[i0];
            glm::vec3 v1 = output_points[i1];
            glm::vec3 v2 = output_points[i2];

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 normal = glm::cross(edge1, edge2);
            normal = glm::normalize(normal);

            normals.push_back(normal);
            normals.push_back(normal);
            normals.push_back(normal);
        }

        // Set mesh data
        output_mesh->set_vertices(output_points);
        output_mesh->set_face_vertex_indices(output_indices);
        output_mesh->set_face_vertex_counts(output_face_counts);
        output_mesh->set_normals(normals);

        params.set_output("Tetrahedral Mesh", std::move(output_geometry));
    }
    catch (const std::exception& e) {
        params.set_error((std::string("TetGen error: ") + e.what()).c_str());
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(tetgen_tetrahedralize);

NODE_DEF_CLOSE_SCOPE
