#ifdef GEOM_USD_EXTENSION

#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/VolumeComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(sdf2mesh)
{
    b.add_input<Geometry>("SDF");
    b.add_input<float>("isovalue").default_val(0.0f).min(-1.0f).max(1.0f);
    b.add_input<float>("adaptivity").min(0.0f).max(1.0f).default_val(0.0f);
    b.add_output<Geometry>("M");
}

NODE_EXECUTION_FUNCTION(sdf2mesh)
{
    auto sdf_geometry = params.get_input<Geometry>("SDF");
    auto volume_component = sdf_geometry.get_component<VolumeComponent>();

    // Check if we have a valid volume
    if (!volume_component) {
        params.set_error("No volume component found in input SDF");
        return false;
    }

    // Get parameters
    float isovalue = params.get_input<float>("isovalue");
    float adaptivity = params.get_input<float>("adaptivity");

    // Get the SDF grid
    openvdb::FloatGrid::Ptr grid =
        openvdb::gridPtrCast<openvdb::FloatGrid>(volume_component->get_grid());

    if (!grid) {
        params.set_error("Input does not contain an SDF grid");
        return false;
    }

    // Initialize OpenVDB if needed
    openvdb::initialize();

    // Output data structures
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;

    // Convert volume to mesh using adaptive dual marching cubes
    openvdb::tools::volumeToMesh(
        *grid, points, triangles, quads, isovalue, adaptivity);

    // Create output mesh geometry
    Geometry mesh_geometry = Geometry::CreateMesh();
    auto mesh_component = mesh_geometry.get_component<MeshComponent>();

    // Convert points
    std::vector<glm::vec3> mesh_vertices;
    mesh_vertices.reserve(points.size());
    for (const auto& point : points) {
        mesh_vertices.emplace_back(point[0], point[1], point[2]);
    }

    // Create face vertex counts and indices
    std::vector<int> mesh_faceVertexCounts;
    std::vector<int> mesh_faceVertexIndices;

    // Add quads
    mesh_faceVertexCounts.reserve(quads.size() + triangles.size());
    mesh_faceVertexIndices.reserve(quads.size() * 4 + triangles.size() * 3);

    for (const auto& quad : quads) {
        mesh_faceVertexCounts.emplace_back(4);
        for (int j = 0; j < 4; ++j) {
            mesh_faceVertexIndices.emplace_back(quad[j]);
        }
    }

    // Add triangles
    for (const auto& triangle : triangles) {
        mesh_faceVertexCounts.emplace_back(3);
        for (int j = 0; j < 3; ++j) {
            mesh_faceVertexIndices.emplace_back(triangle[j]);
        }
    }

    // Set the mesh data
    mesh_component->set_vertices(mesh_vertices);
    mesh_component->set_face_vertex_counts(mesh_faceVertexCounts);
    mesh_component->set_face_vertex_indices(mesh_faceVertexIndices);

    // Set output
    params.set_output("M", mesh_geometry);

    return true;
}

NODE_DECLARATION_UI(sdf2mesh);
NODE_DEF_CLOSE_SCOPE

#endif
