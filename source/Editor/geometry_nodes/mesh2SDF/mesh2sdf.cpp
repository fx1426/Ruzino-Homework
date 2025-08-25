#ifdef GEOM_USD_EXTENSION

#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/VolumeComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(mesh2sdf)
{
    // Function content omitted
    b.add_input<Geometry>("M");
    b.add_input<float>("dx").min(0.002f).max(1).default_val(0.01);
    b.add_output<Geometry>("SDF");
}

NODE_EXECUTION_FUNCTION(mesh2sdf)
{
    auto geometry = params.get_input<Geometry>("M");
    geometry.apply_transform();

    auto mesh_component = geometry.get_component<MeshComponent>();

    // Check if we have a valid mesh
    if (!mesh_component) {
        params.set_error("No mesh component found in input geometry");
        return false;
    }

    // Get dx parameter
    float voxel_size = params.get_input<float>("dx");

    // Create narrow band width (typically 3 is a good value)
    const int narrow_band_width = 3;

    // Get mesh data
    const auto& vertices = mesh_component->get_vertices();
    const auto& indices = mesh_component->get_face_vertex_indices();
    const auto& face_index_count = mesh_component->get_face_vertex_counts();

    // Initialize OpenVDB
    openvdb::initialize();

    if (voxel_size <= 0.00001) {
        return false;
    }

    // Create mesh to SDF converter
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(voxel_size);
    // Create points array from vertices vector
    std::vector<openvdb::Vec3s> points;
    points.reserve(vertices.size());
    for (const auto& v : vertices) {
        points.push_back(openvdb::Vec3s(v[0], v[1], v[2]));
    }

    // Create triangle array from face indices
    std::vector<openvdb::Vec3I> tris;
    tris.reserve(indices.size() / 3);

    std::vector<openvdb::Vec4I> quads;
    quads.reserve(indices.size() / 4);

    for (size_t i = 0; i < face_index_count.size(); ++i) {
        if (face_index_count[i] == 3) {
            tris.push_back(openvdb::Vec3I(
                indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]));
        }
        else if (face_index_count[i] == 4) {
            quads.push_back(openvdb::Vec4I(
                indices[i * 4],
                indices[i * 4 + 1],
                indices[i * 4 + 2],
                indices[i * 4 + 3]));
        }
        else {
            params.set_error("Only triangles and quads are supported");
            return false;
        }
    }
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
            *transform,
            points,
            tris,
            quads,
            narrow_band_width * voxel_size,  // Exterior band width
            narrow_band_width * voxel_size   // Interior band width
        );
    grid->setName("sdf");

    // Create output SDF geometry
    Geometry sdf_geometry = Geometry::CreateVolume();

    auto volume_component = sdf_geometry.get_component<VolumeComponent>();

    volume_component->add_grid(grid);

    // Set output
    params.set_output("SDF", sdf_geometry);

    return true;
}

NODE_DECLARATION_UI(mesh2sdf);
NODE_DEF_CLOSE_SCOPE

#endif
