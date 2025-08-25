#ifdef GEOM_USD_EXTENSION

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>

#include "GCore/Components/VolumeComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(upsample)
{
    b.add_input<Geometry>("SDF");
    b.add_input<int>("Order").default_val(1).min(1).max(2);
    b.add_output<Geometry>("Upsampled");
}

NODE_EXECUTION_FUNCTION(upsample)
{
    auto sdf_geometry = params.get_input<Geometry>("SDF");
    auto volume_component = sdf_geometry.get_component<VolumeComponent>();

    // Check if we have a valid volume
    if (!volume_component) {
        params.set_error("No volume component found in input SDF");
        return false;
    }

    // Get the SDF grid
    openvdb::FloatGrid::Ptr input_grid =
        openvdb::gridPtrCast<openvdb::FloatGrid>(volume_component->get_grid());

    if (!input_grid) {
        params.set_error("Input does not contain an SDF grid");
        return false;
    }

    // Get interpolation order
    int order = params.get_input<int>("Order");

    // Create a new transform with half the voxel size
    openvdb::math::Transform::Ptr input_transform = input_grid->transformPtr();
    double voxel_size = input_transform->voxelSize()[0];
    double new_voxel_size = voxel_size / 2.0;

    openvdb::math::Transform::Ptr new_transform =
        openvdb::math::Transform::createLinearTransform(new_voxel_size);

    // Create the output grid with the new transform
    openvdb::FloatGrid::Ptr output_grid = openvdb::FloatGrid::create();
    output_grid->setTransform(new_transform);
    output_grid->setGridClass(input_grid->getGridClass());
    output_grid->setName(input_grid->getName());

    // Perform grid resampling using different interpolation schemes based on
    // order
    if (order == 1) {
        // Linear interpolation (first order)
        openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(
            *input_grid, *output_grid);
    }
    else if (order == 2) {
        // Quadratic interpolation (second order)
        openvdb::tools::resampleToMatch<openvdb::tools::QuadraticSampler>(
            *input_grid, *output_grid);
    }

    else {
        params.set_error("Invalid interpolation order");
        return false;
    }

    // Create output SDF geometry
    Geometry upsampled_geometry = Geometry::CreateVolume();
    auto output_volume_component =
        upsampled_geometry.get_component<VolumeComponent>();
    output_volume_component->add_grid(output_grid);

    // Set output
    params.set_output("Upsampled", upsampled_geometry);

    return true;
}

NODE_DECLARATION_UI(upsample);
NODE_DEF_CLOSE_SCOPE

#endif
