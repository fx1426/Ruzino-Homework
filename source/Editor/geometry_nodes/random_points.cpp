
#include <random>

#include "GCore/Components/PointsComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(random_points)
{
    // Function content omitted
    b.add_input<float>("x_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("x_max").min(-3).max(3).default_val(1);
    b.add_input<float>("y_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("y_max").min(-3).max(3).default_val(1);
    b.add_input<float>("z_min").min(-3).max(3).default_val(-1);
    b.add_input<float>("z_max").min(-3).max(3).default_val(1);

    b.add_input<float>("width").min(0.01).max(1).default_val(0.1);
    b.add_input<int>("num_points").min(1).max(10000).default_val(100);

    b.add_input<int>("Seed").default_val(0).min(0).max(100);

    b.add_output<Geometry>("Points");
}

NODE_EXECUTION_FUNCTION(random_points)
{
    // Function content omitted

    Geometry points_geometry = Geometry();

    auto points_component = std::make_shared<PointsComponent>(&points_geometry);
    points_geometry.attach_component(points_component);

    std::vector<glm::vec3> vertices;

    float x_min = params.get_input<float>("x_min");
    float x_max = params.get_input<float>("x_max");
    float y_min = params.get_input<float>("y_min");
    float y_max = params.get_input<float>("y_max");
    float z_min = params.get_input<float>("z_min");
    float z_max = params.get_input<float>("z_max");

    std::mt19937 rng(params.get_input<int>("Seed"));

    std::uniform_real_distribution<float> dist_x(x_min, x_max);
    std::uniform_real_distribution<float> dist_y(y_min, y_max);
    std::uniform_real_distribution<float> dist_z(z_min, z_max);

    const int num_points = params.get_input<int>("num_points");

    std::vector<float> widths(num_points, params.get_input<float>("width"));
    vertices.resize(num_points);

    for (int i = 0; i < num_points; i++) {
        vertices[i] = glm::vec3(dist_x(rng), dist_y(rng), dist_z(rng));
    }

    points_component->set_vertices(vertices);
    points_component->set_width(widths);

    // Set the output
    params.set_output("Points", std::move(points_geometry));
    return true;
}

NODE_DECLARATION_UI(random_points);
NODE_DEF_CLOSE_SCOPE
