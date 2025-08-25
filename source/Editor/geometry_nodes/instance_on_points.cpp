#ifdef GEOM_USD_EXTENSION

#include <pxr/base/gf/matrix4d.h>

#include "GCore/Components/InstancerComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/GOP.h"
#include "glm/ext/matrix_transform.hpp"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(instance_on_points)
{
    // Function content omitted

    b.add_input<Geometry>("Geometry");
    b.add_input<Geometry>("Points");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(instance_on_points)
{
    // Function content omitted
    auto points = params.get_input<Geometry>("Points");
    auto geometry = params.get_input<Geometry>("Geometry");

    auto instancer = std::make_shared<InstancerComponent>(&geometry);
    geometry.attach_component(instancer);

    auto points_component = points.get_component<PointsComponent>();

    auto points_vertices = points_component->get_vertices();
    if (!points_component) {
        params.set_error("No points component found in input Points");
        return false;
    }

    for (auto& point : points_vertices) {
        auto instance = glm::translate(
            glm::identity<glm::mat4>(),
            glm::vec3(point[0], point[1], point[2]));
        instancer->add_instance(instance);
    }

    params.set_output("Geometry", std::move(geometry));

    return true;
}

NODE_DECLARATION_UI(instance_on_points);
NODE_DEF_CLOSE_SCOPE

#endif
