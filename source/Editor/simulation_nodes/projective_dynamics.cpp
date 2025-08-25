#ifdef GPU_GEOM_ALGORITHM

#include "GCore/algorithms/intersection.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(projective_dynamics)
{
    // Function content omitted
    b.add_input<Geometry>("Mesh");
    b.add_output<Geometry>("Mesh");
}

NODE_EXECUTION_FUNCTION(projective_dynamics)
{
    // Function content omitted

    auto input_mesh = params.get_input<Geometry>("Mesh");

    auto resource_allocator = get_resource_allocator();

    return true;
}

NODE_DECLARATION_UI(projective_dynamics);
NODE_DEF_CLOSE_SCOPE

#endif
