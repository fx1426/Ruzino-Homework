
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(set_material)
{
    // Function content omitted
    b.add_input<Geometry>("Geometry");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(set_material)
{
    // Function content omitted
    return true;
}

NODE_DECLARATION_UI(set_material);
NODE_DEF_CLOSE_SCOPE
