#include <exprtk/exprtk.hpp>

#include "GCore/GOP.h"
#include "fem_bem/ElementBasis.hpp"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(dirichlet_bc)
{
    // Function content omitted
    b.add_input<std::string>("Expression");
    b.add_input<Geometry>("Geometry");
    b.add_input<fem_bem::ElementBasisHandle>("Element Basis");
    b.add_output<pxr::VtArray<float>>("Dirichlet BC");
}

NODE_EXECUTION_FUNCTION(dirichlet_bc)
{
    auto geom = params.get_input<Geometry>("Geometry");

    auto element_basis =
        params.get_input<fem_bem::ElementBasisHandle>("Element Basis");

    // Function content omitted
    return true;
}

NODE_DECLARATION_UI(dirichlet_bc);
NODE_DEF_CLOSE_SCOPE
