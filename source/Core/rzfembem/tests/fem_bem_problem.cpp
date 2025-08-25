#include <gtest/gtest.h>
#include <pxr/base/vt/array.h>

#include <fem_bem/fem_bem.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/algorithms/delauney.h"
#include "GCore/create_geom.h"

using namespace USTC_CG::fem_bem;
using namespace USTC_CG;

TEST(FEMBEMProblem, Laplacian)
{
    Geometry circle = create_circle_face(64, 1);

    auto delauneyed = geom_algorithm::delaunay(circle);

    ElementSolverDesc desc;
    desc.set_problem_dim(2)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);

    // Should be some notation from the periodic finite element table.
    solver->set_geometry(delauneyed);

    auto n = solver->get_boundary_count();

    EXPECT_EQ(n, 1);

    solver->set_boundary_condition(
        "sin(atan2(y,x)*2)", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();

    auto mesh_component = delauneyed.get_component<MeshComponent>();

    mesh_component->add_vertex_scalar_quantity("solution", solution);
}
