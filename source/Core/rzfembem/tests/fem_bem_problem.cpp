#include <gtest/gtest.h>
#include <pxr/base/vt/array.h>

#include <array>
#include <chrono>
#include <fem_bem/fem_bem.hpp>
#include <numeric>

#include "GCore/Components/MeshComponent.h"
#include "GCore/algorithms/delauney.h"
#include "GCore/create_geom.h"

using namespace Ruzino::fem_bem;
using namespace Ruzino;

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

TEST(FEMBEMProblem, Laplacian3D)
{
    // Create a simple 3D geometry (sphere or tetrahedron)
    // For demonstration, we'll create a simple tetrahedron
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Define vertices of a unit tetrahedron
    std::vector<glm::vec3> vertices = {
        { 0.0f, 0.0f, 0.0f },     // vertex 0
        { 1.0f, 0.0f, 0.0f },     // vertex 1
        { 0.5f, 0.866f, 0.0f },   // vertex 2
        { 0.5f, 0.289f, 0.816f }  // vertex 3
    };

    // Define tetrahedral connectivity (4 triangular faces)
    std::vector<int> face_vertex_indices = {
        0, 1, 2,  // face 0: bottom
        0, 1, 3,  // face 1
        1, 2, 3,  // face 2
        0, 2, 3   // face 3
    };

    std::vector<int> face_vertex_counts = { 3, 3, 3, 3 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(face_vertex_indices);
    mesh_comp->set_face_vertex_counts(face_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);

    solver->set_geometry(*geometry);

    auto n = solver->get_boundary_count();
    EXPECT_EQ(n, 1);

    // Set boundary condition: f(x,y,z) = x^2 + y^2 + z^2
    solver->set_boundary_condition(
        "x*x + y*y + z*z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();

    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    // Verify solution has correct size
    EXPECT_EQ(solution.size(), vertices.size());
}

TEST(FEMBEMProblem, Laplacian3D_SingleTetrahedronAsCell)
{
    // Test with tetrahedron defined as a single 4-vertex cell
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Define vertices of a regular tetrahedron
    std::vector<glm::vec3> vertices = {
        { 0.0f, 0.0f, 0.0f },     // vertex 0
        { 1.0f, 0.0f, 0.0f },     // vertex 1
        { 0.5f, 0.866f, 0.0f },   // vertex 2
        { 0.5f, 0.289f, 0.816f }  // vertex 3
    };

    // Define as single tetrahedral cell
    std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
    std::vector<int> cell_vertex_counts = { 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Linear boundary condition: f(x,y,z) = x + y + z
    solver->set_boundary_condition(
        "x + y + z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());
    // For linear boundary conditions on a tetrahedron, interior values should
    // be reasonable
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(FEMBEMProblem, Laplacian3D_TwoAdjacentTetrahedra)
{
    // Test with two adjacent tetrahedra sharing a face
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Define vertices for two adjacent tetrahedra
    std::vector<glm::vec3> vertices = {
        { 0.0f, 0.0f, 0.0f },      // v0
        { 1.0f, 0.0f, 0.0f },      // v1
        { 0.5f, 0.866f, 0.0f },    // v2
        { 0.5f, 0.289f, 0.816f },  // v3 (apex of first tet)
        { 0.5f, 0.289f, -0.816f }  // v4 (apex of second tet)
    };

    // Define two tetrahedra as cells
    std::vector<int> cell_vertex_indices = {
        0, 1, 2, 3,  // first tetrahedron
        0, 1, 2, 4   // second tetrahedron (shares face 0,1,2)
    };
    std::vector<int> cell_vertex_counts = { 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Quadratic boundary condition
    solver->set_boundary_condition(
        "x*x + y*y + z*z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Verify solution validity
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(FEMBEMProblem, Laplacian3D_CubicMesh)
{
    // Test with a more complex cubic mesh
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a 2x2x2 grid of vertices
    std::vector<glm::vec3> vertices;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                vertices.push_back(
                    glm::vec3(
                        static_cast<float>(i),
                        static_cast<float>(j),
                        static_cast<float>(k)));
            }
        }
    }

    auto get_vertex_index = [](int i, int j, int k) {
        return i * 4 + j * 2 + k;
    };

    // Create tetrahedra by subdividing the cube into 6 tetrahedra
    std::vector<int> cell_vertex_indices;
    std::vector<int> cell_vertex_counts;

    // Get the 8 vertices of the cube
    int v000 = get_vertex_index(0, 0, 0);
    int v001 = get_vertex_index(0, 0, 1);
    int v010 = get_vertex_index(0, 1, 0);
    int v011 = get_vertex_index(0, 1, 1);
    int v100 = get_vertex_index(1, 0, 0);
    int v101 = get_vertex_index(1, 0, 1);
    int v110 = get_vertex_index(1, 1, 0);
    int v111 = get_vertex_index(1, 1, 1);

    // Subdivide cube into 6 tetrahedra
    std::vector<std::array<int, 4>> cube_tets = {
        { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
        { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
        { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
    };

    for (const auto& tet : cube_tets) {
        for (int vertex : tet) {
            cell_vertex_indices.push_back(vertex);
        }
        cell_vertex_counts.push_back(4);
    }

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Trigonometric boundary condition
    solver->set_boundary_condition(
        "sin(3.14159*x)*cos(3.14159*y)*sin(3.14159*z)",
        BoundaryCondition::Dirichlet,
        0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Verify solution validity
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(FEMBEMProblem, Laplacian3D_PyramidMesh)
{
    // Test with a pyramid mesh
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a pyramid with square base
    std::vector<glm::vec3> vertices = {
        // Base vertices (square base at z=0)
        { -1.0f, -1.0f, 0.0f },  // 0
        { 1.0f, -1.0f, 0.0f },   // 1
        { 1.0f, 1.0f, 0.0f },    // 2
        { -1.0f, 1.0f, 0.0f },   // 3
        // Apex vertex
        { 0.0f, 0.0f, 2.0f }  // 4
    };

    // Create tetrahedra by connecting apex to triangular faces of base
    std::vector<int> cell_vertex_indices = {
        // Tetrahedron 1: apex + triangle (0,1,2)
        4,
        0,
        1,
        2,
        // Tetrahedron 2: apex + triangle (0,2,3)
        4,
        0,
        2,
        3
    };
    std::vector<int> cell_vertex_counts = { 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Polynomial boundary condition
    solver->set_boundary_condition(
        "x*x + y*y + 2*z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Check that apex vertex has a reasonable solution value
    float apex_solution = solution[4];
    EXPECT_FALSE(std::isnan(apex_solution));
    EXPECT_FALSE(std::isinf(apex_solution));
}

TEST(FEMBEMProblem, Laplacian3D_ErrorHandling)
{
    // Test error handling for various invalid scenarios

    // Test 1: Empty geometry (should handle gracefully)
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        ElementSolverDesc desc;
        desc.set_problem_dim(3)
            .set_element_family(ElementFamily::P_minus)
            .set_k(0)
            .set_equation_type(EquationType::Laplacian);

        auto solver = create_element_solver(desc);
        EXPECT_NO_THROW(solver->set_geometry(*geometry));
    }

    // Test 2: Degenerate tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        // All vertices in the same plane (degenerate tetrahedron)
        std::vector<glm::vec3> vertices = {
            { 0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f },
            { 0.5f, 0.5f, 0.0f },
            { 0.3f, 0.3f, 0.0f }  // All in z=0 plane
        };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        ElementSolverDesc desc;
        desc.set_problem_dim(3)
            .set_element_family(ElementFamily::P_minus)
            .set_k(0)
            .set_equation_type(EquationType::Laplacian);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x + y + z", BoundaryCondition::Dirichlet, 0);

        // Should handle degenerate tetrahedron gracefully
        EXPECT_NO_THROW(solver->solve());
    }

    // Test 3: Invalid boundary condition expression
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = { { 0.0f, 0.0f, 0.0f },
                                            { 1.0f, 0.0f, 0.0f },
                                            { 0.5f, 0.866f, 0.0f },
                                            { 0.5f, 0.289f, 0.816f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        ElementSolverDesc desc;
        desc.set_problem_dim(3)
            .set_element_family(ElementFamily::P_minus)
            .set_k(0)
            .set_equation_type(EquationType::Laplacian);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);

        // Invalid expression should be handled gracefully
        EXPECT_NO_THROW(solver->set_boundary_condition(
            "invalid_function(x,y,z)", BoundaryCondition::Dirichlet, 0));
    }
}

TEST(FEMBEMProblem, Laplacian3D_LargeScale)
{
    // Test with a larger mesh to check scalability
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a 3x3x3 grid of vertices
    const int grid_size = 3;
    std::vector<glm::vec3> vertices;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            for (int k = 0; k < grid_size; k++) {
                vertices.push_back(
                    glm::vec3(
                        static_cast<float>(i),
                        static_cast<float>(j),
                        static_cast<float>(k)));
            }
        }
    }

    auto get_vertex_index = [grid_size](int i, int j, int k) {
        return i * grid_size * grid_size + j * grid_size + k;
    };

    std::vector<int> cell_vertex_indices;
    std::vector<int> cell_vertex_counts;

    // Create tetrahedra by subdividing each cube
    for (int i = 0; i < grid_size - 1; i++) {
        for (int j = 0; j < grid_size - 1; j++) {
            for (int k = 0; k < grid_size - 1; k++) {
                // Get the 8 vertices of the cube
                int v000 = get_vertex_index(i, j, k);
                int v001 = get_vertex_index(i, j, k + 1);
                int v010 = get_vertex_index(i, j + 1, k);
                int v011 = get_vertex_index(i, j + 1, k + 1);
                int v100 = get_vertex_index(i + 1, j, k);
                int v101 = get_vertex_index(i + 1, j, k + 1);
                int v110 = get_vertex_index(i + 1, j + 1, k);
                int v111 = get_vertex_index(i + 1, j + 1, k + 1);

                // Subdivide cube into 6 tetrahedra
                std::vector<std::array<int, 4>> cube_tets = {
                    { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
                    { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
                    { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
                };

                for (const auto& tet : cube_tets) {
                    for (int vertex : tet) {
                        cell_vertex_indices.push_back(vertex);
                    }
                    cell_vertex_counts.push_back(4);
                }
            }
        }
    }

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Simple linear boundary condition
    solver->set_boundary_condition(
        "x + y + z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Verify solution statistics
    float min_val = *std::min_element(solution.begin(), solution.end());
    float max_val = *std::max_element(solution.begin(), solution.end());

    EXPECT_FALSE(std::isnan(min_val));
    EXPECT_FALSE(std::isnan(max_val));
    EXPECT_FALSE(std::isinf(min_val));
    EXPECT_FALSE(std::isinf(max_val));

    // For linear boundary conditions, solution should have reasonable range
    EXPECT_LE(std::abs(max_val - min_val), 10.0f);  // reasonable spread
}

TEST(FEMBEMProblem, Laplacian3D_BoundaryConditionTypes)
{
    // Test different types of boundary conditions
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a unit cube as 6 tetrahedra
    std::vector<glm::vec3> vertices = {
        { 0.0f, 0.0f, 0.0f },  // v0
        { 1.0f, 0.0f, 0.0f },  // v1
        { 1.0f, 1.0f, 0.0f },  // v2
        { 0.0f, 1.0f, 0.0f },  // v3
        { 0.0f, 0.0f, 1.0f },  // v4
        { 1.0f, 0.0f, 1.0f },  // v5
        { 1.0f, 1.0f, 1.0f },  // v6
        { 0.0f, 1.0f, 1.0f }   // v7
    };

    // Create tetrahedra by subdividing the cube
    std::vector<int> cell_vertex_indices = {
        0, 1, 2, 4,  // tet 1
        2, 4, 6, 1,  // tet 2
        2, 3, 4, 6,  // tet 3
        3, 4, 6, 7,  // tet 4
        1, 2, 6, 5,  // tet 5
        4, 5, 6, 1   // tet 6
    };
    std::vector<int> cell_vertex_counts = { 4, 4, 4, 4, 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test different boundary conditions
    std::vector<std::string> boundary_conditions = {
        "1.0",                    // constant
        "x",                      // linear in x
        "y",                      // linear in y
        "z",                      // linear in z
        "x + y + z",              // linear combination
        "x*x",                    // quadratic in x
        "y*y",                    // quadratic in y
        "z*z",                    // quadratic in z
        "x*x + y*y + z*z",        // radial quadratic
        "x*y",                    // bilinear
        "x*y*z",                  // trilinear
        "sin(3.14159*x)",         // trigonometric
        "cos(3.14159*y)",         // trigonometric
        "exp(x)",                 // exponential
        "sqrt(x*x + y*y + 0.01)"  // square root (with small offset to avoid
                                  // singularity)
    };

    for (const auto& bc : boundary_conditions) {
        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(bc, BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();

        EXPECT_EQ(solution.size(), vertices.size());

        // Verify solution validity for each boundary condition
        for (size_t i = 0; i < solution.size(); i++) {
            EXPECT_FALSE(std::isnan(solution[i]))
                << "NaN found with BC: " << bc << " at vertex " << i;
            EXPECT_FALSE(std::isinf(solution[i]))
                << "Inf found with BC: " << bc << " at vertex " << i;
        }

        mesh_comp->add_vertex_scalar_quantity("solution_" + bc, solution);
    }
}

TEST(FEMBEMProblem, Laplacian3D_MeshQuality)
{
    // Test solver robustness with different mesh qualities
    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test 1: Well-conditioned regular tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = { { 0.0f, 0.0f, 0.0f },
                                            { 1.0f, 0.0f, 0.0f },
                                            { 0.5f, 0.866f, 0.0f },
                                            { 0.5f, 0.289f, 0.816f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x + y + z", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }

    // Test 2: Highly stretched tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = {
            { 0.0f, 0.0f, 0.0f },
            { 10.0f, 0.0f, 0.0f },  // very long edge
            { 0.1f, 0.1f, 0.0f },   // small triangle
            { 0.05f, 0.05f, 0.1f }  // stretched in z
        };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("x", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }

    // Test 3: Nearly degenerate tetrahedron (but not completely flat)
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = {
            { 0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f },
            { 0.5f, 0.866f, 0.0f },
            { 0.5f, 0.433f, 1e-3f }  // very small height
        };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("1.0", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }
}

TEST(FEMBEMProblem, Laplacian3D_ConvergenceStudy)
{
    // Test convergence with mesh refinement
    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test with increasing mesh resolution
    std::vector<int> grid_sizes = { 2, 3, 4 };
    std::vector<float> solution_norms;

    for (int grid_size : grid_sizes) {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        // Create grid of vertices
        std::vector<glm::vec3> vertices;
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                for (int k = 0; k < grid_size; k++) {
                    float x = static_cast<float>(i) / (grid_size - 1);
                    float y = static_cast<float>(j) / (grid_size - 1);
                    float z = static_cast<float>(k) / (grid_size - 1);
                    vertices.push_back(glm::vec3(x, y, z));
                }
            }
        }

        auto get_vertex_index = [grid_size](int i, int j, int k) {
            return i * grid_size * grid_size + j * grid_size + k;
        };

        std::vector<int> cell_vertex_indices;
        std::vector<int> cell_vertex_counts;

        // Create tetrahedra
        for (int i = 0; i < grid_size - 1; i++) {
            for (int j = 0; j < grid_size - 1; j++) {
                for (int k = 0; k < grid_size - 1; k++) {
                    // Standard cube-to-tet subdivision
                    int v000 = get_vertex_index(i, j, k);
                    int v001 = get_vertex_index(i, j, k + 1);
                    int v010 = get_vertex_index(i, j + 1, k);
                    int v011 = get_vertex_index(i, j + 1, k + 1);
                    int v100 = get_vertex_index(i + 1, j, k);
                    int v101 = get_vertex_index(i + 1, j, k + 1);
                    int v110 = get_vertex_index(i + 1, j + 1, k);
                    int v111 = get_vertex_index(i + 1, j + 1, k + 1);

                    std::vector<std::array<int, 4>> cube_tets = {
                        { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
                        { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
                        { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
                    };

                    for (const auto& tet : cube_tets) {
                        for (int vertex : tet) {
                            cell_vertex_indices.push_back(vertex);
                        }
                        cell_vertex_counts.push_back(4);
                    }
                }
            }
        }

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x*x + y*y + z*z", BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();

        // Compute L2 norm of solution
        float norm = 0.0f;
        for (float val : solution) {
            norm += val * val;
        }
        norm = std::sqrt(norm / solution.size());
        solution_norms.push_back(norm);

        mesh_comp->add_vertex_scalar_quantity("solution", solution);
    }

    // Check that solution norms are reasonable and converging
    EXPECT_GT(solution_norms.size(), 1);
    for (float norm : solution_norms) {
        EXPECT_GT(norm, 0.0f);
        EXPECT_LT(norm, 1000.0f);  // Should be bounded
    }
}

TEST(FEMBEMProblem, Laplacian3D_PerformanceTest)
{
    // Test performance with moderately large mesh
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    const int grid_size = 5;  // 5x5x5 grid = 125 vertices
    std::vector<glm::vec3> vertices;

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            for (int k = 0; k < grid_size; k++) {
                vertices.push_back(
                    glm::vec3(
                        static_cast<float>(i),
                        static_cast<float>(j),
                        static_cast<float>(k)));
            }
        }
    }

    auto get_vertex_index = [grid_size](int i, int j, int k) {
        return i * grid_size * grid_size + j * grid_size + k;
    };

    std::vector<int> cell_vertex_indices;
    std::vector<int> cell_vertex_counts;

    for (int i = 0; i < grid_size - 1; i++) {
        for (int j = 0; j < grid_size - 1; j++) {
            for (int k = 0; k < grid_size - 1; k++) {
                int v000 = get_vertex_index(i, j, k);
                int v001 = get_vertex_index(i, j, k + 1);
                int v010 = get_vertex_index(i, j + 1, k);
                int v011 = get_vertex_index(i, j + 1, k + 1);
                int v100 = get_vertex_index(i + 1, j, k);
                int v101 = get_vertex_index(i + 1, j, k + 1);
                int v110 = get_vertex_index(i + 1, j + 1, k);
                int v111 = get_vertex_index(i + 1, j + 1, k + 1);

                std::vector<std::array<int, 4>> cube_tets = {
                    { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
                    { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
                    { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
                };

                for (const auto& tet : cube_tets) {
                    for (int vertex : tet) {
                        cell_vertex_indices.push_back(vertex);
                    }
                    cell_vertex_counts.push_back(4);
                }
            }
        }
    }

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);
    solver->set_boundary_condition(
        "x + y + z", BoundaryCondition::Dirichlet, 0);

    // Time the solve (this is a performance test, not just correctness)
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> solution = solver->solve();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    EXPECT_EQ(solution.size(), vertices.size());
    EXPECT_LT(duration.count(), 10000);  // Should complete within 10 seconds

    // Verify solution quality
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    mesh_comp->add_vertex_scalar_quantity("solution", solution);
}

TEST(FEMBEMProblem, Laplacian3D_RobustnessTest)
{
    // Test solver robustness with various challenging scenarios
    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test 1: Very small tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = { { 0.0f, 0.0f, 0.0f },
                                            { 1e-6f, 0.0f, 0.0f },
                                            { 0.5e-6f, 0.866e-6f, 0.0f },
                                            { 0.5e-6f, 0.289e-6f, 0.816e-6f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("1.0", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }

    // Test 2: Very large tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = { { 0.0f, 0.0f, 0.0f },
                                            { 1e6f, 0.0f, 0.0f },
                                            { 0.5e6f, 0.866e6f, 0.0f },
                                            { 0.5e6f, 0.289e6f, 0.816e6f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("x", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }

    // Test 3: Negative coordinates
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = { { -1.0f, -1.0f, -1.0f },
                                            { 0.0f, -1.0f, -1.0f },
                                            { -0.5f, 0.0f, -1.0f },
                                            { -0.5f, -0.5f, 0.0f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x + y + z", BoundaryCondition::Dirichlet, 0);

        auto solution = solver->solve();
        EXPECT_EQ(solution.size(), vertices.size());

        for (float val : solution) {
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
        }
    }

    // Test 4: Mixed scales in same mesh
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        // Two tetrahedra with very different scales
        std::vector<glm::vec3> vertices = { // Small tetrahedron
                                            { 0.0f, 0.0f, 0.0f },
                                            { 0.001f, 0.0f, 0.0f },
                                            { 0.0005f, 0.000866f, 0.0f },
                                            { 0.0005f, 0.000289f, 0.000816f },
                                            // Large tetrahedron
                                            { 10.0f, 10.0f, 10.0f },
                                            { 20.0f, 10.0f, 10.0f },
                                            { 15.0f, 18.66f, 10.0f },
                                            { 15.0f, 12.89f, 18.16f }
        };

        std::vector<int> cell_vertex_indices = {
            0, 1, 2, 3,  // small tet
            4, 5, 6, 7   // large tet
        };
        std::vector<int> cell_vertex_counts = { 4, 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("x", BoundaryCondition::Dirichlet, 0);

        auto solution = solver->solve();
        EXPECT_EQ(solution.size(), vertices.size());

        for (float val : solution) {
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
        }
    }
}

TEST(FEMBEMProblem, Laplacian3D_SolutionAccuracy)
{
    // Test solution accuracy for problems with known analytical solutions
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a unit cube mesh
    const int n = 3;  // 3x3x3 grid
    std::vector<glm::vec3> vertices;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                float x = static_cast<float>(i) / (n - 1);
                float y = static_cast<float>(j) / (n - 1);
                float z = static_cast<float>(k) / (n - 1);
                vertices.push_back(glm::vec3(x, y, z));
            }
        }
    }

    auto get_vertex_index = [n](int i, int j, int k) {
        return i * n * n + j * n + k;
    };

    std::vector<int> cell_vertex_indices;
    std::vector<int> cell_vertex_counts;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            for (int k = 0; k < n - 1; k++) {
                int v000 = get_vertex_index(i, j, k);
                int v001 = get_vertex_index(i, j, k + 1);
                int v010 = get_vertex_index(i, j + 1, k);
                int v011 = get_vertex_index(i, j + 1, k + 1);
                int v100 = get_vertex_index(i + 1, j, k);
                int v101 = get_vertex_index(i + 1, j, k + 1);
                int v110 = get_vertex_index(i + 1, j + 1, k);
                int v111 = get_vertex_index(i + 1, j + 1, k + 1);

                std::vector<std::array<int, 4>> cube_tets = {
                    { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
                    { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
                    { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
                };

                for (const auto& tet : cube_tets) {
                    for (int vertex : tet) {
                        cell_vertex_indices.push_back(vertex);
                    }
                    cell_vertex_counts.push_back(4);
                }
            }
        }
    }

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test linear solution (should be exact for P1 elements)
    {
        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x + y + z", BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();

        // For linear boundary conditions, FEM should reproduce the exact
        // solution
        for (size_t i = 0; i < vertices.size(); i++) {
            float expected = vertices[i].x + vertices[i].y + vertices[i].z;
            float computed = solution[i];

            // Check if vertex is on boundary
            bool is_boundary =
                (vertices[i].x == 0.0f || vertices[i].x == 1.0f ||
                 vertices[i].y == 0.0f || vertices[i].y == 1.0f ||
                 vertices[i].z == 0.0f || vertices[i].z == 1.0f);

            if (is_boundary) {
                EXPECT_NEAR(computed, expected, 1e-5f)
                    << "Boundary value mismatch at vertex " << i;
            }
        }

        mesh_comp->add_vertex_scalar_quantity("linear_solution", solution);
    }

    // Test constant solution
    {
        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("5.0", BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();

        // All vertices should have solution value close to 5.0
        for (size_t i = 0; i < solution.size(); i++) {
            EXPECT_NEAR(solution[i], 5.0f, 1e-3f)
                << "Constant solution mismatch at vertex " << i;
        }

        mesh_comp->add_vertex_scalar_quantity("constant_solution", solution);
    }
}

TEST(FEMBEMProblem, Laplacian3D_MemoryAndResourceManagement)
{
    // Test memory management and resource cleanup
    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Create and destroy multiple solvers
    for (int iter = 0; iter < 10; iter++) {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        // Create a simple tetrahedron
        std::vector<glm::vec3> vertices = { { 0.0f, 0.0f, 0.0f },
                                            { 1.0f, 0.0f, 0.0f },
                                            { 0.5f, 0.866f, 0.0f },
                                            { 0.5f, 0.289f, 0.816f } };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            "x + y + z", BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();
        EXPECT_EQ(solution.size(), vertices.size());

        // Force cleanup by resetting shared_ptr
        solver.reset();
        geometry.reset();
        mesh_comp.reset();
    }

    // If we reach here without crashes, memory management is working
    SUCCEED();
}

TEST(FEMBEMProblem, Laplacian3D_NonConvexMesh)
{
    // Test with a non-convex tetrahedral mesh (L-shaped domain)
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create an L-shaped domain using multiple tetrahedra
    std::vector<glm::vec3> vertices = {
        // Bottom face of L (square)
        { 0.0f, 0.0f, 0.0f },  // 0
        { 2.0f, 0.0f, 0.0f },  // 1
        { 2.0f, 1.0f, 0.0f },  // 2
        { 1.0f, 1.0f, 0.0f },  // 3
        { 1.0f, 2.0f, 0.0f },  // 4
        { 0.0f, 2.0f, 0.0f },  // 5
        // Top face of L (same layout but elevated)
        { 0.0f, 0.0f, 1.0f },  // 6
        { 2.0f, 0.0f, 1.0f },  // 7
        { 2.0f, 1.0f, 1.0f },  // 8
        { 1.0f, 1.0f, 1.0f },  // 9
        { 1.0f, 2.0f, 1.0f },  // 10
        { 0.0f, 2.0f, 1.0f }   // 11
    };

    // Create tetrahedra to fill the L-shaped prism
    std::vector<int> cell_vertex_indices = {
        // First rectangular section (0,1,2,3 to 6,7,8,9)
        0,
        1,
        2,
        6,  // tet 1
        2,
        6,
        7,
        1,  // tet 2
        2,
        3,
        6,
        9,  // tet 3
        3,
        6,
        9,
        0,  // tet 4
        6,
        7,
        8,
        9,  // tet 5
        2,
        7,
        8,
        9,  // tet 6
        // Second rectangular section (0,3,4,5 to 6,9,10,11)
        0,
        3,
        4,
        6,  // tet 7
        4,
        6,
        9,
        3,  // tet 8
        4,
        5,
        6,
        10,  // tet 9
        5,
        6,
        10,
        0,  // tet 10
        6,
        9,
        10,
        11,  // tet 11
        4,
        9,
        10,
        11  // tet 12
    };
    std::vector<int> cell_vertex_counts(12, 4);  // 12 tetrahedra

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    EXPECT_EQ(solver->get_boundary_count(), 1);

    // Test with harmonic boundary condition
    solver->set_boundary_condition(
        "x*y + y*z + z*x", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Verify solution validity
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(FEMBEMProblem, Laplacian3D_IrregularTetrahedra)
{
    // Test with various irregular tetrahedral shapes
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    std::vector<glm::vec3> vertices = {
        // Irregular tetrahedron 1 (needle-like)
        { 0.0f, 0.0f, 0.0f },
        { 5.0f, 0.0f, 0.0f },
        { 0.1f, 0.1f, 0.0f },
        { 0.05f, 0.05f, 0.1f },

        // Irregular tetrahedron 2 (flat)
        { 1.0f, 1.0f, 1.0f },
        { 2.0f, 1.0f, 1.0f },
        { 1.5f, 2.0f, 1.0f },
        { 1.5f, 1.5f, 1.01f },  // very small height

        // Irregular tetrahedron 3 (twisted)
        { 3.0f, 0.0f, 0.0f },
        { 3.0f, 1.0f, 1.0f },
        { 4.0f, 0.5f, 0.5f },
        { 3.2f, 0.3f, 2.0f }
    };

    std::vector<int> cell_vertex_indices = {
        0, 1, 2,  3,  // needle tetrahedron
        4, 5, 6,  7,  // flat tetrahedron
        8, 9, 10, 11  // twisted tetrahedron
    };
    std::vector<int> cell_vertex_counts = { 4, 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    solver->set_boundary_condition(
        "x + y + z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Check that solver handles irregular shapes gracefully
    for (float val : solution) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(FEMBEMProblem, Laplacian3D_BoundaryConditionAccuracy)
{
    // Test accuracy of boundary condition application
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a simple cube
    std::vector<glm::vec3> vertices = {
        { 0.0f, 0.0f, 0.0f },  // v0
        { 1.0f, 0.0f, 0.0f },  // v1
        { 1.0f, 1.0f, 0.0f },  // v2
        { 0.0f, 1.0f, 0.0f },  // v3
        { 0.0f, 0.0f, 1.0f },  // v4
        { 1.0f, 0.0f, 1.0f },  // v5
        { 1.0f, 1.0f, 1.0f },  // v6
        { 0.0f, 1.0f, 1.0f }   // v7
    };

    std::vector<int> cell_vertex_indices = {
        0, 1, 2, 4,  // tet 1
        2, 4, 6, 1,  // tet 2
        2, 3, 4, 6,  // tet 3
        3, 4, 6, 7,  // tet 4
        1, 2, 6, 5,  // tet 5
        4, 5, 6, 1   // tet 6
    };
    std::vector<int> cell_vertex_counts = { 4, 4, 4, 4, 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test polynomial boundary conditions of different orders
    std::vector<std::pair<std::string, std::string>> bc_tests = {
        { "constant", "3.14159" },
        { "linear_x", "2*x" },
        { "linear_y", "3*y" },
        { "linear_z", "4*z" },
        { "linear_combo", "x + 2*y + 3*z" },
        { "quadratic", "x*x + y*y" },
        { "cubic", "x*x*x" },
        { "mixed_degree", "x*x + y*z" },
        { "harmonic", "x*y + y*z + z*x" }
    };

    for (const auto& [name, bc_expr] : bc_tests) {
        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition(
            bc_expr, BoundaryCondition::Dirichlet, 0);

        std::vector<float> solution = solver->solve();

        EXPECT_EQ(solution.size(), vertices.size())
            << "Failed for BC: " << name;

        // Verify solution validity
        for (size_t i = 0; i < solution.size(); i++) {
            EXPECT_FALSE(std::isnan(solution[i]))
                << "NaN in solution for BC: " << name << " at vertex " << i;
            EXPECT_FALSE(std::isinf(solution[i]))
                << "Inf in solution for BC: " << name << " at vertex " << i;
        }

        mesh_comp->add_vertex_scalar_quantity("solution_" + name, solution);
    }
}

TEST(FEMBEMProblem, Laplacian3D_MeshConnectivity)
{
    // Test with complex mesh connectivity patterns
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a mesh where vertices have varying connectivity
    std::vector<glm::vec3> vertices = {
        // Central hub vertex (high connectivity)
        { 0.0f, 0.0f, 0.0f },  // 0 - hub

        // Ring of vertices around hub
        { 1.0f, 0.0f, 0.0f },        // 1
        { 0.707f, 0.707f, 0.0f },    // 2
        { 0.0f, 1.0f, 0.0f },        // 3
        { -0.707f, 0.707f, 0.0f },   // 4
        { -1.0f, 0.0f, 0.0f },       // 5
        { -0.707f, -0.707f, 0.0f },  // 6
        { 0.0f, -1.0f, 0.0f },       // 7
        { 0.707f, -0.707f, 0.0f },   // 8

        // Top and bottom vertices
        { 0.0f, 0.0f, 1.0f },   // 9 - top
        { 0.0f, 0.0f, -1.0f },  // 10 - bottom

        // Outer vertices (low connectivity)
        { 2.0f, 0.0f, 0.0f },   // 11
        { 0.0f, 2.0f, 0.0f },   // 12
        { -2.0f, 0.0f, 0.0f },  // 13
        { 0.0f, -2.0f, 0.0f }   // 14
    };

    // Create tetrahedra with hub vertex connected to many elements
    std::vector<int> cell_vertex_indices = {
        // Tetrahedra connecting hub to ring (high connectivity for vertex 0)
        0,
        1,
        2,
        9,  // hub connected to 1,2,9
        0,
        2,
        3,
        9,  // hub connected to 2,3,9
        0,
        3,
        4,
        9,  // hub connected to 3,4,9
        0,
        4,
        5,
        9,  // hub connected to 4,5,9
        0,
        5,
        6,
        9,  // hub connected to 5,6,9
        0,
        6,
        7,
        9,  // hub connected to 6,7,9
        0,
        7,
        8,
        9,  // hub connected to 7,8,9
        0,
        8,
        1,
        9,  // hub connected to 8,1,9

        // Bottom tetrahedra
        0,
        1,
        2,
        10,
        0,
        2,
        3,
        10,
        0,
        3,
        4,
        10,
        0,
        4,
        5,
        10,

        // Outer tetrahedra (low connectivity for outer vertices)
        1,
        11,
        2,
        9,  // outer vertex 11 with low connectivity
        3,
        12,
        4,
        9,  // outer vertex 12 with low connectivity
        5,
        13,
        6,
        10,  // outer vertex 13 with low connectivity
        7,
        14,
        8,
        10  // outer vertex 14 with low connectivity
    };
    std::vector<int> cell_vertex_counts(16, 4);

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    solver->set_boundary_condition(
        "x*x + y*y + z*z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Verify that high-connectivity vertex (hub) has reasonable solution
    float hub_solution = solution[0];
    EXPECT_FALSE(std::isnan(hub_solution));
    EXPECT_FALSE(std::isinf(hub_solution));

    // Verify low-connectivity vertices also have valid solutions
    for (int i = 11; i <= 14; i++) {
        EXPECT_FALSE(std::isnan(solution[i]))
            << "Invalid solution at low-connectivity vertex " << i;
        EXPECT_FALSE(std::isinf(solution[i]))
            << "Invalid solution at low-connectivity vertex " << i;
    }
}

TEST(FEMBEMProblem, Laplacian3D_SymmetryTest)
{
    // Test that symmetric meshes produce symmetric solutions
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    // Create a symmetric double tetrahedron
    std::vector<glm::vec3> vertices = {
        // Shared edge vertices
        { 0.0f, -1.0f, 0.0f },  // 0
        { 0.0f, 1.0f, 0.0f },   // 1

        // Left tetrahedron vertices
        { -1.0f, 0.0f, 0.0f },  // 2
        { 0.0f, 0.0f, 1.0f },   // 3

        // Right tetrahedron vertices (symmetric)
        { 1.0f, 0.0f, 0.0f },  // 4
        { 0.0f, 0.0f, -1.0f }  // 5
    };

    std::vector<int> cell_vertex_indices = {
        0, 1, 2, 3,  // left tetrahedron
        0, 1, 4, 5   // right tetrahedron (symmetric)
    };
    std::vector<int> cell_vertex_counts = { 4, 4 };

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    // Use symmetric boundary condition
    solver->set_boundary_condition(
        "y*y + z*z", BoundaryCondition::Dirichlet, 0);

    std::vector<float> solution = solver->solve();
    mesh_comp->add_vertex_scalar_quantity("solution", solution);

    EXPECT_EQ(solution.size(), vertices.size());

    // Check symmetry: left and right symmetric vertices should have similar
    // solutions Vertices 2 and 4 are symmetric (x-coordinate sign flipped)
    float left_solution = solution[2];
    float right_solution = solution[4];

    EXPECT_NEAR(left_solution, right_solution, 1e-3f)
        << "Symmetric vertices should have similar solutions";

    // Vertices 3 and 5 are symmetric (z-coordinate sign flipped)
    float top_solution = solution[3];
    float bottom_solution = solution[5];

    EXPECT_NEAR(top_solution, bottom_solution, 1e-3f)
        << "Symmetric vertices should have similar solutions";
}

TEST(FEMBEMProblem, Laplacian3D_ExtremeAspectRatios)
{
    // Test solver stability with extreme aspect ratio tetrahedra
    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    // Test 1: Very thin tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = {
            { 0.0f, 0.0f, 0.0f },
            { 100.0f, 0.0f, 0.0f },   // very long in x
            { 50.0f, 0.01f, 0.0f },   // thin in y
            { 50.0f, 0.005f, 0.01f }  // thin in z
        };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("x", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }

    // Test 2: Very flat tetrahedron
    {
        auto geometry = std::make_shared<Geometry>();
        auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
        geometry->attach_component(mesh_comp);

        std::vector<glm::vec3> vertices = {
            { 0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f },
            { 0.5f, 1.0f, 0.0f },
            { 0.5f, 0.5f, 1e-5f }  // extremely small height
        };

        std::vector<int> cell_vertex_indices = { 0, 1, 2, 3 };
        std::vector<int> cell_vertex_counts = { 4 };

        mesh_comp->set_vertices(vertices);
        mesh_comp->set_face_vertex_indices(cell_vertex_indices);
        mesh_comp->set_face_vertex_counts(cell_vertex_counts);

        auto solver = create_element_solver(desc);
        solver->set_geometry(*geometry);
        solver->set_boundary_condition("1.0", BoundaryCondition::Dirichlet, 0);

        EXPECT_NO_THROW(auto solution = solver->solve());
    }
}

TEST(FEMBEMProblem, Laplacian3D_StressTestLarge)
{
    // Stress test with a larger, more complex mesh
    auto geometry = std::make_shared<Geometry>();
    auto mesh_comp = std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh_comp);

    const int grid_size = 6;  // 6x6x6 = 216 vertices, 750 tetrahedra
    std::vector<glm::vec3> vertices;

    // Create vertices with some randomness to avoid perfectly regular mesh
    std::srand(42);  // Deterministic randomness
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            for (int k = 0; k < grid_size; k++) {
                float noise_scale = 0.05f;
                float noise_x =
                    (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) *
                    noise_scale;
                float noise_y =
                    (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) *
                    noise_scale;
                float noise_z =
                    (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) *
                    noise_scale;

                vertices.push_back(
                    glm::vec3(
                        static_cast<float>(i) + noise_x,
                        static_cast<float>(j) + noise_y,
                        static_cast<float>(k) + noise_z));
            }
        }
    }

    auto get_vertex_index = [grid_size](int i, int j, int k) {
        return i * grid_size * grid_size + j * grid_size + k;
    };

    std::vector<int> cell_vertex_indices;
    std::vector<int> cell_vertex_counts;

    for (int i = 0; i < grid_size - 1; i++) {
        for (int j = 0; j < grid_size - 1; j++) {
            for (int k = 0; k < grid_size - 1; k++) {
                int v000 = get_vertex_index(i, j, k);
                int v001 = get_vertex_index(i, j, k + 1);
                int v010 = get_vertex_index(i, j + 1, k);
                int v011 = get_vertex_index(i, j + 1, k + 1);
                int v100 = get_vertex_index(i + 1, j, k);
                int v101 = get_vertex_index(i + 1, j, k + 1);
                int v110 = get_vertex_index(i + 1, j + 1, k);
                int v111 = get_vertex_index(i + 1, j + 1, k + 1);

                std::vector<std::array<int, 4>> cube_tets = {
                    { v000, v001, v011, v111 }, { v000, v011, v010, v111 },
                    { v000, v010, v110, v111 }, { v000, v110, v100, v111 },
                    { v000, v100, v101, v111 }, { v000, v101, v001, v111 }
                };

                for (const auto& tet : cube_tets) {
                    for (int vertex : tet) {
                        cell_vertex_indices.push_back(vertex);
                    }
                    cell_vertex_counts.push_back(4);
                }
            }
        }
    }

    mesh_comp->set_vertices(vertices);
    mesh_comp->set_face_vertex_indices(cell_vertex_indices);
    mesh_comp->set_face_vertex_counts(cell_vertex_counts);

    ElementSolverDesc desc;
    desc.set_problem_dim(3)
        .set_element_family(ElementFamily::P_minus)
        .set_k(0)
        .set_equation_type(EquationType::Laplacian);

    auto solver = create_element_solver(desc);
    solver->set_geometry(*geometry);

    solver->set_boundary_condition(
        "sin(x)*cos(y)*sin(z)", BoundaryCondition::Dirichlet, 0);

    auto start_time = std::chrono::steady_clock::now();
    std::vector<float> solution = solver->solve();
    auto end_time = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    EXPECT_EQ(solution.size(), vertices.size());
    EXPECT_LT(duration.count(), 30000);  // Should complete within 30 seconds

    // Verify solution quality and statistics
    float min_val = *std::min_element(solution.begin(), solution.end());
    float max_val = *std::max_element(solution.begin(), solution.end());
    float sum = std::accumulate(solution.begin(), solution.end(), 0.0f);
    float mean = sum / solution.size();

    EXPECT_FALSE(std::isnan(min_val));
    EXPECT_FALSE(std::isnan(max_val));
    EXPECT_FALSE(std::isnan(mean));
    EXPECT_FALSE(std::isinf(min_val));
    EXPECT_FALSE(std::isinf(max_val));
    EXPECT_FALSE(std::isinf(mean));

    // Check solution range is reasonable
    EXPECT_LT(std::abs(max_val - min_val), 100.0f);

    mesh_comp->add_vertex_scalar_quantity("solution", solution);
}
