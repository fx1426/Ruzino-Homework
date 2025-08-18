#include <Eigen/Sparse>
#include <iostream>
#include <map>
#include <set>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "RZSolver/Solver.hpp"
#include "fem_bem/ElementBasis.hpp"
#include "fem_bem/api.h"
#include "fem_bem/fem_bem.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

class FEMSolver2D : public ElementSolver {
   public:
    FEMSolver2D(const ElementSolverDesc& desc) : desc_(desc)
    {
        // Create 2D finite element basis with P- family
        basis_ = fem_bem::make_fem_2d();

        // For P_minus with k=0 (H1 space), we need linear shape functions on
        // triangle vertices P-1Λ0 space has linear basis functions: u1, u2, u3
        // where u3 = 1 - u1 - u2
        basis_->add_vertex_expression(
            "1 - u1 - u2");                   // shape function at vertex 2
        basis_->add_vertex_expression("u1");  // shape function at vertex 0
        basis_->add_vertex_expression("u2");  // shape function at vertex 1
    }

    void set_geometry(const Geometry& geom) override
    {
        geometry_ = geom;

        // Get mesh component
        mesh_comp_ = geometry_.get_component<MeshComponent>();
        if (!mesh_comp_) {
            throw std::runtime_error("Geometry must have MeshComponent");
        }

        // Convert to OpenMesh for half-edge operations
        auto geom_ptr = const_cast<Geometry*>(&geometry_);
        openmesh_ = operand_to_openmesh(geom_ptr);

        // Extract mesh connectivity
        extract_mesh_data();
    }

    unsigned get_boundary_count() const override
    {
        return 1;  // Single boundary for now
    }

    void set_boundary_condition(
        const std::string& expr,
        BoundaryCondition type,
        unsigned boundary_id) override
    {
        if (type != BoundaryCondition::Dirichlet) {
            throw std::runtime_error(
                "Only Dirichlet boundary conditions supported");
        }
        boundary_expr_ = expr;
    }

    pxr::VtArray<float> solve() override
    {
        // Assemble system matrix and RHS
        auto [A, b] = assemble_system();

        // Solve linear system
        auto solution = solve_linear_system(A, b);

        return solution;
    }

   private:
    ElementSolverDesc desc_;
    Geometry geometry_;
    std::shared_ptr<MeshComponent> mesh_comp_;
    std::shared_ptr<PolyMesh> openmesh_;
    fem_bem::ElementBasisHandle basis_;
    std::string boundary_expr_;

    void extract_mesh_data()
    {
        if (!mesh_comp_)
            return;
        openmesh_ = operand_to_openmesh(&geometry_);
    }

    std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> assemble_system()
    {
        int n_vertices = openmesh_->n_vertices();

        // Initialize sparse matrix and RHS vector
        Eigen::SparseMatrix<float> A(n_vertices, n_vertices);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(n_vertices);

        // Triplets for sparse matrix assembly
        std::vector<Eigen::Triplet<float>> triplets;

        // First deal with the vertex based element.

        for (auto v_it : openmesh_->vertices()) {
            // This id
            int vertex_id = v_it.idx();  // matrix id i
            // if this vertex on the boundary, then delta_ij in the matrix, and
            // RHS = boundary_value
            if (openmesh_->is_boundary(v_it)) {
                float boundary_value = get_boundary_value(vertex_id);
                triplets.emplace_back(vertex_id, vertex_id, 1.0f);
                b[vertex_id] = boundary_value;
                continue;
            }

            // Otherwise, first find all connected faces
            for (auto f_it : openmesh_->vf_range(v_it)) {
                // Then, get the other two counter-clockwise vertex ids within
                // the face.
                auto face_vertices = openmesh_->fv_range(f_it);
                std::vector<pxr::GfVec2d> tri_verts;
                std::vector<int> face_vertex_ids;

                std::vector<int> ordered_vertices;
                for (auto fv_it : face_vertices) {
                    ordered_vertices.push_back(fv_it.idx());
                    tri_verts.push_back(pxr::GfVec2d(
                        openmesh_->point(fv_it)[0],
                        openmesh_->point(fv_it)[1]));
                }

                // Find the position of current vertex in the face
                int vertex_pos = -1;
                for (int i = 0; i < 3; i++) {
                    if (ordered_vertices[i] == vertex_id) {
                        vertex_pos = i;
                        break;
                    }
                }

                // Get the other two vertices in counter-clockwise order
                int next1 = (vertex_pos + 1) % 3;
                int next2 = (vertex_pos + 2) % 3;
                face_vertex_ids.push_back(ordered_vertices[next1]);
                face_vertex_ids.push_back(ordered_vertices[next2]);

                // Now we assemble the stiffness matrix and load vector for the
                // triangle

                auto expressions = basis_->get_vertex_expression_strings();

                auto triangle_area = compute_triangle_area(tri_verts);

                auto integral = basis_->integrate_vertex_against_with_mapping(
                    "1-u1-u2", tri_verts);
                triplets.emplace_back(
                    vertex_id, vertex_id, integral[0] * triangle_area);
                triplets.emplace_back(
                    vertex_id, face_vertex_ids[0], integral[1] * triangle_area);
                triplets.emplace_back(
                    vertex_id, face_vertex_ids[1], integral[2] * triangle_area);
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        return { A, b };
    }

    float get_boundary_value(int vertex_id)
    {
        // Evaluate boundary expression at vertex
        auto& vertex = openmesh_->point(openmesh_->vertex_handle(vertex_id));
        double x = vertex[0], y = vertex[1];

        fem_bem::Expression boundary_func(boundary_expr_);
        return static_cast<float>(
            boundary_func.evaluate_at({ { "x", x }, { "y", y } }));
    }

    float compute_triangle_area(const std::vector<pxr::GfVec2d>& tri_verts)
    {
        double x1 = tri_verts[0][0], y1 = tri_verts[0][1];
        double x2 = tri_verts[1][0], y2 = tri_verts[1][1];
        double x3 = tri_verts[2][0], y3 = tri_verts[2][1];

        return static_cast<float>(
            0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)));
    }

    pxr::VtArray<float> solve_linear_system(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b)
    {
        // Use iterative solver for sparse system
        auto solver = Solver::SolverFactory::create(
            Solver::SolverType::EIGEN_ITERATIVE_CG);

        Eigen::VectorXf x = Eigen::VectorXf::Zero(b.size());

        Solver::SolverConfig config;
        config.tolerance = 1e-6f;
        config.max_iterations = 1000;
        config.verbose = true;

        auto result = solver->solve(A, b, x, config);

        if (!result.converged) {
            std::cerr
                << "Warning: Linear solver did not converge. Final residual: "
                << result.final_residual << std::endl;
        }

        // Convert to VtArray
        pxr::VtArray<float> solution(x.size());
        for (int i = 0; i < x.size(); i++) {
            solution[i] = x[i];
        }

        return solution;
    }
};

std::shared_ptr<ElementSolver> create_element_solver(
    const ElementSolverDesc& desc)
{
    if (desc.get_problem_dim() != 2) {
        throw std::runtime_error("Only 2D problems supported");
    }

    if (desc.get_element_family() != ElementFamily::P_minus) {
        throw std::runtime_error("Only P_minus element family supported");
    }

    if (desc.get_equation_type() != EquationType::Laplacian) {
        throw std::runtime_error("Only Laplacian equation supported");
    }

    return std::make_shared<FEMSolver2D>(desc);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
