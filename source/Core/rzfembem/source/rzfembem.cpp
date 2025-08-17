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
        basis_->add_vertex_expression("u1");  // shape function at vertex 0
        basis_->add_vertex_expression("u2");  // shape function at vertex 1
        basis_->add_vertex_expression(
            "1 - u1 - u2");  // shape function at vertex 2
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

        // Apply boundary conditions
        apply_boundary_conditions(A, b);

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

    // Mesh data
    std::vector<pxr::GfVec3f> vertices_;
    std::vector<std::array<int, 3>> triangles_;
    std::set<int> boundary_vertices_;

    void extract_mesh_data()
    {
        if (!mesh_comp_)
            return;

        // Get vertices - ignore z coordinate for 2D solver
        auto mesh_vertices = mesh_comp_->get_vertices();
        vertices_.reserve(mesh_vertices.size());
        for (const auto& v : mesh_vertices) {
            vertices_.push_back(v);
        }

        // Get triangles
        auto indices = mesh_comp_->get_face_vertex_indices();
        triangles_.reserve(indices.size() / 3);
        for (size_t i = 0; i < indices.size(); i += 3) {
            triangles_.push_back(
                { indices[i], indices[i + 1], indices[i + 2] });
        }

        // Find boundary vertices using OpenMesh
        boundary_vertices_.clear();
        for (auto v_it = openmesh_->vertices_begin();
             v_it != openmesh_->vertices_end();
             ++v_it) {
            if (openmesh_->is_boundary(*v_it)) {
                boundary_vertices_.insert(v_it->idx());
            }
        }
    }

    std::pair<Eigen::SparseMatrix<float>, Eigen::VectorXf> assemble_system()
    {
        int n_vertices = vertices_.size();

        // Initialize sparse matrix and RHS vector
        Eigen::SparseMatrix<float> A(n_vertices, n_vertices);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(n_vertices);

        // Triplets for sparse matrix assembly
        std::vector<Eigen::Triplet<float>> triplets;

        // Assemble over all triangles
        for (const auto& triangle : triangles_) {
            assemble_element(triangle, triplets, b);
        }

        A.setFromTriplets(triplets.begin(), triplets.end());
        return { A, b };
    }

    void assemble_element(
        const std::array<int, 3>& triangle,
        std::vector<Eigen::Triplet<float>>& triplets,
        Eigen::VectorXf& b)
    {
        // Get triangle vertices in 2D (ignore z)
        std::vector<pxr::GfVec2d> tri_verts_2d;
        for (int i = 0; i < 3; i++) {
            auto& v = vertices_[triangle[i]];
            tri_verts_2d.push_back(pxr::GfVec2d(v[0], v[1]));
        }

        // Compute gradients of shape functions in physical coordinates
        auto gradients = compute_shape_gradients(tri_verts_2d);

        // Compute element area
        float area = compute_triangle_area(tri_verts_2d);

        // Assemble local stiffness matrix: ∫∇φᵢ·∇φⱼ dΩ
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float local_entry = area * (gradients[i][0] * gradients[j][0] +
                                            gradients[i][1] * gradients[j][1]);

                triplets.emplace_back(triangle[i], triangle[j], local_entry);
            }
        }

        // For Laplace equation with zero RHS, b remains zero
        // (RHS assembly would go here for non-homogeneous problems)
    }

    std::vector<std::array<float, 2>> compute_shape_gradients(
        const std::vector<pxr::GfVec2d>& tri_verts)
    {
        // For linear elements on triangle, gradients are constant
        // Shape functions: φ₀ = (a₀ + b₀x + c₀y)/2A, etc.

        double x1 = tri_verts[0][0], y1 = tri_verts[0][1];
        double x2 = tri_verts[1][0], y2 = tri_verts[1][1];
        double x3 = tri_verts[2][0], y3 = tri_verts[2][1];

        double det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        double inv_det = 1.0 / det;

        std::vector<std::array<float, 2>> gradients(3);

        // ∇φ₀ = 1/(2A) * [y2-y3, x3-x2]
        gradients[0][0] = static_cast<float>(inv_det * (y2 - y3));
        gradients[0][1] = static_cast<float>(inv_det * (x3 - x2));

        // ∇φ₁ = 1/(2A) * [y3-y1, x1-x3]
        gradients[1][0] = static_cast<float>(inv_det * (y3 - y1));
        gradients[1][1] = static_cast<float>(inv_det * (x1 - x3));

        // ∇φ₂ = 1/(2A) * [y1-y2, x2-x1]
        gradients[2][0] = static_cast<float>(inv_det * (y1 - y2));
        gradients[2][1] = static_cast<float>(inv_det * (x2 - x1));

        return gradients;
    }

    float compute_triangle_area(const std::vector<pxr::GfVec2d>& tri_verts)
    {
        double x1 = tri_verts[0][0], y1 = tri_verts[0][1];
        double x2 = tri_verts[1][0], y2 = tri_verts[1][1];
        double x3 = tri_verts[2][0], y3 = tri_verts[2][1];

        return static_cast<float>(
            0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)));
    }

    void apply_boundary_conditions(
        Eigen::SparseMatrix<float>& A,
        Eigen::VectorXf& b)
    {
        // Parse boundary expression and apply Dirichlet conditions
        fem_bem::ExpressionD boundary_func(boundary_expr_);

        for (int vertex_id : boundary_vertices_) {
            // Evaluate boundary condition at vertex
            auto& vertex = vertices_[vertex_id];
            double x = vertex[0], y = vertex[1];

            double boundary_value =
                boundary_func.evaluate_at({ { "x", x }, { "y", y } });

            // Apply strong enforcement: set row to identity, RHS to boundary
            // value Clear row
            for (Eigen::SparseMatrix<float>::InnerIterator it(A, vertex_id); it;
                 ++it) {
                if (it.row() != vertex_id) {
                    it.valueRef() = 0.0f;
                }
            }

            // Clear column
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    if (it.col() == vertex_id && it.row() != vertex_id) {
                        it.valueRef() = 0.0f;
                    }
                }
            }

            // Set diagonal and RHS
            A.coeffRef(vertex_id, vertex_id) = 1.0f;
            b[vertex_id] = static_cast<float>(boundary_value);
        }

        A.makeCompressed();
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
