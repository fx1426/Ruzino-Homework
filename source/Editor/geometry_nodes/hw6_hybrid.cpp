#include <time.h>

#include <array>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unsupported/Eigen/Polynomials>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

/*
** @brief HW6_Hybrid_Parameterization
**
** This file introduces a dedicated Hybrid node. The current patch establishes
** the full local/global skeleton, the Hybrid-specific data interfaces, and the
** shared preprocessing pieces reused from the ARAP/ASAP path.
*/

namespace {

constexpr double kHybridEps = 1e-12;

template <typename PointT>
Eigen::Vector3d point_to_eigen3(const PointT& point)
{
    return Eigen::Vector3d(
        static_cast<double>(point[0]),
        static_cast<double>(point[1]),
        static_cast<double>(point[2]));
}

template <typename PointT>
Eigen::Vector2d point_xy_to_eigen2(const PointT& point)
{
    return Eigen::Vector2d(
        static_cast<double>(point[0]),
        static_cast<double>(point[1]));
}

class HybridPinConstraint
{
public:
    HybridPinConstraint() = default;

    HybridPinConstraint(int vertex_id, const Eigen::Vector2d& target_uv)
        : vertex_id_(vertex_id), target_uv_(target_uv)
    {
    }

    int vertex_id() const
    {
        return vertex_id_;
    }

    const Eigen::Vector2d& target_uv() const
    {
        return target_uv_;
    }

private:
    int vertex_id_ = -1;
    Eigen::Vector2d target_uv_ = Eigen::Vector2d::Zero();
};

class HybridTriangleData
{
public:
    HybridTriangleData() = default;

    HybridTriangleData(
        int face_id,
        const std::array<int, 3>& vertex_ids,
        const std::array<Eigen::Vector3d, 3>& reference_positions_3d,
        const std::array<Eigen::Vector2d, 3>& local_reference_triangle,
        const std::array<double, 3>& cotangent_weights,
        double area)
        : face_id_(face_id),
          vertex_ids_(vertex_ids),
          reference_positions_3d_(reference_positions_3d),
          local_reference_triangle_(local_reference_triangle),
          cotangent_weights_(cotangent_weights),
          area_(area)
    {
    }

    int face_id() const
    {
        return face_id_;
    }

    const std::array<int, 3>& vertex_ids() const
    {
        return vertex_ids_;
    }

    int vertex_id(int local_index) const
    {
        return vertex_ids_[local_index];
    }

    int local_index_of_vertex(int vertex_id) const
    {
        for (int local_index = 0; local_index < 3; ++local_index) {
            if (vertex_ids_[local_index] == vertex_id) {
                return local_index;
            }
        }
        return -1;
    }

    const Eigen::Vector2d& local_reference_vertex(int local_index) const
    {
        return local_reference_triangle_[local_index];
    }

    double cotangent_weight(int local_index) const
    {
        return cotangent_weights_[local_index];
    }

    double cotangent_weight_for_edge(int local_i, int local_j) const
    {
        const int local_k = 3 - local_i - local_j;
        return cotangent_weights_[local_k];
    }

    double area() const
    {
        return area_;
    }

    std::array<int, 2> local_edge_indices_from_corner(int corner) const
    {
        return { (corner + 1) % 3, (corner + 2) % 3 };
    }

    Eigen::Vector2d local_reference_edge_from_corner(int corner) const
    {
        const auto edge_indices = local_edge_indices_from_corner(corner);
        return local_reference_triangle_[edge_indices[0]] -
               local_reference_triangle_[edge_indices[1]];
    }

    static HybridTriangleData from_reference_face(
        const Ruzino::PolyMesh& reference_mesh,
        const OpenMesh::SmartFaceHandle& face_handle);

private:
    static std::array<Eigen::Vector2d, 3> build_reference_triangle(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2);

    static std::array<double, 3> compute_cotangent_weights(
        const std::array<Eigen::Vector2d, 3>& local_reference_triangle);

    int face_id_ = -1;
    std::array<int, 3> vertex_ids_{ -1, -1, -1 };
    std::array<Eigen::Vector3d, 3> reference_positions_3d_{
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
    };
    std::array<Eigen::Vector2d, 3> local_reference_triangle_{
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
    };
    std::array<double, 3> cotangent_weights_{ 0.0, 0.0, 0.0 };
    double area_ = 0.0;
};

struct SimilarityParams
{
    double a = 1.0;
    double b = 0.0;
};

class HybridParameterizer
{
public:
    HybridParameterizer(
        const Ruzino::PolyMesh& reference_mesh,
        const Ruzino::PolyMesh& initial_parameterization_mesh,
        int iterations,
        double hybrid_lambda);

    void initialize();
    void iterate();
    const std::vector<Eigen::Vector2d>& current_uv() const;
    std::shared_ptr<Ruzino::Geometry> build_planar_output_geometry() const;

private:
    void build_fixed_data();
    void local_phase();
    void global_phase();
    void assemble_fixed_global_matrix();
    void apply_pin_constraints_to_matrix();
    void factorize_global_matrix();

    SimilarityParams solve_local_similarity_for_face(
        const HybridTriangleData& triangle) const;
    void collect_local_coefficients(
        const HybridTriangleData& triangle,
        double& coefficient_C,
        double& coefficient_P,
        double& coefficient_Q) const;
    std::vector<double> solve_nonnegative_real_rho_roots(
        double coefficient_C,
        double target_R) const;
    double evaluate_local_energy(
        const HybridTriangleData& triangle,
        double a,
        double b) const;
    Eigen::Matrix2d build_similarity_matrix(
        const SimilarityParams& similarity) const;
    void assemble_global_rhs(
        Eigen::VectorXd& rhs_x,
        Eigen::VectorXd& rhs_y) const;
    std::array<OpenMesh::SmartFaceHandle, 2> find_incident_faces_for_edge(
        int vertex_i,
        int vertex_j) const;

    static void validate_inputs(
        const Ruzino::PolyMesh& reference_mesh,
        const Ruzino::PolyMesh& initial_parameterization_mesh);
    static std::vector<Eigen::Vector2d> extract_initial_uv(
        const Ruzino::PolyMesh& initial_parameterization_mesh);
    static std::array<HybridPinConstraint, 2> select_two_pins_from_boundary(
        const std::vector<int>& boundary_loop,
        const std::vector<Eigen::Vector2d>& uv);
    static std::vector<int> collect_boundary_loop(
        const Ruzino::PolyMesh& mesh);

    const Ruzino::PolyMesh& reference_mesh_;
    const Ruzino::PolyMesh& initial_parameterization_mesh_;
    int iterations_ = 10;
    double hybrid_lambda_ = 1.0;

    std::vector<HybridTriangleData> triangles_;
    std::vector<HybridPinConstraint> pin_constraints_;
    std::vector<Eigen::Vector2d> initial_uv_;
    std::vector<Eigen::Vector2d> current_uv_;
    std::vector<SimilarityParams> face_similarity_;

    Eigen::SparseMatrix<double> global_matrix_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> global_solver_;
};

void HybridParameterizer::validate_inputs(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    if (reference_mesh.n_vertices() == 0) {
        throw std::runtime_error("hw6_hybrid: Reference Mesh is empty.");
    }

    if (reference_mesh.n_vertices() != initial_parameterization_mesh.n_vertices()) {
        throw std::runtime_error(
            "hw6_hybrid: Reference Mesh and Initial Parameterization vertex count mismatch.");
    }

    if (reference_mesh.n_faces() != initial_parameterization_mesh.n_faces()) {
        throw std::runtime_error(
            "hw6_hybrid: Reference Mesh and Initial Parameterization face count mismatch.");
    }

    for (const auto& face_handle : reference_mesh.faces()) {
        if (reference_mesh.valence(face_handle) != 3) {
            throw std::runtime_error("hw6_hybrid: only triangle meshes are supported.");
        }
    }
}

std::vector<Eigen::Vector2d> HybridParameterizer::extract_initial_uv(
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(initial_parameterization_mesh.n_vertices());

    for (const auto& vertex_handle : initial_parameterization_mesh.vertices()) {
        const Eigen::Vector2d vertex_uv =
            point_xy_to_eigen2(initial_parameterization_mesh.point(vertex_handle));
        if (!vertex_uv.allFinite()) {
            throw std::runtime_error("hw6_hybrid: Initial Parameterization contains invalid UVs.");
        }
        uv.push_back(vertex_uv);
    }

    return uv;
}

std::array<Eigen::Vector2d, 3> HybridTriangleData::build_reference_triangle(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2)
{
    const Eigen::Vector3d e01 = p1 - p0;
    const Eigen::Vector3d e02 = p2 - p0;
    const double len01 = e01.norm();
    if (len01 <= kHybridEps) {
        throw std::runtime_error("hw6_hybrid: degenerate reference edge.");
    }

    const double x2 = e01.dot(e02) / len01;
    const double y2_sq = e02.squaredNorm() - x2 * x2;
    if (y2_sq < -1e-10) {
        throw std::runtime_error("hw6_hybrid: failed to build a local reference triangle.");
    }

    const double y2 = std::sqrt(std::max(0.0, y2_sq));
    if (y2 <= kHybridEps) {
        throw std::runtime_error("hw6_hybrid: degenerate reference triangle.");
    }

    return {
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(len01, 0.0),
        Eigen::Vector2d(x2, y2),
    };
}

std::array<double, 3> HybridTriangleData::compute_cotangent_weights(
    const std::array<Eigen::Vector2d, 3>& local_reference_triangle)
{
    std::array<double, 3> cotangent_weights{ 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i) {
        const int i1 = (i + 1) % 3;
        const int i2 = (i + 2) % 3;
        const Eigen::Vector2d e1 = local_reference_triangle[i1] - local_reference_triangle[i];
        const Eigen::Vector2d e2 = local_reference_triangle[i2] - local_reference_triangle[i];
        const double sin_angle = e1.x() * e2.y() - e1.y() * e2.x();
        if (std::abs(sin_angle) <= kHybridEps) {
            throw std::runtime_error(
                "hw6_hybrid: degenerate angle in cotangent computation.");
        }
        const double cos_angle = e1.dot(e2);
        cotangent_weights[i] = cos_angle / sin_angle;
    }

    return cotangent_weights;
}

HybridTriangleData HybridTriangleData::from_reference_face(
    const Ruzino::PolyMesh& reference_mesh,
    const OpenMesh::SmartFaceHandle& face_handle)
{
    std::array<int, 3> vertex_ids{ -1, -1, -1 };
    int k = 0;
    for (const auto& vertex_handle : face_handle.vertices()) {
        vertex_ids[k++] = vertex_handle.idx();
    }

    const auto p0 = point_to_eigen3(
        reference_mesh.point(reference_mesh.vertex_handle(vertex_ids[0])));
    const auto p1 = point_to_eigen3(
        reference_mesh.point(reference_mesh.vertex_handle(vertex_ids[1])));
    const auto p2 = point_to_eigen3(
        reference_mesh.point(reference_mesh.vertex_handle(vertex_ids[2])));
    const std::array<Eigen::Vector3d, 3> reference_positions_3d{ p0, p1, p2 };

    const auto local_reference_triangle = build_reference_triangle(p0, p1, p2);

    Eigen::Matrix2d dm;
    dm.col(0) = local_reference_triangle[1] - local_reference_triangle[0];
    dm.col(1) = local_reference_triangle[2] - local_reference_triangle[0];

    const double det = dm.determinant();
    if (std::abs(det) <= kHybridEps) {
        throw std::runtime_error("hw6_hybrid: degenerate local reference matrix.");
    }

    return HybridTriangleData(
        face_handle.idx(),
        vertex_ids,
        reference_positions_3d,
        local_reference_triangle,
        compute_cotangent_weights(local_reference_triangle),
        0.5 * std::abs(det));
}

std::array<HybridPinConstraint, 2> HybridParameterizer::select_two_pins_from_boundary(
    const std::vector<int>& boundary_loop,
    const std::vector<Eigen::Vector2d>& uv)
{
    if (boundary_loop.size() < 2) {
        throw std::runtime_error("hw6_hybrid: need at least two boundary vertices.");
    }

    const int first_pin = boundary_loop.front();
    double best_distance_sq = -1.0;
    int second_pin = -1;

    for (size_t i = 1; i < boundary_loop.size(); ++i) {
        const int candidate = boundary_loop[i];
        const double distance_sq = (uv[candidate] - uv[first_pin]).squaredNorm();
        if (distance_sq > best_distance_sq) {
            best_distance_sq = distance_sq;
            second_pin = candidate;
        }
    }

    if (second_pin < 0 || best_distance_sq <= kHybridEps) {
        throw std::runtime_error("hw6_hybrid: failed to choose two separated pins.");
    }

    return {
        HybridPinConstraint(first_pin, uv[first_pin]),
        HybridPinConstraint(second_pin, uv[second_pin]),
    };
}

std::vector<int> HybridParameterizer::collect_boundary_loop(const Ruzino::PolyMesh& mesh)
{
    const int n_vertices = static_cast<int>(mesh.n_vertices());
    std::vector<int> boundary_successor(n_vertices, -1);
    int boundary_edge_count = 0;

    for (const auto& halfedge_handle : mesh.halfedges()) {
        if (!halfedge_handle.is_boundary()) {
            continue;
        }

        const int from = halfedge_handle.from().idx();
        const int to = halfedge_handle.to().idx();
        if (from < 0 || to < 0) {
            continue;
        }

        if (boundary_successor[from] != -1 && boundary_successor[from] != to) {
            throw std::runtime_error("hw6_hybrid: ambiguous boundary successor.");
        }

        if (boundary_successor[from] == -1) {
            boundary_successor[from] = to;
            boundary_edge_count++;
        }
    }

    if (boundary_edge_count == 0) {
        throw std::runtime_error("hw6_hybrid: mesh has no boundary.");
    }

    int start = -1;
    for (int vertex_id = 0; vertex_id < n_vertices; ++vertex_id) {
        if (boundary_successor[vertex_id] != -1) {
            start = vertex_id;
            break;
        }
    }
    if (start < 0) {
        throw std::runtime_error("hw6_hybrid: failed to find boundary start.");
    }

    std::vector<int> loop;
    loop.reserve(boundary_edge_count);
    std::vector<bool> visited(n_vertices, false);

    int current = start;
    while (true) {
        if (current < 0 || current >= n_vertices) {
            throw std::runtime_error("hw6_hybrid: boundary traversal escaped range.");
        }

        if (visited[current]) {
            if (current == start) {
                break;
            }
            throw std::runtime_error("hw6_hybrid: encountered repeated boundary vertex.");
        }

        visited[current] = true;
        loop.push_back(current);

        const int next = boundary_successor[current];
        if (next == -1) {
            throw std::runtime_error("hw6_hybrid: broken boundary successor chain.");
        }
        current = next;
    }

    if (loop.size() != static_cast<size_t>(boundary_edge_count)) {
        throw std::runtime_error("hw6_hybrid: expected a single boundary loop.");
    }

    return loop;
}

std::shared_ptr<Ruzino::Geometry> HybridParameterizer::build_planar_output_geometry() const
{
    if (static_cast<int>(current_uv_.size()) != reference_mesh_.n_vertices()) {
        throw std::runtime_error("hw6_hybrid: UV size does not match vertex count.");
    }

    auto planar_mesh = std::make_shared<Ruzino::PolyMesh>(reference_mesh_);

    for (const auto& vertex_handle : planar_mesh->vertices()) {
        const int vertex_id = vertex_handle.idx();
        planar_mesh->point(vertex_handle) = Ruzino::PolyMesh::Point(
            static_cast<float>(current_uv_[vertex_id].x()),
            static_cast<float>(current_uv_[vertex_id].y()),
            0.0f);
    }

    auto geometry = Ruzino::openmesh_to_operand(planar_mesh.get());
    auto mesh_component = geometry->get_component<::Ruzino::MeshComponent>();
    if (!mesh_component) {
        throw std::runtime_error("hw6_hybrid: output geometry missing mesh component.");
    }

    std::vector<glm::vec2> texcoords(current_uv_.size(), glm::vec2(0.0f));
    for (size_t vertex_id = 0; vertex_id < current_uv_.size(); ++vertex_id) {
        texcoords[vertex_id] = glm::vec2(
            static_cast<float>(current_uv_[vertex_id].x()),
            static_cast<float>(current_uv_[vertex_id].y()));
    }
    mesh_component->set_texcoords_array(texcoords);

    return geometry;
}

HybridParameterizer::HybridParameterizer(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh,
    int iterations,
    double hybrid_lambda)
    : reference_mesh_(reference_mesh),
      initial_parameterization_mesh_(initial_parameterization_mesh),
      iterations_(iterations),
      hybrid_lambda_(hybrid_lambda)
{
}

void HybridParameterizer::initialize()
{
    validate_inputs(reference_mesh_, initial_parameterization_mesh_);

    initial_uv_ = extract_initial_uv(initial_parameterization_mesh_);
    current_uv_ = initial_uv_;

    build_fixed_data();

    face_similarity_.assign(reference_mesh_.n_faces(), SimilarityParams());
}

void HybridParameterizer::iterate()
{
    for (int iter = 0; iter < iterations_; ++iter) {
        local_phase();
        global_phase();
    }
}

const std::vector<Eigen::Vector2d>& HybridParameterizer::current_uv() const
{
    return current_uv_;
}

void HybridParameterizer::build_fixed_data()
{
    triangles_.assign(reference_mesh_.n_faces(), HybridTriangleData());
    pin_constraints_.clear();

    for (const auto& face_handle : reference_mesh_.faces()) {
        triangles_[face_handle.idx()] =
            HybridTriangleData::from_reference_face(reference_mesh_, face_handle);
    }

    const auto boundary_loop = collect_boundary_loop(reference_mesh_);
    if (!boundary_loop.empty()) {
        const auto pin_pair =
            select_two_pins_from_boundary(boundary_loop, initial_uv_);
        pin_constraints_.assign(pin_pair.begin(), pin_pair.end());
    }

    if (pin_constraints_.size() != 2) {
        throw std::runtime_error("hw6_hybrid: failed to initialize two pin constraints.");
    }

    assemble_fixed_global_matrix();
    apply_pin_constraints_to_matrix();
    factorize_global_matrix();
}

void HybridParameterizer::local_phase()
{
    for (const auto& triangle : triangles_) {
        face_similarity_[triangle.face_id()] =
            solve_local_similarity_for_face(triangle);
    }
}

void HybridParameterizer::global_phase()
{
    Eigen::VectorXd rhs_x = Eigen::VectorXd::Zero(reference_mesh_.n_vertices());
    Eigen::VectorXd rhs_y = Eigen::VectorXd::Zero(reference_mesh_.n_vertices());

    assemble_global_rhs(rhs_x, rhs_y);

    for (const auto& pin : pin_constraints_) {
        rhs_x[pin.vertex_id()] = pin.target_uv().x();
        rhs_y[pin.vertex_id()] = pin.target_uv().y();
    }

    const Eigen::VectorXd solved_x = global_solver_.solve(rhs_x);
    if (global_solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_hybrid: failed to solve global x system.");
    }

    const Eigen::VectorXd solved_y = global_solver_.solve(rhs_y);
    if (global_solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_hybrid: failed to solve global y system.");
    }

    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vertex_id = vertex_handle.idx();
        current_uv_[vertex_id] = Eigen::Vector2d(
            solved_x[vertex_id],
            solved_y[vertex_id]);
    }
}

void HybridParameterizer::assemble_fixed_global_matrix()
{
    const int n_vertices = static_cast<int>(reference_mesh_.n_vertices());
    global_matrix_.resize(n_vertices, n_vertices);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(reference_mesh_.n_edges()) * 4);

    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vertex_id = vertex_handle.idx();
        for (const auto& neighbor_handle : reference_mesh_.vv_range(vertex_handle)) {
            const int neighbor_id = neighbor_handle.idx();
            if (vertex_id >= neighbor_id) {
                continue;
            }

            double edge_weight = 0.0;
            const auto incident_faces =
                find_incident_faces_for_edge(vertex_id, neighbor_id);
            for (const auto& face_handle : incident_faces) {
                if (!reference_mesh_.is_valid_handle(face_handle)) {
                    continue;
                }

                const auto& triangle = triangles_[face_handle.idx()];
                const int local_i = triangle.local_index_of_vertex(vertex_id);
                const int local_j = triangle.local_index_of_vertex(neighbor_id);
                if (local_i < 0 || local_j < 0) {
                    throw std::runtime_error(
                        "hw6_hybrid: failed to match edge endpoints inside triangle.");
                }

                edge_weight += triangle.cotangent_weight_for_edge(local_i, local_j);
            }

            if (std::abs(edge_weight) <= kHybridEps) {
                continue;
            }

            triplets.emplace_back(vertex_id, vertex_id, edge_weight);
            triplets.emplace_back(neighbor_id, neighbor_id, edge_weight);
            triplets.emplace_back(vertex_id, neighbor_id, -edge_weight);
            triplets.emplace_back(neighbor_id, vertex_id, -edge_weight);
        }
    }

    global_matrix_.setFromTriplets(
        triplets.begin(),
        triplets.end(),
        [](double lhs, double rhs) { return lhs + rhs; });
}

void HybridParameterizer::apply_pin_constraints_to_matrix()
{
    global_matrix_.makeCompressed();

    for (const auto& pin : pin_constraints_) {
        const int row = pin.vertex_id();
        for (int outer = 0; outer < global_matrix_.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(global_matrix_, outer); it; ++it) {
                if (it.row() == row) {
                    it.valueRef() = 0.0;
                }
            }
        }
        global_matrix_.coeffRef(row, row) = 1.0;
    }
}

void HybridParameterizer::factorize_global_matrix()
{
    global_solver_.analyzePattern(global_matrix_);
    global_solver_.factorize(global_matrix_);
    if (global_solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_hybrid: global matrix factorization failed.");
    }
}

void HybridParameterizer::collect_local_coefficients(
    const HybridTriangleData& triangle,
    double& coefficient_C,
    double& coefficient_P,
    double& coefficient_Q) const
{
    coefficient_C = 0.0;
    coefficient_P = 0.0;
    coefficient_Q = 0.0;

    // Each cotangent weight is attached to the opposite edge in the current
    // triangle data representation.
    for (int corner = 0; corner < 3; ++corner) {
        const auto edge_indices = triangle.local_edge_indices_from_corner(corner);
        const int vertex_i = triangle.vertex_id(edge_indices[0]);
        const int vertex_j = triangle.vertex_id(edge_indices[1]);
        const Eigen::Vector2d reference_edge =
            triangle.local_reference_edge_from_corner(corner);
        const Eigen::Vector2d uv_edge =
            current_uv_[vertex_i] - current_uv_[vertex_j];
        const double weight = triangle.cotangent_weight(corner);

        coefficient_C += weight * reference_edge.squaredNorm();
        coefficient_P += weight * uv_edge.dot(reference_edge);
        coefficient_Q += weight *
                         (uv_edge.x() * reference_edge.y() -
                          uv_edge.y() * reference_edge.x());
    }
}

std::vector<double> HybridParameterizer::solve_nonnegative_real_rho_roots(
    double coefficient_C,
    double target_R) const
{
    std::vector<double> candidate_rhos;

    if (target_R <= kHybridEps) {
        candidate_rhos.push_back(0.0);
        return candidate_rhos;
    }

    const double cubic_coefficient = 2.0 * hybrid_lambda_;
    const double linear_coefficient = coefficient_C - 2.0 * hybrid_lambda_;

    if (std::abs(cubic_coefficient) <= kHybridEps) {
        if (std::abs(linear_coefficient) <= kHybridEps) {
            return candidate_rhos;
        }

        const double rho = target_R / linear_coefficient;
        if (rho >= -kHybridEps) {
            candidate_rhos.push_back(std::max(0.0, rho));
        }
        return candidate_rhos;
    }

    Eigen::Matrix<double, 4, 1> polynomial;
    polynomial <<
        -target_R,
        linear_coefficient,
        0.0,
        cubic_coefficient;

    Eigen::PolynomialSolver<double, 3> solver(polynomial);
    const auto roots = solver.roots();

    constexpr double kRootImagTolerance = 1e-9;
    constexpr double kRootMergeTolerance = 1e-8;

    for (Eigen::Index root_id = 0; root_id < roots.size(); ++root_id) {
        const auto& root = roots[root_id];
        if (std::abs(root.imag()) > kRootImagTolerance) {
            continue;
        }

        const double rho = root.real();
        if (rho < -kHybridEps) {
            continue;
        }

        const double clamped_rho = std::max(0.0, rho);
        bool duplicated = false;
        for (const double existing_rho : candidate_rhos) {
            if (std::abs(existing_rho - clamped_rho) <= kRootMergeTolerance) {
                duplicated = true;
                break;
            }
        }
        if (!duplicated) {
            candidate_rhos.push_back(clamped_rho);
        }
    }

    std::sort(candidate_rhos.begin(), candidate_rhos.end());
    return candidate_rhos;
}

double HybridParameterizer::evaluate_local_energy(
    const HybridTriangleData& triangle,
    double a,
    double b) const
{
    const Eigen::Matrix2d similarity = build_similarity_matrix({ a, b });
    double energy = 0.0;

    for (int corner = 0; corner < 3; ++corner) {
        const auto edge_indices = triangle.local_edge_indices_from_corner(corner);
        const int vertex_i = triangle.vertex_id(edge_indices[0]);
        const int vertex_j = triangle.vertex_id(edge_indices[1]);
        const Eigen::Vector2d reference_edge =
            triangle.local_reference_edge_from_corner(corner);
        const Eigen::Vector2d uv_edge =
            current_uv_[vertex_i] - current_uv_[vertex_j];
        const Eigen::Vector2d residual =
            uv_edge - similarity * reference_edge;
        const double weight = triangle.cotangent_weight(corner);

        energy += weight * residual.squaredNorm();
    }

    const double scale_sq = a * a + b * b;
    energy += hybrid_lambda_ * (scale_sq - 1.0) * (scale_sq - 1.0);
    return 0.5 * energy;
}

SimilarityParams HybridParameterizer::solve_local_similarity_for_face(
    const HybridTriangleData& triangle) const
{
    double coefficient_C = 0.0;
    double coefficient_P = 0.0;
    double coefficient_Q = 0.0;
    collect_local_coefficients(
        triangle, coefficient_C, coefficient_P, coefficient_Q);

    const double target_R =
        std::sqrt(coefficient_P * coefficient_P + coefficient_Q * coefficient_Q);
    if (target_R <= kHybridEps) {
        return { 0.0, 0.0 };
    }

    const auto candidate_rhos =
        solve_nonnegative_real_rho_roots(coefficient_C, target_R);
    if (candidate_rhos.empty()) {
        throw std::runtime_error(
            "hw6_hybrid: no nonnegative real rho root found for a face.");
    }

    double best_energy = std::numeric_limits<double>::infinity();
    SimilarityParams best_similarity{ 0.0, 0.0 };

    for (const double rho : candidate_rhos) {
        const double a = rho * coefficient_P / target_R;
        const double b = rho * coefficient_Q / target_R;
        const double local_energy = evaluate_local_energy(triangle, a, b);
        if (local_energy < best_energy) {
            best_energy = local_energy;
            best_similarity = { a, b };
        }
    }

    return best_similarity;
}

Eigen::Matrix2d HybridParameterizer::build_similarity_matrix(
    const SimilarityParams& similarity) const
{
    Eigen::Matrix2d matrix;
    matrix <<
        similarity.a,  similarity.b,
        -similarity.b, similarity.a;
    return matrix;
}

void HybridParameterizer::assemble_global_rhs(
    Eigen::VectorXd& rhs_x,
    Eigen::VectorXd& rhs_y) const
{
    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vertex_id = vertex_handle.idx();
        Eigen::Vector2d rhs_contribution = Eigen::Vector2d::Zero();

        for (const auto& neighbor_handle : reference_mesh_.vv_range(vertex_handle)) {
            const int neighbor_id = neighbor_handle.idx();
            const auto incident_faces =
                find_incident_faces_for_edge(vertex_id, neighbor_id);

            for (const auto& face_handle : incident_faces) {
                if (!reference_mesh_.is_valid_handle(face_handle)) {
                    continue;
                }

                const auto& triangle = triangles_[face_handle.idx()];
                const int local_i = triangle.local_index_of_vertex(vertex_id);
                const int local_j = triangle.local_index_of_vertex(neighbor_id);
                if (local_i < 0 || local_j < 0) {
                    throw std::runtime_error(
                        "hw6_hybrid: failed to match edge endpoints inside triangle.");
                }

                const double edge_weight =
                    triangle.cotangent_weight_for_edge(local_i, local_j);
                const Eigen::Vector2d reference_edge =
                    triangle.local_reference_vertex(local_i) -
                    triangle.local_reference_vertex(local_j);
                const Eigen::Matrix2d similarity =
                    build_similarity_matrix(face_similarity_[face_handle.idx()]);

                rhs_contribution +=
                    edge_weight * (similarity * reference_edge);
            }
        }

        rhs_x[vertex_id] = rhs_contribution.x();
        rhs_y[vertex_id] = rhs_contribution.y();
    }
}

std::array<OpenMesh::SmartFaceHandle, 2>
HybridParameterizer::find_incident_faces_for_edge(int vertex_i, int vertex_j) const
{
    const auto vh_i = reference_mesh_.vertex_handle(vertex_i);
    const auto vh_j = reference_mesh_.vertex_handle(vertex_j);
    const auto halfedge_handle = reference_mesh_.find_halfedge(vh_i, vh_j);
    if (!reference_mesh_.is_valid_handle(halfedge_handle)) {
        throw std::runtime_error("hw6_hybrid: failed to find halfedge for neighbor pair.");
    }

    return { halfedge_handle.face(), halfedge_handle.opp().face() };
}

}  // namespace

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw6_hybrid)
{
    b.add_input<Geometry>("Reference Mesh");
    b.add_input<Geometry>("Initial Parameterization");
    b.add_input<int>("Iterations").min(1).max(50).default_val(10);
    b.add_input<float>("Lambda").min(0.0f).default_val(1.0f);
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw6_hybrid)
{
    auto reference_input = params.get_input<Geometry>("Reference Mesh");
    auto initial_input = params.get_input<Geometry>("Initial Parameterization");
    auto iterations = params.get_input<int>("Iterations");
    auto hybrid_lambda = params.get_input<float>("Lambda");

    if (!reference_input.get_component<MeshComponent>()) {
        throw std::runtime_error("Need Reference Mesh.");
    }
    if (!initial_input.get_component<MeshComponent>()) {
        throw std::runtime_error("Need Initial Parameterization.");
    }

    auto reference_halfedge_mesh = operand_to_openmesh(&reference_input);
    auto initial_halfedge_mesh = operand_to_openmesh(&initial_input);

    HybridParameterizer parameterizer(
        *reference_halfedge_mesh,
        *initial_halfedge_mesh,
        iterations,
        static_cast<double>(hybrid_lambda));
    parameterizer.initialize();
    parameterizer.iterate();

    auto output = parameterizer.build_planar_output_geometry();
    params.set_output("Output", std::move(*output));
    return true;
}

NODE_DECLARATION_UI(hw6_hybrid);
NODE_DEF_CLOSE_SCOPE
