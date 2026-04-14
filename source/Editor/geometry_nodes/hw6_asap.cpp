#include <time.h>

#include <array>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

/*
** @brief HW6_ASAP_Parameterization
**
** This file introduces a dedicated ASAP node instead of overloading the ARAP
** node. The goal of the current patch is to establish the full data-flow
** skeleton and the helper interfaces needed by the one-shot ASAP solve.
*/

namespace {

constexpr double kAsapEps = 1e-12;

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

class ASAPPinConstraint
{
public:
    ASAPPinConstraint() = default;

    ASAPPinConstraint(int vertex_id, const Eigen::Vector2d& target_uv)
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

class ASAPTriangleData
{
public:
    ASAPTriangleData() = default;

    ASAPTriangleData(
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

    const Eigen::Vector2d& local_reference_vertex(int local_index) const
    {
        return local_reference_triangle_[local_index];
    }

    double cotangent_weight(int local_index) const
    {
        return cotangent_weights_[local_index];
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

    static ASAPTriangleData from_reference_face(
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

class ASAPParameterizer
{
public:
    ASAPParameterizer(
        const Ruzino::PolyMesh& reference_mesh,
        const Ruzino::PolyMesh& initial_parameterization_mesh);

    void initialize();
    void solve();
    const std::vector<Eigen::Vector2d>& solved_uv() const;
    std::shared_ptr<Ruzino::Geometry> build_planar_output_geometry() const;

private:
    int idx_ux(int vertex_id) const
    {
        return vertex_id;
    }

    int idx_uy(int vertex_id) const
    {
        return n_vertices_ + vertex_id;
    }

    int idx_a(int face_id) const
    {
        return 2 * n_vertices_ + face_id;
    }

    int idx_b(int face_id) const
    {
        return 2 * n_vertices_ + n_faces_ + face_id;
    }

    int system_size() const
    {
        return 2 * n_vertices_ + 2 * n_faces_;
    }

    void build_fixed_data();
    void build_hessian();
    std::array<int, 6> build_local_edge_global_indices(
        int face_id,
        int vertex_i,
        int vertex_j) const;
    Eigen::Matrix<double, 6, 6> build_local_edge_hessian_block(
        double dx,
        double dy,
        double weight) const;
    void assemble_local_edge_hessian(
        int face_id,
        int vertex_i,
        int vertex_j,
        double dx,
        double dy,
        double weight,
        std::vector<Eigen::Triplet<double>>& triplets) const;
    void apply_scalar_pin(
        Eigen::SparseMatrix<double>& matrix,
        Eigen::VectorXd& rhs,
        int row,
        double value) const;
    void apply_vertex_pin(
        Eigen::SparseMatrix<double>& matrix,
        Eigen::VectorXd& rhs,
        const ASAPPinConstraint& pin) const;
    void factorize_system();
    void unpack_solution(const Eigen::VectorXd& solution);

    static void validate_inputs(
        const Ruzino::PolyMesh& reference_mesh,
        const Ruzino::PolyMesh& initial_parameterization_mesh);
    static std::vector<Eigen::Vector2d> extract_initial_uv(
        const Ruzino::PolyMesh& initial_parameterization_mesh);
    static std::array<ASAPPinConstraint, 2> select_two_pins_from_boundary(
        const std::vector<int>& boundary_loop,
        const std::vector<Eigen::Vector2d>& uv);
    static std::vector<int> collect_boundary_loop(
        const Ruzino::PolyMesh& mesh);

    const Ruzino::PolyMesh& reference_mesh_;
    const Ruzino::PolyMesh& initial_parameterization_mesh_;

    int n_vertices_ = 0;
    int n_faces_ = 0;

    std::vector<ASAPTriangleData> triangles_;
    std::vector<ASAPPinConstraint> pin_constraints_;
    std::vector<Eigen::Vector2d> initial_uv_;
    std::vector<Eigen::Vector2d> solved_uv_;

    Eigen::SparseMatrix<double> hessian_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    Eigen::VectorXd rhs_;
};

void ASAPParameterizer::validate_inputs(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    if (reference_mesh.n_vertices() == 0) {
        throw std::runtime_error("hw6_asap: Reference Mesh is empty.");
    }

    if (reference_mesh.n_vertices() != initial_parameterization_mesh.n_vertices()) {
        throw std::runtime_error(
            "hw6_asap: Reference Mesh and Initial Parameterization vertex count mismatch.");
    }

    if (reference_mesh.n_faces() != initial_parameterization_mesh.n_faces()) {
        throw std::runtime_error(
            "hw6_asap: Reference Mesh and Initial Parameterization face count mismatch.");
    }

    for (const auto& face_handle : reference_mesh.faces()) {
        if (reference_mesh.valence(face_handle) != 3) {
            throw std::runtime_error("hw6_asap: only triangle meshes are supported.");
        }
    }
}

std::vector<Eigen::Vector2d> ASAPParameterizer::extract_initial_uv(
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(initial_parameterization_mesh.n_vertices());

    for (const auto& vertex_handle : initial_parameterization_mesh.vertices()) {
        const Eigen::Vector2d vertex_uv =
            point_xy_to_eigen2(initial_parameterization_mesh.point(vertex_handle));
        if (!vertex_uv.allFinite()) {
            throw std::runtime_error("hw6_asap: Initial Parameterization contains invalid UVs.");
        }
        uv.push_back(vertex_uv);
    }

    return uv;
}

std::array<Eigen::Vector2d, 3> ASAPTriangleData::build_reference_triangle(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2)
{
    const Eigen::Vector3d e01 = p1 - p0;
    const Eigen::Vector3d e02 = p2 - p0;
    const double len01 = e01.norm();
    if (len01 <= kAsapEps) {
        throw std::runtime_error("hw6_asap: degenerate reference edge.");
    }

    const double x2 = e01.dot(e02) / len01;
    const double y2_sq = e02.squaredNorm() - x2 * x2;
    if (y2_sq < -1e-10) {
        throw std::runtime_error("hw6_asap: failed to build a local reference triangle.");
    }

    const double y2 = std::sqrt(std::max(0.0, y2_sq));
    if (y2 <= kAsapEps) {
        throw std::runtime_error("hw6_asap: degenerate reference triangle.");
    }

    return {
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(len01, 0.0),
        Eigen::Vector2d(x2, y2),
    };
}

std::array<double, 3> ASAPTriangleData::compute_cotangent_weights(
    const std::array<Eigen::Vector2d, 3>& local_reference_triangle)
{
    std::array<double, 3> cotangent_weights{ 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i) {
        const int i1 = (i + 1) % 3;
        const int i2 = (i + 2) % 3;
        const Eigen::Vector2d e1 = local_reference_triangle[i1] - local_reference_triangle[i];
        const Eigen::Vector2d e2 = local_reference_triangle[i2] - local_reference_triangle[i];
        const double sin_angle = e1.x() * e2.y() - e1.y() * e2.x();
        if (std::abs(sin_angle) <= kAsapEps) {
            throw std::runtime_error(
                "hw6_asap: degenerate angle in cotangent computation.");
        }
        const double cos_angle = e1.dot(e2);
        cotangent_weights[i] = cos_angle / sin_angle;
    }

    return cotangent_weights;
}

ASAPTriangleData ASAPTriangleData::from_reference_face(
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
    if (std::abs(det) <= kAsapEps) {
        throw std::runtime_error("hw6_asap: degenerate local reference matrix.");
    }

    return ASAPTriangleData(
        face_handle.idx(),
        vertex_ids,
        reference_positions_3d,
        local_reference_triangle,
        compute_cotangent_weights(local_reference_triangle),
        0.5 * std::abs(det));
}

std::array<ASAPPinConstraint, 2> ASAPParameterizer::select_two_pins_from_boundary(
    const std::vector<int>& boundary_loop,
    const std::vector<Eigen::Vector2d>& uv)
{
    if (boundary_loop.size() < 2) {
        throw std::runtime_error("hw6_asap: need at least two boundary vertices.");
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

    if (second_pin < 0 || best_distance_sq <= kAsapEps) {
        throw std::runtime_error("hw6_asap: failed to choose two separated pins.");
    }

    return {
        ASAPPinConstraint(first_pin, uv[first_pin]),
        ASAPPinConstraint(second_pin, uv[second_pin]),
    };
}

std::vector<int> ASAPParameterizer::collect_boundary_loop(const Ruzino::PolyMesh& mesh)
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
            throw std::runtime_error("hw6_asap: ambiguous boundary successor.");
        }

        if (boundary_successor[from] == -1) {
            boundary_successor[from] = to;
            boundary_edge_count++;
        }
    }

    if (boundary_edge_count == 0) {
        throw std::runtime_error("hw6_asap: mesh has no boundary.");
    }

    int start = -1;
    for (int vertex_id = 0; vertex_id < n_vertices; ++vertex_id) {
        if (boundary_successor[vertex_id] != -1) {
            start = vertex_id;
            break;
        }
    }
    if (start < 0) {
        throw std::runtime_error("hw6_asap: failed to find boundary start.");
    }

    std::vector<int> loop;
    loop.reserve(boundary_edge_count);
    std::vector<bool> visited(n_vertices, false);

    int current = start;
    while (true) {
        if (current < 0 || current >= n_vertices) {
            throw std::runtime_error("hw6_asap: boundary traversal escaped range.");
        }

        if (visited[current]) {
            if (current == start) {
                break;
            }
            throw std::runtime_error("hw6_asap: encountered repeated boundary vertex.");
        }

        visited[current] = true;
        loop.push_back(current);

        const int next = boundary_successor[current];
        if (next == -1) {
            throw std::runtime_error("hw6_asap: broken boundary successor chain.");
        }
        current = next;
    }

    if (loop.size() != static_cast<size_t>(boundary_edge_count)) {
        throw std::runtime_error("hw6_asap: expected a single boundary loop.");
    }

    return loop;
}

std::shared_ptr<Ruzino::Geometry> ASAPParameterizer::build_planar_output_geometry() const
{
    if (static_cast<int>(solved_uv_.size()) != reference_mesh_.n_vertices()) {
        throw std::runtime_error("hw6_asap: UV size does not match vertex count.");
    }

    auto planar_mesh = std::make_shared<Ruzino::PolyMesh>(reference_mesh_);

    for (const auto& vertex_handle : planar_mesh->vertices()) {
        const int vertex_id = vertex_handle.idx();
        planar_mesh->point(vertex_handle) = Ruzino::PolyMesh::Point(
            static_cast<float>(solved_uv_[vertex_id].x()),
            static_cast<float>(solved_uv_[vertex_id].y()),
            0.0f);
    }

    auto geometry = Ruzino::openmesh_to_operand(planar_mesh.get());
    auto mesh_component = geometry->get_component<::Ruzino::MeshComponent>();
    if (!mesh_component) {
        throw std::runtime_error("hw6_asap: output geometry missing mesh component.");
    }

    std::vector<glm::vec2> texcoords(solved_uv_.size(), glm::vec2(0.0f));
    for (size_t vertex_id = 0; vertex_id < solved_uv_.size(); ++vertex_id) {
        texcoords[vertex_id] = glm::vec2(
            static_cast<float>(solved_uv_[vertex_id].x()),
            static_cast<float>(solved_uv_[vertex_id].y()));
    }
    mesh_component->set_texcoords_array(texcoords);

    return geometry;
}

ASAPParameterizer::ASAPParameterizer(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh)
    : reference_mesh_(reference_mesh),
      initial_parameterization_mesh_(initial_parameterization_mesh)
{
}

void ASAPParameterizer::initialize()
{
    validate_inputs(reference_mesh_, initial_parameterization_mesh_);

    n_vertices_ = static_cast<int>(reference_mesh_.n_vertices());
    n_faces_ = static_cast<int>(reference_mesh_.n_faces());
    initial_uv_ = extract_initial_uv(initial_parameterization_mesh_);
    solved_uv_ = initial_uv_;

    build_fixed_data();
}

void ASAPParameterizer::build_fixed_data()
{
    triangles_.assign(reference_mesh_.n_faces(), ASAPTriangleData());
    pin_constraints_.clear();

    for (const auto& face_handle : reference_mesh_.faces()) {
        triangles_[face_handle.idx()] =
            ASAPTriangleData::from_reference_face(reference_mesh_, face_handle);
    }

    const auto boundary_loop = collect_boundary_loop(reference_mesh_);
    if (!boundary_loop.empty()) {
        const auto pin_pair =
            select_two_pins_from_boundary(boundary_loop, initial_uv_);
        pin_constraints_.assign(pin_pair.begin(), pin_pair.end());
    }

    if (pin_constraints_.size() != 2) {
        throw std::runtime_error("hw6_asap: failed to initialize two pin constraints.");
    }

    build_hessian();

    rhs_ = Eigen::VectorXd::Zero(system_size());
    for (const auto& pin : pin_constraints_) {
        apply_vertex_pin(hessian_, rhs_, pin);
    }

    factorize_system();
}

std::array<int, 6> ASAPParameterizer::build_local_edge_global_indices(
    int face_id,
    int vertex_i,
    int vertex_j) const
{
    return {
        idx_ux(vertex_i),
        idx_ux(vertex_j),
        idx_uy(vertex_i),
        idx_uy(vertex_j),
        idx_a(face_id),
        idx_b(face_id),
    };
}

Eigen::Matrix<double, 6, 6> ASAPParameterizer::build_local_edge_hessian_block(
    double dx,
    double dy,
    double weight) const
{
    Eigen::Matrix<double, 6, 6> local_hessian;
    local_hessian <<
         1.0, -1.0,  0.0,  0.0, -dx, -dy,
        -1.0,  1.0,  0.0,  0.0,  dx,  dy,
         0.0,  0.0,  1.0, -1.0, -dy,  dx,
         0.0,  0.0, -1.0,  1.0,  dy, -dx,
         -dx,  dx, -dy,  dy, dx * dx + dy * dy, 0.0,
         -dy,  dy,  dx, -dx, 0.0, dx * dx + dy * dy;
    local_hessian *= weight;
    return local_hessian;
}

void ASAPParameterizer::assemble_local_edge_hessian(
    int face_id,
    int vertex_i,
    int vertex_j,
    double dx,
    double dy,
    double weight,
    std::vector<Eigen::Triplet<double>>& triplets) const
{
    const auto global_indices =
        build_local_edge_global_indices(face_id, vertex_i, vertex_j);
    const auto local_hessian =
        build_local_edge_hessian_block(dx, dy, weight);

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            triplets.emplace_back(
                global_indices[row],
                global_indices[col],
                local_hessian(row, col));
        }
    }
}

void ASAPParameterizer::build_hessian()
{
    hessian_.resize(system_size(), system_size());

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(n_faces_) * 3 * 36);

    for (const auto& triangle : triangles_) {
        for (int corner = 0; corner < 3; ++corner) {
            const auto edge_indices = triangle.local_edge_indices_from_corner(corner);
            const int vertex_i = triangle.vertex_id(edge_indices[0]);
            const int vertex_j = triangle.vertex_id(edge_indices[1]);
            const Eigen::Vector2d local_edge =
                triangle.local_reference_edge_from_corner(corner);
            const double weight = triangle.cotangent_weight(corner);

            if (std::abs(weight) <= kAsapEps) {
                continue;
            }

            assemble_local_edge_hessian(
                triangle.face_id(),
                vertex_i,
                vertex_j,
                local_edge.x(),
                local_edge.y(),
                weight,
                triplets);
        }
    }

    hessian_.setFromTriplets(
        triplets.begin(),
        triplets.end(),
        [](double lhs, double rhs) { return lhs + rhs; });
    hessian_.makeCompressed();
}

void ASAPParameterizer::apply_scalar_pin(
    Eigen::SparseMatrix<double>& matrix,
    Eigen::VectorXd& rhs,
    int row,
    double value) const
{
    matrix.makeCompressed();

    for (int outer = 0; outer < matrix.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, outer); it; ++it) {
            if (it.row() == row) {
                it.valueRef() = 0.0;
            }
        }
    }

    matrix.coeffRef(row, row) = 1.0;
    rhs[row] = value;
}

void ASAPParameterizer::apply_vertex_pin(
    Eigen::SparseMatrix<double>& matrix,
    Eigen::VectorXd& rhs,
    const ASAPPinConstraint& pin) const
{
    apply_scalar_pin(matrix, rhs, idx_ux(pin.vertex_id()), pin.target_uv().x());
    apply_scalar_pin(matrix, rhs, idx_uy(pin.vertex_id()), pin.target_uv().y());
}

void ASAPParameterizer::factorize_system()
{
    solver_.analyzePattern(hessian_);
    solver_.factorize(hessian_);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_asap: Hessian factorization failed.");
    }
}

void ASAPParameterizer::unpack_solution(const Eigen::VectorXd& solution)
{
    solved_uv_.assign(n_vertices_, Eigen::Vector2d::Zero());
    for (int vertex_id = 0; vertex_id < n_vertices_; ++vertex_id) {
        solved_uv_[vertex_id] = Eigen::Vector2d(
            solution[idx_ux(vertex_id)],
            solution[idx_uy(vertex_id)]);
    }
}

void ASAPParameterizer::solve()
{
    const Eigen::VectorXd solution = solver_.solve(rhs_);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_asap: linear solve failed.");
    }

    unpack_solution(solution);
}

const std::vector<Eigen::Vector2d>& ASAPParameterizer::solved_uv() const
{
    return solved_uv_;
}

}  // namespace

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw6_asap)
{
    b.add_input<Geometry>("Reference Mesh");
    b.add_input<Geometry>("Initial Parameterization");
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw6_asap)
{
    auto reference_input = params.get_input<Geometry>("Reference Mesh");
    auto initial_input = params.get_input<Geometry>("Initial Parameterization");

    if (!reference_input.get_component<MeshComponent>()) {
        throw std::runtime_error("Need Reference Mesh.");
    }
    if (!initial_input.get_component<MeshComponent>()) {
        throw std::runtime_error("Need Initial Parameterization.");
    }

    auto reference_halfedge_mesh = operand_to_openmesh(&reference_input);
    auto initial_halfedge_mesh = operand_to_openmesh(&initial_input);

    ASAPParameterizer parameterizer(*reference_halfedge_mesh, *initial_halfedge_mesh);
    parameterizer.initialize();
    parameterizer.solve();

    auto output = parameterizer.build_planar_output_geometry();
    params.set_output("Output", std::move(*output));
    return true;
}

NODE_DECLARATION_UI(hw6_asap);
NODE_DEF_CLOSE_SCOPE
