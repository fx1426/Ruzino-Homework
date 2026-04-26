#include <time.h>

#include <array>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(HW6_ARAP_ENABLE_OPENMP) && defined(_OPENMP)
#include <omp.h>
#endif

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

/*
** @brief HW6_ARAP_Parameterization
**
** This file presents the basic framework of a "node", which processes inputs
** received from the left and outputs specific variables for downstream nodes to
** use.
**
** - In the first function, node_declare, you can set up the node's input and
** output variables.
**
** - The second function, node_exec is the execution part of the node, where we
** need to implement the node's functionality.
**
** - The third function generates the node's registration information, which
** eventually allows placing this node in the GUI interface.
**
** Your task is to fill in the required logic at the specified locations
** within this template, especially in node_exec.
*/

namespace {

constexpr double kArapEps = 1e-12;

constexpr bool is_openmp_available()
{
#if defined(HW6_ARAP_ENABLE_OPENMP) && defined(_OPENMP)
    return true;
#else
    return false;
#endif
}

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

// pin constraint
class ARAPPinConstraint
{
public:
    ARAPPinConstraint() = default;

    ARAPPinConstraint(int vertex_id, const Eigen::Vector2d& target_uv)
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

class ARAPTriangleData
{
public:
    ARAPTriangleData() = default;

    ARAPTriangleData(
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

    const std::array<Eigen::Vector3d, 3>& reference_positions_3d() const
    {
        return reference_positions_3d_;
    }

    const Eigen::Vector3d& reference_position_3d(int local_index) const
    {
        return reference_positions_3d_[local_index];
    }

    const std::array<Eigen::Vector2d, 3>& local_reference_triangle() const
    {
        return local_reference_triangle_;
    }

    const Eigen::Vector2d& local_reference_vertex(int local_index) const
    {
        return local_reference_triangle_[local_index];
    }

    const std::array<double, 3>& cotangent_weights() const
    {
        return cotangent_weights_;
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

    void set_cotangent_weights(const std::array<double, 3>& cotangent_weights)
    {
        cotangent_weights_ = cotangent_weights;
    }

    void set_cotangent_weight(int local_index, double cotangent_weight)
    {
        cotangent_weights_[local_index] = cotangent_weight;
    }

    double area() const
    {
        return area_;
    }

    static ARAPTriangleData from_reference_face(
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
    }; // 先存起来再算
    std::array<Eigen::Vector2d, 3> local_reference_triangle_{
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
    };
    std::array<double, 3> cotangent_weights_{ 0.0, 0.0, 0.0 };
    double area_ = 0.0;
};


// 局部参数化
class ARAPParameterizer
{
public:
    ARAPParameterizer(
        const Ruzino::PolyMesh& reference_mesh,
        const Ruzino::PolyMesh& initial_parameterization_mesh,
        int iterations,
        bool use_openmp);

    void initialize();
    void iterate();
    const std::vector<Eigen::Vector2d>& current_uv() const;
    double local_phase_time_seconds() const;
    double total_time_seconds() const;
    std::shared_ptr<Ruzino::Geometry> build_planar_output_geometry() const;

private:
    void build_fixed_data();
    void local_phase();
    void global_phase();
    void assemble_fixed_global_matrix();
    void apply_pin_constraints_to_matrix();
    void factorize_global_matrix();
    Eigen::Matrix2d compute_triangle_covariance(
        const ARAPTriangleData& triangle) const;
    Eigen::Matrix2d compute_signed_rotation(
        const Eigen::Matrix2d& covariance) const;
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

    static std::array<ARAPPinConstraint, 2> select_two_pins_from_boundary(
        const std::vector<int>& boundary_loop,
        const std::vector<Eigen::Vector2d>& uv);
    static std::vector<int> collect_boundary_loop(
        const Ruzino::PolyMesh& mesh);

    const Ruzino::PolyMesh& reference_mesh_;
    const Ruzino::PolyMesh& initial_parameterization_mesh_;
    int iterations_ = 10;
    bool use_openmp_ = false;
    double local_phase_time_seconds_ = 0.0;
    double total_time_seconds_ = 0.0;

    std::vector<ARAPTriangleData> triangles_;
    std::vector<ARAPPinConstraint> pin_constraints_;
    std::vector<Eigen::Vector2d> initial_uv_;
    std::vector<Eigen::Vector2d> current_uv_;
    std::vector<Eigen::Matrix2d> local_rotations_;

    Eigen::SparseMatrix<double> global_matrix_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> global_solver_;
};

// 检查输入的合理性
inline void ARAPParameterizer::validate_inputs(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    if (reference_mesh.n_vertices() == 0) {
        throw std::runtime_error("hw6_arap: Reference Mesh is empty.");
    }

    if (reference_mesh.n_vertices() != initial_parameterization_mesh.n_vertices()) {
        throw std::runtime_error(
            "hw6_arap: Reference Mesh and Initial Parameterization vertex count mismatch.");
    }

    if (reference_mesh.n_faces() != initial_parameterization_mesh.n_faces()) {
        throw std::runtime_error(
            "hw6_arap: Reference Mesh and Initial Parameterization face count mismatch.");
    }
    for(auto face : reference_mesh.faces()) {
        if (reference_mesh.valence(face) != 3) {
            throw std::runtime_error(
                "hw6_arap: only triangle meshes are supported.");
        }
    }
}

// 初始化uv坐标（二维）)
std::vector<Eigen::Vector2d> ARAPParameterizer::extract_initial_uv(
    const Ruzino::PolyMesh& initial_parameterization_mesh)
{
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(initial_parameterization_mesh.n_vertices());

    for (const auto& vertex_handle : initial_parameterization_mesh.vertices()) {
        const Eigen::Vector2d vertex_uv = point_xy_to_eigen2(initial_parameterization_mesh.point(vertex_handle));
        if (!vertex_uv.allFinite()) {
            throw std::runtime_error("hw6_arap: Initial Parameterization contains invalid UVs.");
        }
        uv.push_back(vertex_uv);
    }

    return uv;
}

// 构建每个三角形的局部参考三角形（二维）
std::array<Eigen::Vector2d, 3> ARAPTriangleData::build_reference_triangle(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2)
{
    const Eigen::Vector3d e01 = p1 - p0;
    const Eigen::Vector3d e02 = p2 - p0;
    const double len01 = e01.norm();
    if (len01 <= kArapEps) {
        throw std::runtime_error("hw6_arap: degenerate reference edge.");
    }

    const double x2 = e01.dot(e02) / len01;
    const double y2_sq = e02.squaredNorm() - x2 * x2;
    if (y2_sq < -1e-10) {
        throw std::runtime_error(
            "hw6_arap: failed to build a local reference triangle.");
    }

    const double y2 = std::sqrt(std::max(0.0, y2_sq));
    if (y2 <= kArapEps) {
        throw std::runtime_error("hw6_arap: degenerate reference triangle.");
    }

    return {
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(len01, 0.0),
        Eigen::Vector2d(x2, y2),
    };
}

// 构建每个三角形的数据，包括局部参考三角形、面积等
std::array<double, 3> ARAPTriangleData::compute_cotangent_weights(
    const std::array<Eigen::Vector2d, 3>& local_reference_triangle)
{
    std::array<double, 3> cotangent_weights{ 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i) {
        const int i1 = (i + 1) % 3;
        const int i2 = (i + 2) % 3;
        const Eigen::Vector2d e1 =
            local_reference_triangle[i1] - local_reference_triangle[i];
        const Eigen::Vector2d e2 =
            local_reference_triangle[i2] - local_reference_triangle[i];
        const double sin_angle = e1.x() * e2.y() - e1.y() * e2.x();
        if (std::abs(sin_angle) <= kArapEps) {
            throw std::runtime_error(
                "hw6_arap: degenerate angle in cotangent computation.");
        }
        const double cos_angle = e1.dot(e2);
        cotangent_weights[i] = cos_angle / sin_angle;
    }

    return cotangent_weights;
}

ARAPTriangleData ARAPTriangleData::from_reference_face(
    const Ruzino::PolyMesh& reference_mesh,
    const OpenMesh::SmartFaceHandle& face_handle)
{
    std::array<int, 3> vertex_ids{ -1, -1, -1 };
    int k = 0;
    for (const auto& vh : face_handle.vertices()) {
        vertex_ids[k++] = vh.idx();
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
    if (std::abs(det) <= kArapEps) {
        throw std::runtime_error("hw6_arap: degenerate local reference matrix.");
    }

    const auto cotangent_weights =
        compute_cotangent_weights(local_reference_triangle);

    return ARAPTriangleData(
        face_handle.idx(),
        vertex_ids,
        reference_positions_3d,
        local_reference_triangle,
        cotangent_weights,
        0.5 * std::abs(det));
}

// 从边界中选择两个点作为pin constraints
std::array<ARAPPinConstraint, 2> ARAPParameterizer::select_two_pins_from_boundary(
    const std::vector<int>& boundary_loop,
    const std::vector<Eigen::Vector2d>& uv)
{
    if (boundary_loop.size() < 2) {
        throw std::runtime_error("hw6_arap: need at least two boundary vertices.");
    }

    const int first_pin = boundary_loop.front(); // 选择第一个点作为第一个pin
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

    if (second_pin < 0 || best_distance_sq <= kArapEps) {
        throw std::runtime_error("hw6_arap: failed to choose two separated pins.");
    }

    return {
        ARAPPinConstraint(first_pin, uv[first_pin]),
        ARAPPinConstraint(second_pin, uv[second_pin]),
    };
}

std::shared_ptr<Ruzino::Geometry> ARAPParameterizer::build_planar_output_geometry()
    const
{
    if (static_cast<int>(current_uv_.size()) != reference_mesh_.n_vertices()) {
        throw std::runtime_error("hw6_arap: UV size does not match vertex count.");
    }

    auto planar_mesh = std::make_shared<Ruzino::PolyMesh>(reference_mesh_);

    for (const auto& vh : planar_mesh->vertices()) {
        const int vid = vh.idx();
        planar_mesh->point(vh) = Ruzino::PolyMesh::Point(
            static_cast<float>(current_uv_[vid][0]),
            static_cast<float>(current_uv_[vid][1]),
            0.0f);
    }

    auto geometry = Ruzino::openmesh_to_operand(planar_mesh.get());
    auto mesh_component = geometry->get_component<::Ruzino::MeshComponent>();
    if (!mesh_component) {
        throw std::runtime_error("hw6_arap: output geometry missing mesh component.");
    }

    std::vector<glm::vec2> texcoords(current_uv_.size(), glm::vec2(0.0f));
    for (size_t vid = 0; vid < current_uv_.size(); ++vid) {
        texcoords[vid] = glm::vec2(
            static_cast<float>(current_uv_[vid].x()),
            static_cast<float>(current_uv_[vid].y()));
    }
    mesh_component->set_texcoords_array(texcoords);

    return geometry;
}

ARAPParameterizer::ARAPParameterizer(
    const Ruzino::PolyMesh& reference_mesh,
    const Ruzino::PolyMesh& initial_parameterization_mesh,
    int iterations,
    bool use_openmp)
    : reference_mesh_(reference_mesh),
      initial_parameterization_mesh_(initial_parameterization_mesh),
      iterations_(iterations),
      use_openmp_(use_openmp)
{
}

void ARAPParameterizer::initialize()
{
    validate_inputs(reference_mesh_, initial_parameterization_mesh_);

    initial_uv_ = extract_initial_uv(initial_parameterization_mesh_);
    current_uv_ = initial_uv_;

    build_fixed_data();

    // One local rotation matrix per reference face.
    local_rotations_.assign(reference_mesh_.n_faces(), Eigen::Matrix2d::Identity());
}

void ARAPParameterizer::iterate()
{
    local_phase_time_seconds_ = 0.0;
    total_time_seconds_ = 0.0;

    const auto total_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations_; ++iter) {
        local_phase();
        global_phase();
    }
    const auto total_end = std::chrono::steady_clock::now();
    total_time_seconds_ =
        std::chrono::duration<double>(total_end - total_start).count();
}

const std::vector<Eigen::Vector2d>& ARAPParameterizer::current_uv() const
{
    return current_uv_;
}

double ARAPParameterizer::local_phase_time_seconds() const
{
    return local_phase_time_seconds_;
}

double ARAPParameterizer::total_time_seconds() const
{
    return total_time_seconds_;
}

void ARAPParameterizer::build_fixed_data()
{
    triangles_.assign(reference_mesh_.n_faces(), ARAPTriangleData());
    pin_constraints_.clear();
    for (const auto& face_handle : reference_mesh_.faces()) {
        triangles_[face_handle.idx()] =
            ARAPTriangleData::from_reference_face(reference_mesh_, face_handle);
    }

    const auto boundary_loop = collect_boundary_loop(reference_mesh_);
    if (!boundary_loop.empty()) {
        const auto pin_pair =
            select_two_pins_from_boundary(boundary_loop, initial_uv_);
        pin_constraints_.assign(pin_pair.begin(), pin_pair.end());
    }
    if (pin_constraints_.size() != 2) {
        throw std::runtime_error("hw6_arap: failed to initialize two pin constraints.");
    }

    assemble_fixed_global_matrix();
    apply_pin_constraints_to_matrix();
    factorize_global_matrix();
}

void ARAPParameterizer::local_phase()
{
    const auto local_start = std::chrono::steady_clock::now();
    const int triangle_count = static_cast<int>(triangles_.size());

#if defined(HW6_ARAP_ENABLE_OPENMP) && defined(_OPENMP)
    const bool run_parallel = use_openmp_;
#pragma omp parallel for if(run_parallel) schedule(static)
#endif
    for (int triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
        const auto& triangle = triangles_[triangle_index];
        const Eigen::Matrix2d covariance = compute_triangle_covariance(triangle);
        local_rotations_[triangle.face_id()] = compute_signed_rotation(covariance);
    }
    const auto local_end = std::chrono::steady_clock::now();
    local_phase_time_seconds_ +=
        std::chrono::duration<double>(local_end - local_start).count();
}

void ARAPParameterizer::global_phase()
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
        throw std::runtime_error("hw6_arap: failed to solve global x system.");
    }

    const Eigen::VectorXd solved_y = global_solver_.solve(rhs_y);
    if (global_solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_arap: failed to solve global y system.");
    }

    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vid = vertex_handle.idx();
        current_uv_[vid] = Eigen::Vector2d(solved_x[vid], solved_y[vid]);
    }
}

void ARAPParameterizer::assemble_fixed_global_matrix()
{
    const int n_vertices = static_cast<int>(reference_mesh_.n_vertices());
    global_matrix_.resize(n_vertices, n_vertices);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(reference_mesh_.n_edges()) * 4);

    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vid = vertex_handle.idx();
        for (const auto& neighbor_handle : reference_mesh_.vv_range(vertex_handle)) {
            const int neighbor_id = neighbor_handle.idx();
            if (vid >= neighbor_id) {
                continue;
            }

            double edge_weight = 0.0;
            const auto incident_faces = find_incident_faces_for_edge(vid, neighbor_id);
            for (const auto& face_handle : incident_faces) {
                if (!reference_mesh_.is_valid_handle(face_handle)) {
                    continue;
                }

                const auto& triangle = triangles_[face_handle.idx()];
                const int local_i = triangle.local_index_of_vertex(vid);
                const int local_j = triangle.local_index_of_vertex(neighbor_id);
                if (local_i < 0 || local_j < 0) {
                    throw std::runtime_error(
                        "hw6_arap: failed to match edge endpoints inside triangle.");
                }

                edge_weight += triangle.cotangent_weight_for_edge(local_i, local_j);
            }

            if (std::abs(edge_weight) <= kArapEps) {
                continue;
            }

            triplets.emplace_back(vid, vid, edge_weight);
            triplets.emplace_back(neighbor_id, neighbor_id, edge_weight);
            triplets.emplace_back(vid, neighbor_id, -edge_weight);
            triplets.emplace_back(neighbor_id, vid, -edge_weight);
        }
    }

    global_matrix_.setFromTriplets(
        triplets.begin(),
        triplets.end(),
        [](double lhs, double rhs) { return lhs + rhs; });
}

void ARAPParameterizer::apply_pin_constraints_to_matrix()
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

void ARAPParameterizer::factorize_global_matrix()
{
    global_solver_.analyzePattern(global_matrix_);
    global_solver_.factorize(global_matrix_);
    if (global_solver_.info() != Eigen::Success) {
        throw std::runtime_error("hw6_arap: global matrix factorization failed.");
    }
}

Eigen::Matrix2d ARAPParameterizer::compute_triangle_covariance(
    const ARAPTriangleData& triangle) const
{
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();

    for (int corner = 0; corner < 3; ++corner) {
        const int local_i = (corner + 1) % 3;
        const int local_j = (corner + 2) % 3;
        const int vertex_i = triangle.vertex_id(local_i);
        const int vertex_j = triangle.vertex_id(local_j);

        const Eigen::Vector2d uv_edge = current_uv_[vertex_i] - current_uv_[vertex_j];
        const Eigen::Vector2d ref_edge =
            triangle.local_reference_vertex(local_i) -
            triangle.local_reference_vertex(local_j);

        covariance += triangle.cotangent_weight(corner) *
                      (uv_edge * ref_edge.transpose());
    }

    return covariance;
}

Eigen::Matrix2d ARAPParameterizer::compute_signed_rotation(
    const Eigen::Matrix2d& covariance) const
{
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix2d U = svd.matrixU();
    const Eigen::Matrix2d V = svd.matrixV();

    Eigen::Matrix2d correction = Eigen::Matrix2d::Identity();
    if ((U * V.transpose()).determinant() < 0.0) {
        correction(1, 1) = -1.0;
    }

    return U * correction * V.transpose();
}

void ARAPParameterizer::assemble_global_rhs(
    Eigen::VectorXd& rhs_x,
    Eigen::VectorXd& rhs_y) const
{
    for (const auto& vertex_handle : reference_mesh_.vertices()) {
        const int vid = vertex_handle.idx();
        Eigen::Vector2d rhs_contribution = Eigen::Vector2d::Zero();

        for (const auto& neighbor_handle : reference_mesh_.vv_range(vertex_handle)) {
            const int neighbor_id = neighbor_handle.idx();
            const auto incident_faces = find_incident_faces_for_edge(vid, neighbor_id);

            for (const auto& face_handle : incident_faces) {
                if (!reference_mesh_.is_valid_handle(face_handle)) {
                    continue;
                }

                const auto& triangle = triangles_[face_handle.idx()];
                const int local_i = triangle.local_index_of_vertex(vid);
                const int local_j = triangle.local_index_of_vertex(neighbor_id);
                if (local_i < 0 || local_j < 0) {
                    throw std::runtime_error(
                        "hw6_arap: failed to match edge endpoints inside triangle.");
                }

                const double edge_weight =
                    triangle.cotangent_weight_for_edge(local_i, local_j);
                const Eigen::Vector2d reference_edge =
                    triangle.local_reference_vertex(local_i) -
                    triangle.local_reference_vertex(local_j);

                rhs_contribution +=
                    edge_weight *
                    (local_rotations_[face_handle.idx()] * reference_edge);
            }
        }

        rhs_x[vid] = rhs_contribution.x();
        rhs_y[vid] = rhs_contribution.y();
    }
}

std::array<OpenMesh::SmartFaceHandle, 2>
ARAPParameterizer::find_incident_faces_for_edge(int vertex_i, int vertex_j) const
{
    const auto vh_i = reference_mesh_.vertex_handle(vertex_i);
    const auto vh_j = reference_mesh_.vertex_handle(vertex_j);
    const auto halfedge_handle = reference_mesh_.find_halfedge(vh_i, vh_j);
    if (!reference_mesh_.is_valid_handle(halfedge_handle)) {
        throw std::runtime_error("hw6_arap: failed to find halfedge for neighbor pair.");
    }

    return { halfedge_handle.face(), halfedge_handle.opp().face() };
}

std::vector<int> ARAPParameterizer::collect_boundary_loop(
    const Ruzino::PolyMesh& mesh)
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
            throw std::runtime_error(
                "hw6_arap: ambiguous boundary successor.");
        }

        if (boundary_successor[from] == -1) {
            boundary_successor[from] = to;
            boundary_edge_count++;
        }
    }

    if (boundary_edge_count == 0) {
        throw std::runtime_error("hw6_arap: mesh has no boundary.");
    }

    int start = -1;
    for (int vid = 0; vid < n_vertices; ++vid) {
        if (boundary_successor[vid] != -1) {
            start = vid;
            break;
        }
    }
    if (start < 0) {
        throw std::runtime_error("hw6_arap: failed to find boundary start.");
    }

    std::vector<int> loop;
    loop.reserve(boundary_edge_count);
    std::vector<bool> visited(n_vertices, false);

    int current = start;
    while (true) {
        if (current < 0 || current >= n_vertices) {
            throw std::runtime_error("hw6_arap: boundary traversal escaped range.");
        }

        if (visited[current]) {
            if (current == start) {
                break;
            }
            throw std::runtime_error(
                "hw6_arap: encountered repeated boundary vertex.");
        }

        visited[current] = true;
        loop.push_back(current);

        const int next = boundary_successor[current];
        if (next == -1) {
            throw std::runtime_error("hw6_arap: broken boundary successor chain.");
        }
        current = next;
    }

    if (loop.size() != static_cast<size_t>(boundary_edge_count)) {
        throw std::runtime_error("hw6_arap: expected a single boundary loop.");
    }

    return loop;
}


}  // namespace

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw6_arap)
{
    b.add_input<Geometry>("Reference Mesh");
    b.add_input<Geometry>("Initial Parameterization");
    b.add_input<int>("Iterations").min(1).max(50).default_val(10);
    b.add_input<bool>("Use OpenMP").default_val(false);

    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    ** some cases, additional information (e.g. other mesh geometry, other
    ** parameters) is required to perform the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    ** one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    b.add_output<Geometry>("Output");
    b.add_output<float>("Local Phase Time");
    b.add_output<float>("Total Time");
    b.add_output<float>("Speedup");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    try {
        // Get the input from params
        auto reference_input = params.get_input<Geometry>("Reference Mesh");
        auto initial_input = params.get_input<Geometry>("Initial Parameterization");
        auto iterations = params.get_input<int>("Iterations");
        auto use_openmp = params.get_input<bool>("Use OpenMP");

        // Avoid processing the node when there is no input
        if (!reference_input.get_component<MeshComponent>()) {
            params.set_error("hw6_arap: Need Reference Mesh.");
            return false;
        }
        if (!initial_input.get_component<MeshComponent>()) {
            params.set_error("hw6_arap: Need Initial Parameterization.");
            return false;
        }

        /* ----------------------------- Preprocess -------------------------------
        ** Create a halfedge structure (using OpenMesh) for the input mesh. The
        ** half-edge data structure is a widely used data structure in geometric
        ** processing, offering convenient operations for traversing and modifying
        ** mesh elements.
        */
        auto reference_halfedge_mesh = operand_to_openmesh(&reference_input);
        auto initial_halfedge_mesh = operand_to_openmesh(&initial_input);

        /* ------------- [HW6_TODO] ARAP Parameterization Implementation -----------
        ** Implement ARAP mesh parameterization to minimize local distortion.
        **
        ** Steps:
        ** 1. Initial Setup: Use a HW4 parameterization result as initial setup.
        **
        ** 2. Local Phase: For each triangle, compute local orthogonal approximation
        **    (Lt) by computing SVD of Jacobian(Jt) with fixed u.
        **
        ** 3. Global Phase: With Lt fixed, update parameter coordinates(u) by
        *solving
        **    a pre-factored global sparse linear system.
        **
        ** 4. Iteration: Repeat Steps 2 and 3 to refine parameterization.
        **
        ** Note:
        **  - Fixed points' selection is crucial for ARAP and ASAP.
        **  - Encapsulate algorithms into classes for modularity.
        */

        float local_phase_time = 0.0f;
        float total_time = 0.0f;
        float speedup = 1.0f;
        std::shared_ptr<Geometry> output;
        // Keep the node usable even when this binary was built without OpenMP.
        const bool run_openmp_benchmark = use_openmp && is_openmp_available();

        if (run_openmp_benchmark) {
            // Run a serial baseline first so Speedup = T_serial / T_parallel.
            ARAPParameterizer serial_parameterizer(
                *reference_halfedge_mesh, *initial_halfedge_mesh, iterations, false);
            serial_parameterizer.initialize();
            serial_parameterizer.iterate();

            ARAPParameterizer parallel_parameterizer(
                *reference_halfedge_mesh, *initial_halfedge_mesh, iterations, true);
            parallel_parameterizer.initialize();
            parallel_parameterizer.iterate();

            output = parallel_parameterizer.build_planar_output_geometry();
            local_phase_time =
                static_cast<float>(parallel_parameterizer.local_phase_time_seconds());
            total_time =
                static_cast<float>(parallel_parameterizer.total_time_seconds());
            speedup = parallel_parameterizer.total_time_seconds() > kArapEps
                          ? static_cast<float>(
                                serial_parameterizer.total_time_seconds() /
                                parallel_parameterizer.total_time_seconds())
                          : 0.0f;
        }
        else {
            ARAPParameterizer parameterizer(
                *reference_halfedge_mesh, *initial_halfedge_mesh, iterations, false);
            parameterizer.initialize();
            parameterizer.iterate();

            output = parameterizer.build_planar_output_geometry();
            local_phase_time =
                static_cast<float>(parameterizer.local_phase_time_seconds());
            total_time = static_cast<float>(parameterizer.total_time_seconds());
        }

        // Set the output of the node
        params.set_output("Output", std::move(*output));
        params.set_output("Local Phase Time", local_phase_time);
        params.set_output("Total Time", total_time);
        params.set_output("Speedup", speedup);
        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE
