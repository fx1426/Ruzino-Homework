#include <time.h>

#include <Eigen/Sparse>
#include <cmath>

//#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include <pxr/usd/usdGeom/mesh.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

/*
** @brief HW4_TutteParameterization
**
** This file presents the basic framework of a "node", which processes inputs
** received from the left and outputs specific variables for downstream nodes to
** use.
** - In the first function, node_declare, you can set up the node's input and
** output variables.
** - The second function, node_exec is the execution part of the node, where we
** need to implement the node's functionality.
** Your task is to fill in the required logic at the specified locations
** within this template, especially in node_exec.
*/

namespace {

enum class WeightMode
{
    Uniform,
    Cotangent,
    Floater,
};

struct MinimalSurfaceTopology // 边界及内部顶点的拓扑信息
{
    std::vector<bool> is_boundary;
    std::vector<int> vertex_to_unknown;
    std::vector<int> unknown_to_vertex;
};

struct MinimalSurfaceSystem
{
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd bx;
    Eigen::VectorXd by;
    Eigen::VectorXd bz;
};

constexpr double kGeometricEps = 1e-12;
constexpr double kTwoPi = 6.28318530717958647692;

template <typename PointT>
Eigen::Vector3d point_to_eigen3(const PointT& point)
{
    return Eigen::Vector3d(
        static_cast<double>(point[0]),
        static_cast<double>(point[1]),
        static_cast<double>(point[2]));
}

inline double cross2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

inline double safe_angle_between(
    const Eigen::Vector3d& lhs,
    const Eigen::Vector3d& rhs)
{
    const double lhs_norm = lhs.norm();
    const double rhs_norm = rhs.norm();
    if (lhs_norm <= kGeometricEps || rhs_norm <= kGeometricEps) {
        throw std::runtime_error(
            "safe_angle_between: degenerate vector length.");
    }

    const double cos_theta = std::clamp(
        lhs.dot(rhs) / (lhs_norm * rhs_norm), -1.0, 1.0);
    return std::acos(cos_theta);
}

template <typename MeshT>
std::vector<int> collect_ordered_one_ring_neighbors(const MeshT& mesh, int vid)
{
    if (vid < 0 || vid >= static_cast<int>(mesh.n_vertices())) {
        throw std::runtime_error(
            "collect_ordered_one_ring_neighbors: vertex id out of range.");
    }

    const auto vertex_handle = mesh.vertex_handle(vid);
    if (!mesh.is_valid_handle(vertex_handle)) {
        throw std::runtime_error(
            "collect_ordered_one_ring_neighbors: invalid vertex handle.");
    }

    std::vector<int> neighbors;
    std::unordered_set<int> seen;

    for (const auto& halfedge_handle : mesh.voh_range(vertex_handle)) {
        const auto to_handle = halfedge_handle.to();
        if (!mesh.is_valid_handle(to_handle)) {
            continue;
        }

        const int neighbor_id = to_handle.idx();
        if (neighbor_id < 0 || neighbor_id == vid) {
            continue;
        }

        if (seen.insert(neighbor_id).second) {
            neighbors.push_back(neighbor_id);
        }
    }

    return neighbors;
}

template <typename MeshT>
std::vector<int> collect_neighbors(const MeshT& mesh, int vid)
{
    if (vid < 0 || vid >= static_cast<int>(mesh.n_vertices())) {
        throw std::runtime_error("collect_neighbors: vertex id out of range.");
    } // 边界编号合法

    const auto vertex_handle = mesh.vertex_handle(vid); // 把边界编号转换为顶点句柄
    if (!mesh.is_valid_handle(vertex_handle)) {
        throw std::runtime_error("collect_neighbors: invalid vertex handle.");
    }

    std::vector<int> neighbors;
    std::unordered_set<int> seen; // 用于跟踪已访问的邻居，避免重复添加
    for (const auto& halfedge_handle : mesh.voh_range(vertex_handle)) {
        const auto to_handle = halfedge_handle.to();
        if (!mesh.is_valid_handle(to_handle)) {
            continue;
        }

        const int neighbor_id = to_handle.idx();
        if (neighbor_id < 0 || neighbor_id == vid) {
            continue;
        }

        if (seen.insert(neighbor_id).second) {
            neighbors.push_back(neighbor_id);
        }
    }

    return neighbors;
}

template <typename MeshT>
std::vector<Eigen::Vector2d> build_floater_local_polygon(
    const MeshT& mesh,
    int vid,
    const std::vector<int>& ordered_neighbors)
{
    const int degree = static_cast<int>(ordered_neighbors.size());
    if (degree < 3) {
        throw std::runtime_error(
            "build_floater_local_polygon: Floater weights need degree >= 3.");
    }

    const auto center_handle = mesh.vertex_handle(vid);
    const Eigen::Vector3d center = point_to_eigen3(mesh.point(center_handle));

    std::vector<double> local_angles(degree, 0.0);
    double total_angle = 0.0;
    for (int k = 0; k < degree; ++k) {
        const int next_k = (k + 1) % degree;
        const auto vk = point_to_eigen3(
                            mesh.point(mesh.vertex_handle(ordered_neighbors[k]))) -
                        center;
        const auto vnext =
            point_to_eigen3(
                mesh.point(mesh.vertex_handle(ordered_neighbors[next_k]))) -
            center;
        local_angles[k] = safe_angle_between(vk, vnext);
        total_angle += local_angles[k];
    }

    if (total_angle <= kGeometricEps) {
        throw std::runtime_error(
            "build_floater_local_polygon: invalid one-ring total angle.");
    }

    std::vector<Eigen::Vector2d> local_polygon(degree, Eigen::Vector2d::Zero());
    double current_theta = 0.0;

    for (int k = 0; k < degree; ++k) {
        const auto point_k =
            point_to_eigen3(
                mesh.point(mesh.vertex_handle(ordered_neighbors[k]))) -
            center;
        const double radius = point_k.norm();
        if (radius <= kGeometricEps) {
            throw std::runtime_error(
                "build_floater_local_polygon: degenerate one-ring radius.");
        }

        local_polygon[k] = radius *
                           Eigen::Vector2d(
                               std::cos(current_theta), std::sin(current_theta));

        if (k + 1 < degree) {
            current_theta += kTwoPi * local_angles[k] / total_angle;
        }
    }

    return local_polygon;
}

struct FloaterIntersection
{
    int edge_index = -1;
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
};

inline FloaterIntersection find_floater_opposite_intersection(
    const std::vector<Eigen::Vector2d>& local_polygon,
    int l)
{
    const int degree = static_cast<int>(local_polygon.size());
    if (l < 0 || l >= degree) {
        throw std::runtime_error(
            "find_floater_opposite_intersection: neighbor index out of range.");
    }

    const Eigen::Vector2d pl = local_polygon[l];
    const double pl_norm = pl.norm();
    if (pl_norm <= kGeometricEps) {
        throw std::runtime_error(
            "find_floater_opposite_intersection: degenerate local point.");
    }

    const Eigen::Vector2d ray_dir = -pl / pl_norm;
    double best_t = std::numeric_limits<double>::infinity();
    FloaterIntersection best_intersection;

    for (int k = 0; k < degree; ++k) {
        const int next_k = (k + 1) % degree;
        if (k == l || next_k == l) {
            continue;
        }

        const Eigen::Vector2d a = local_polygon[k];
        const Eigen::Vector2d b = local_polygon[next_k];
        const Eigen::Vector2d edge = b - a;
        const double denom = cross2d(ray_dir, edge);
        if (std::abs(denom) <= kGeometricEps) {
            continue;
        }

        const double t = cross2d(a, edge) / denom;
        const double u = cross2d(a, ray_dir) / denom;
        if (t <= kGeometricEps) {
            continue;
        }
        if (u < -kGeometricEps || u > 1.0 + kGeometricEps) {
            continue;
        }

        const double clamped_u = std::clamp(u, 0.0, 1.0);
        const Eigen::Vector2d q = a + clamped_u * edge;
        if ((q - pl).norm() <= 1e-10) {
            continue;
        }

        if (t < best_t) {
            best_t = t;
            best_intersection.edge_index = k;
            best_intersection.point = q;
        }
    }

    if (best_intersection.edge_index < 0) {
        throw std::runtime_error(
            "find_floater_opposite_intersection: failed to find r(l).");
    }

    return best_intersection;
}

inline Eigen::Vector3d compute_origin_barycentric(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    Eigen::Matrix2d basis;
    basis.col(0) = b - a;
    basis.col(1) = c - a;

    const double det = basis.determinant();
    if (std::abs(det) <= kGeometricEps) {
        throw std::runtime_error(
            "compute_origin_barycentric: degenerate Floater triangle.");
    }

    const Eigen::Vector2d uv = basis.fullPivLu().solve(-a);
    Eigen::Vector3d barycentric(1.0 - uv[0] - uv[1], uv[0], uv[1]);

    for (int i = 0; i < 3; ++i) {
        if (std::abs(barycentric[i]) <= 1e-10) {
            barycentric[i] = 0.0;
        }
        if (barycentric[i] < -1e-8) {
            throw std::runtime_error(
                "compute_origin_barycentric: negative Floater delta.");
        }
    }

    const double sum = barycentric.sum();
    if (sum <= kGeometricEps) {
        throw std::runtime_error(
            "compute_origin_barycentric: invalid Floater delta sum.");
    }
    barycentric /= sum;
    return barycentric;
}

template <typename MeshT>
std::vector<double> compute_floater_weights(
    const MeshT& mesh,
    int vid,
    const std::vector<int>& ordered_neighbors)
{
    const int degree = static_cast<int>(ordered_neighbors.size());
    if (degree < 3) {
        throw std::runtime_error(
            "compute_floater_weights: Floater weights need degree >= 3.");
    }

    const auto local_polygon =
        build_floater_local_polygon(mesh, vid, ordered_neighbors);

    std::vector<double> lambdas(degree, 0.0);
    for (int l = 0; l < degree; ++l) {
        const auto opposite =
            find_floater_opposite_intersection(local_polygon, l);
        const int r = opposite.edge_index;
        const int r_next = (r + 1) % degree;

        const Eigen::Vector3d deltas = compute_origin_barycentric(
            local_polygon[l], local_polygon[r], local_polygon[r_next]);

        lambdas[l] += deltas[0];
        lambdas[r] += deltas[1];
        lambdas[r_next] += deltas[2];
    }

    for (double& lambda : lambdas) {
        lambda /= static_cast<double>(degree);
    }

    return lambdas;
}

template <typename MeshT>
MinimalSurfaceTopology build_minimal_surface_topology(const MeshT& mesh)
{
    MinimalSurfaceTopology topo;
    topo.is_boundary.assign(mesh.n_vertices(), false);
    topo.vertex_to_unknown.assign(mesh.n_vertices(), -1);

    for (const auto& halfedge_handle : mesh.halfedges()) {
        if (!halfedge_handle.is_boundary()) {
            continue;
        }
        topo.is_boundary[halfedge_handle.from().idx()] = true;
        topo.is_boundary[halfedge_handle.to().idx()] = true;
    }

    for (const auto& vertex_handle : mesh.vertices()) {
        const int vid = vertex_handle.idx();
        if (topo.is_boundary[vid]) {
            continue;
        }

        topo.vertex_to_unknown[vid] =
            static_cast<int>(topo.unknown_to_vertex.size());
        topo.unknown_to_vertex.push_back(vid);
    }

    return topo;
}

template <typename MeshT>
void validate_reference_mesh_compatibility(
    const MeshT& mesh,
    const MeshT& reference_mesh)
{
    if (mesh.n_vertices() != reference_mesh.n_vertices() ||
        mesh.n_edges() != reference_mesh.n_edges() ||
        mesh.n_faces() != reference_mesh.n_faces()) {
        throw std::runtime_error(
            "hw5_param: Input and Reference Mesh must share the same topology.");
    }

    for (const auto& vertex_handle : mesh.vertices()) {
        if (mesh.is_boundary(vertex_handle) !=
            reference_mesh.is_boundary(vertex_handle)) {
            throw std::runtime_error(
                "hw5_param: Input and Reference Mesh boundary layout mismatch.");
        }
    }
}

template <typename MeshT>
std::vector<double> compute_normalized_weights(
    const MeshT& reference_mesh,
    int vid,
    const std::vector<int>& neighbors,
    WeightMode mode)
{
    if (neighbors.empty()) {
        throw std::runtime_error(
            "compute_normalized_weights: vertex has no neighbors.");
    }

    std::vector<double> lambdas(neighbors.size(), 0.0); // 权重


    switch (mode) {
        case WeightMode::Uniform: {
            lambdas.assign(neighbors.size(), 1.0/neighbors.size());
            // TODO:
            // 1. Set all unnormalized weights to 1.
            // 2. Normalize by degree(i).
            // 3. Write the result into lambdas.
            break;
        }
        case WeightMode::Cotangent: {
            double sum = 0.0;
            for (size_t j = 0; j < neighbors.size(); ++j) {
                auto hen = reference_mesh.find_halfedge(
                    reference_mesh.vertex_handle(vid),
                    reference_mesh.vertex_handle(neighbors[j]));
                if (!reference_mesh.is_valid_handle(hen)) 
                {
                    throw std::runtime_error(
                        "compute_normalized_weights: invalid halfedge handle.");
                }

                auto vertex_handle_i = reference_mesh.vertex_handle(vid);
                auto vertex_handle_j = reference_mesh.vertex_handle(neighbors[j]);
                auto point_i = reference_mesh.point(vertex_handle_i);
                auto point_j = reference_mesh.point(vertex_handle_j);

                if (!hen.is_boundary()) {
                    auto k = hen.next().to();
                    auto point_k = reference_mesh.point(k);
                    double cot_angle = dot(point_i - point_k, point_j - point_k) / (point_i - point_k).cross(point_j - point_k).norm();
                    lambdas[j] += cot_angle;
                }
                
                auto hen_opp = hen.opp();
                if (!hen_opp.is_boundary()) {
                    auto l = hen_opp.next().to();
                    auto point_l = reference_mesh.point(l);
                    double cot_angle = dot(point_i - point_l, point_j - point_l) / (point_i - point_l).cross(point_j - point_l).norm();
                    lambdas[j] += cot_angle;
                }
                sum += lambdas[j];
            }

            for (size_t j = 0; j < neighbors.size(); ++j) {
                lambdas[j] /= sum;
            }
            // TODO:
            // 1. For each neighbor j, compute cot(alpha_ij) + cot(beta_ij)
            //    on the original 3D mesh.
            // 2. Handle boundary edges that only have one adjacent triangle.
            // 3. Normalize the weights.
            break;
        }
        case WeightMode::Floater: {
            const auto ordered_neighbors =
                collect_ordered_one_ring_neighbors(reference_mesh, vid);
            if (ordered_neighbors.size() != neighbors.size()) {
                throw std::runtime_error(
                    "compute_normalized_weights: inconsistent one-ring size for Floater.");
            }

            lambdas =
                compute_floater_weights(reference_mesh, vid, ordered_neighbors);
            break;
        }
        default:
            throw std::runtime_error("compute_normalized_weights: unknown mode.");
    }

    return lambdas;
}

template <typename MeshT>
MinimalSurfaceSystem assemble_minimal_surface_system(
    const MeshT& mesh,
    const MeshT& reference_mesh,
    const MinimalSurfaceTopology& topo,
    WeightMode mode)
{
    const int unknown_count = static_cast<int>(topo.unknown_to_vertex.size());

    MinimalSurfaceSystem system{
        Eigen::SparseMatrix<double>(unknown_count, unknown_count),
        Eigen::VectorXd::Zero(unknown_count),
        Eigen::VectorXd::Zero(unknown_count),
        Eigen::VectorXd::Zero(unknown_count),
    };

    for (int row = 0; row < unknown_count; ++row) {
        const int vid = topo.unknown_to_vertex[row];
        const auto neighbors = collect_neighbors(reference_mesh, vid);
        const auto lambdas =
            compute_normalized_weights(reference_mesh, vid, neighbors, mode);
        system.A.coeffRef(row, row) = 1.0;
        for (size_t j = 0; j < neighbors.size(); ++j) {
            const int neighbor_vid = neighbors[j];
            const double lambda_ij = lambdas[j];
            if (topo.is_boundary[neighbor_vid]) {
                const auto neighbor_handle = mesh.vertex_handle(neighbor_vid);
                system.bx[row] += lambda_ij * mesh.point(neighbor_handle)[0];
                system.by[row] += lambda_ij * mesh.point(neighbor_handle)[1];
                system.bz[row] += lambda_ij * mesh.point(neighbor_handle)[2];
            }
            else {
                const int col_j = topo.vertex_to_unknown[neighbor_vid];
                system.A.coeffRef(row, col_j) -= lambda_ij;
            } // 边界邻居贡献右端项，内部邻居贡献矩阵
        }
    }
    // TODO:
    // For each interior vertex i:
    // 1. neighbors = collect_neighbors(mesh, i)
    // 2. lambdas  = compute_normalized_weights(mesh, i, neighbors, mode)
    // 3. row = topo.vertex_to_unknown[i]
    // 4. Set system.A(row, row) = 1
    // 5. For each neighbor j:
    //    - if j is interior: system.A(row, col_j) -= lambda_ij
    //    - if j is boundary: system.bx/by/bz(row) += lambda_ij * boundary_pos(j)
    //
    // Note:
    // - Minimal surface keeps boundary 3D positions fixed.
    // - So RHS uses mesh.point(vertex_handle(j))[0/1/2].

    return system;
}

inline Eigen::VectorXd solve_scalar_system(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b)
{
    Eigen::VectorXd x(b.size());
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_scalar_system: matrix factorization failed.");
    }

    x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("solve_scalar_system: solving linear system failed.");
    }
    return x;
    // TODO:
    // 1. Pick a sparse solver.
    // 2. Factorize A.
    // 3. Solve A * x = b.
    // 4. Check solver status if needed.
}

template <typename MeshT>
void apply_minimal_surface_solution(
    MeshT& mesh,
    const MinimalSurfaceTopology& topo,
    const Eigen::VectorXd& solved_x,
    const Eigen::VectorXd& solved_y,
    const Eigen::VectorXd& solved_z)
{
    // TODO:
    // Write solved interior coordinates back to the mesh.
    // Boundary vertices stay unchanged.
    for (size_t row = 0; row < topo.unknown_to_vertex.size(); ++row) 
    {
        const int vid = topo.unknown_to_vertex[row];
        auto vertex_handle = mesh.vertex_handle(vid);
        mesh.point(vertex_handle)[0] = solved_x[row];
        mesh.point(vertex_handle)[1] = solved_y[row];
        mesh.point(vertex_handle)[2] = solved_z[row];
    }
}

inline Ruzino::Geometry solve_with_weight_mode(
    Ruzino::Geometry input,
    Ruzino::Geometry reference_input,
    WeightMode weight_mode)
{
    auto halfedge_mesh = operand_to_openmesh(&input);
    auto reference_halfedge_mesh = operand_to_openmesh(&reference_input);

    validate_reference_mesh_compatibility(
        *halfedge_mesh, *reference_halfedge_mesh);

    auto topo = build_minimal_surface_topology(*halfedge_mesh);
    auto system =
        assemble_minimal_surface_system(
            *halfedge_mesh, *reference_halfedge_mesh, topo, weight_mode);

    Eigen::VectorXd solved_x = solve_scalar_system(system.A, system.bx);
    Eigen::VectorXd solved_y = solve_scalar_system(system.A, system.by);
    Eigen::VectorXd solved_z = solve_scalar_system(system.A, system.bz);

    apply_minimal_surface_solution(
        *halfedge_mesh, topo, solved_x, solved_y, solved_z);

    auto geometry = Ruzino::openmesh_to_operand(halfedge_mesh.get());
    return std::move(*geometry);
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_param)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference Mesh");

    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    *some cases,
    ** additional information (e.g. other mesh geometry, other parameters) is
    *required to perform
    ** the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    *one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    b.add_output<Geometry>("Uniform");
    b.add_output<Geometry>("Cotangent");
    b.add_output<Geometry>("Floater");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");
        auto reference_input = input;

        // (TO BE UPDATED) Avoid processing the node when there is no input
        if (!input.get_component<MeshComponent>()) {
            params.set_error("Minimal Surface: Need Geometry Input.");
            return false;
        }
        if (params.has_input("Reference Mesh")) {
            reference_input = params.get_input<Geometry>("Reference Mesh");
            if (!reference_input.get_component<MeshComponent>()) {
                params.set_error(
                    "Minimal Surface: Reference Mesh must contain a mesh.");
                return false;
            }
        }

        params.set_output(
            "Uniform",
            solve_with_weight_mode(input, reference_input, WeightMode::Uniform));
        params.set_output(
            "Cotangent",
            solve_with_weight_mode(
                input, reference_input, WeightMode::Cotangent));
        params.set_output(
            "Floater",
            solve_with_weight_mode(input, reference_input, WeightMode::Floater));

        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

NODE_DECLARATION_UI(hw5_param);
NODE_DEF_CLOSE_SCOPE
