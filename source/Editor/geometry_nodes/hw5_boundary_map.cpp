#include <cmath>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

/*
** @brief HW4_TutteParameterization
**
** This file contains two nodes whose primary function is to map the boundary of
*a mesh to a plain
** convex closed curve (circle of square), setting the stage for subsequent
*Laplacian equation
** solution and mesh parameterization tasks.
**
** Key to this node's implementation is the adept manipulation of half-edge data
*structures
** to identify and modify the boundary of the mesh.
**
** Task Overview:
** - The two execution functions (node_map_boundary_to_square_exec,
** node_map_boundary_to_circle_exec) require an update to accurately map the
*mesh boundary to a and
** circles. This entails identifying the boundary edges, evenly distributing
*boundary vertices along
** the square's perimeter, and ensuring the internal vertices' positions remain
*unchanged.
** - A focus on half-edge data structures to efficiently traverse and modify
*mesh boundaries.
*/

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBoundaryLengthEps = 1e-12;

template <typename MeshT>
std::vector<int> collect_boundary_loop(const MeshT& mesh)
{
    const int n_vertices = static_cast<int>(mesh.n_vertices());
    std::vector<int> boundary_successor(n_vertices, -1);
    int boundary_edge_count = 0;

    for (const auto& halfedge_handle : mesh.halfedges()) {
        if (!halfedge_handle.is_boundary()) {
            continue;
        } // 只处理边界半边

        const int from = halfedge_handle.from().idx();
        const int to = halfedge_handle.to().idx();
        if (from < 0 || to < 0) {
            continue;
        }

        if (boundary_successor[from] != -1 && boundary_successor[from] != to) {
            throw std::runtime_error(
                "collect_boundary_loop: ambiguous boundary successor.");
        }

        if (boundary_successor[from] == -1) {
            boundary_successor[from] = to; // 记录边界后继关系
            boundary_edge_count++;
        }
    }

    if (boundary_edge_count == 0) {
        throw std::runtime_error("collect_boundary_loop: mesh has no boundary.");
    }

    int start = -1;
    for (int vid = 0; vid < n_vertices; ++vid) {
        if (boundary_successor[vid] != -1) {
            start = vid;
            break;
        }
    } // 随便找一个边界顶点作为起点

    if (start == -1) {
        throw std::runtime_error(
            "collect_boundary_loop: failed to find boundary start.");
    }

    std::vector<int> loop;
    loop.reserve(boundary_edge_count);
    std::vector<bool> visited(n_vertices, false);

    int current = start;
    while (true) {
        if (current < 0 || current >= n_vertices) {
            throw std::runtime_error(
                "collect_boundary_loop: boundary traversal escaped vertex range.");
        }

        if (visited[current]) {
            if (current == start) {
                break;
            }
            throw std::runtime_error(
                "collect_boundary_loop: encountered repeated boundary vertex.");
        } // 检测边界循环中的重复顶点，确保形成一个简单的闭合边界

        visited[current] = true;
        loop.push_back(current);

        const int next = boundary_successor[current];
        if (next == -1) {
            throw std::runtime_error(
                "collect_boundary_loop: broken boundary successor chain.");
        }
        current = next; // 沿着边界后继关系继续遍历
    }

    if (loop.size() != static_cast<size_t>(boundary_edge_count)) {
        throw std::runtime_error(
            "collect_boundary_loop: expected a single boundary loop.");
    }

    return loop;
}

template <typename MeshT>
std::vector<double> compute_boundary_prefix_ratios(
    const MeshT& mesh,
    const std::vector<int>& boundary_loop)
{
    if (boundary_loop.size() < 2) {
        throw std::runtime_error(
            "compute_boundary_prefix_ratios: boundary loop is too short.");
    }

    const int loop_size = static_cast<int>(boundary_loop.size());
    std::vector<double> edge_lengths(loop_size, 0.0);
    double total_length = 0.0;

    for (int i = 0; i < loop_size; ++i) {
        const int from = boundary_loop[i];
        const int to = boundary_loop[(i + 1) % loop_size];
        const auto from_handle = mesh.vertex_handle(from);
        const auto to_handle = mesh.vertex_handle(to);
        if (!mesh.is_valid_handle(from_handle) || !mesh.is_valid_handle(to_handle)) {
            throw std::runtime_error(
                "compute_boundary_prefix_ratios: invalid boundary vertex handle.");
        }

        edge_lengths[i] =
            (mesh.point(to_handle) - mesh.point(from_handle)).length();
        total_length += edge_lengths[i];
    }

    if (total_length <= kBoundaryLengthEps) {
        throw std::runtime_error(
            "compute_boundary_prefix_ratios: degenerate boundary length.");
    }

    std::vector<double> ratios(loop_size, 0.0);
    double prefix_length = 0.0;
    for (int i = 1; i < loop_size; ++i) {
        prefix_length += edge_lengths[i - 1];
        ratios[i] = std::min(prefix_length / total_length, 1.0 - 1e-12);
    }

    return ratios;
}

inline double wrap_unit_interval(double t)
{
    const double wrapped = t - std::floor(t);
    return wrapped < 0.0 ? wrapped + 1.0 : wrapped;
}

inline Eigen::Vector2d map_ratio_to_circle(double t)
{
    t = wrap_unit_interval(t);
    const double theta = 2.0 * kPi * t;
    return Eigen::Vector2d(
        0.5 + 0.5 * std::cos(theta),
        0.5 + 0.5 * std::sin(theta));
}

inline Eigen::Vector2d map_ratio_to_square(double t)
{
    t = wrap_unit_interval(t);

    if (t < 0.25) {
        const double s = 4.0 * t;
        return Eigen::Vector2d(s, 0.0);
    }
    if (t < 0.5) {
        const double s = 4.0 * (t - 0.25);
        return Eigen::Vector2d(1.0, s);
    }
    if (t < 0.75) {
        const double s = 4.0 * (t - 0.5);
        return Eigen::Vector2d(1.0 - s, 1.0);
    }

    const double s = 4.0 * (t - 0.75);
    return Eigen::Vector2d(0.0, 1.0 - s);
}

template <typename MeshT>
void apply_boundary_uv_positions(
    MeshT& mesh,
    const std::vector<int>& boundary_loop,
    const std::vector<Eigen::Vector2d>& boundary_uv)
{
    if (boundary_loop.size() != boundary_uv.size()) {
        throw std::runtime_error(
            "apply_boundary_uv_positions: boundary loop and uv size mismatch.");
    }

    for (size_t i = 0; i < boundary_loop.size(); ++i) {
        const auto vertex_handle = mesh.vertex_handle(boundary_loop[i]);
        if (!mesh.is_valid_handle(vertex_handle)) {
            throw std::runtime_error(
                "apply_boundary_uv_positions: invalid boundary vertex handle.");
        }

        mesh.point(vertex_handle)[0] = boundary_uv[i][0];
        mesh.point(vertex_handle)[1] = boundary_uv[i][1];
        mesh.point(vertex_handle)[2] = 0.0;
    }
}

template <typename MeshT, typename MapperT>
void map_boundary_by_arc_length(MeshT& mesh, MapperT&& map_ratio_to_uv)
{
    const auto boundary_loop = collect_boundary_loop(mesh);
    const auto ratios = compute_boundary_prefix_ratios(mesh, boundary_loop);

    if (boundary_loop.size() != ratios.size()) {
        throw std::runtime_error(
            "map_boundary_by_arc_length: boundary data size mismatch.");
    }

    std::vector<Eigen::Vector2d> boundary_uv;
    boundary_uv.reserve(boundary_loop.size());
    for (double t : ratios) {
        boundary_uv.push_back(map_ratio_to_uv(t));
    }

    apply_boundary_uv_positions(mesh, boundary_loop, boundary_uv);
}

}  // namespace

NODE_DEF_OPEN_SCOPE

/*
** HW4_TODO: Node to map the mesh boundary to a circle.
*/

NODE_DECLARATION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_circle_boundary_mapping)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");

        // (TO BE UPDATED) Avoid processing the node when there is no input
        if (!input.get_component<MeshComponent>()) {
            params.set_error("Boundary Mapping: Need Geometry Input.");
            return false;
        }

        /* ----------------------------- Preprocess -------------------------------
        ** Create a halfedge structure (using OpenMesh) for the input mesh. The
        ** half-edge data structure is a widely used data structure in geometric
        ** processing, offering convenient operations for traversing and modifying
        ** mesh elements.
        */
        auto halfedge_mesh = operand_to_openmesh(&input);

        map_boundary_by_arc_length(
            *halfedge_mesh,
            [](double t) { return map_ratio_to_circle(t); });

        /* ----------- [HW4_TODO] TASK 2.1: Boundary Mapping (to circle)
         *------------
         ** In this task, you are required to map the boundary of the mesh to a
         *circle
         ** shape while ensuring the internal vertices remain unaffected. This step
         *is
         ** crucial for setting up the mesh for subsequent parameterization tasks.
         **
         ** Algorithm Pseudocode for Boundary Mapping to Circle
         ** ------------------------------------------------------------------------
         ** 1. Identify the boundary loop(s) of the mesh using the half-edge
         *structure.
         **
         ** 2. Calculate the total length of the boundary loop to determine the
         *spacing
         **    between vertices when mapped to a square.
         **
         ** 3. Sequentially assign each boundary vertex a new position along the
         *square's
         **    perimeter, maintaining the calculated spacing to ensure proper
         *distribution.
         **
         ** 4. Keep the interior vertices' positions unchanged during this process.
         **
         ** Note: How to distribute the points on the circle?
         **
         ** Note: It would be better to normalize the boundary to a unit circle in
         *[0,1]x[0,1] for
         ** texture mapping.
         */

        /* ----------------------------- Postprocess ------------------------------
        ** Convert the result mesh from the halfedge structure back to Geometry
        *format as the node's
        ** output.
        */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        // Set the output of the nodes
        params.set_output("Output", std::move(*geometry));
        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

/*
** HW4_TODO: Node to map the mesh boundary to a square.
*/

NODE_DECLARATION_FUNCTION(hw5_square_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_square_boundary_mapping)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");

        // (TO BE UPDATED) Avoid processing the node when there is no input
        if (!input.get_component<MeshComponent>()) {
            params.set_error("Input does not contain a mesh");
            return false;
        }

        /* ----------------------------- Preprocess -------------------------------
        ** Create a halfedge structure (using OpenMesh) for the input mesh.
        */
        auto halfedge_mesh = operand_to_openmesh(&input);

        map_boundary_by_arc_length(
            *halfedge_mesh,
            [](double t) { return map_ratio_to_square(t); });

        /* ----------- [HW4_TODO] TASK 2.2: Boundary Mapping (to square)
         *------------
         ** In this task, you are required to map the boundary of the mesh to a
         *circle
         ** shape while ensuring the internal vertices remain unaffected.
         **
         ** Algorithm Pseudocode for Boundary Mapping to Square
         ** ------------------------------------------------------------------------
         ** (omitted)
         **
         ** Note: Can you perserve the 4 corners of the square after boundary
         *mapping?
         **
         ** Note: It would be better to normalize the boundary to a unit circle in
         *[0,1]x[0,1] for
         ** texture mapping.
         */

        /* ----------------------------- Postprocess ------------------------------
        ** Convert the result mesh from the halfedge structure back to Geometry
        *format as the node's
        ** output.
        */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        // Set the output of the nodes
        params.set_output("Output", std::move(*geometry));
        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

NODE_DECLARATION_UI(hw5_circle_boundary_mapping);
NODE_DECLARATION_UI(hw5_square_boundary_mapping);
NODE_DEF_CLOSE_SCOPE
