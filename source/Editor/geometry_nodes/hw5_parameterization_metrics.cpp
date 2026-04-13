#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glm/glm.hpp>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

namespace {

constexpr double kMetricEps = 1e-12;

struct FaceMetrics
{
    double avg_angle_distortion = 0.0;
    double max_angle_distortion = 0.0;
    double area_3d = 0.0;
    double signed_area_uv = 0.0;
    bool flipped = false;
};

struct ParameterizationMetrics
{
    double max_angle_distortion = 0.0;
    double avg_angle_distortion = 0.0;
    double avg_area_distortion = 0.0;
    int flip_count = 0;
    double flip_ratio = 0.0;

    std::vector<float> face_avg_angle_distortion;
    std::vector<float> face_max_angle_distortion;
    std::vector<float> face_area_distortion;
    std::vector<float> face_flip_mask;
};

inline Eigen::Vector3d glm_to_eigen3(const glm::vec3& v)
{
    return Eigen::Vector3d(
        static_cast<double>(v.x),
        static_cast<double>(v.y),
        static_cast<double>(v.z));
}

inline Eigen::Vector2d glm_to_eigen2_xy(const glm::vec3& v)
{
    return Eigen::Vector2d(
        static_cast<double>(v.x),
        static_cast<double>(v.y));
}

template <typename DerivedA, typename DerivedB>
inline double safe_angle_between(
    const Eigen::MatrixBase<DerivedA>& lhs,
    const Eigen::MatrixBase<DerivedB>& rhs)
{
    const double lhs_norm = lhs.norm();
    const double rhs_norm = rhs.norm();
    if (lhs_norm <= kMetricEps || rhs_norm <= kMetricEps) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: degenerate edge vector.");
    }

    const double cos_theta =
        std::clamp(lhs.dot(rhs) / (lhs_norm * rhs_norm), -1.0, 1.0);
    return std::acos(cos_theta);
}

inline double triangle_area_3d(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    return 0.5 * (b - a).cross(c - a).norm();
}

inline double signed_triangle_area_2d(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    return  0.5 * ((b - a).x() * (c - a).y() - (b - a).y() * (c - a).x());
}

inline std::array<double, 3> triangle_angles_3d(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    std::array<double, 3> angles;
    angles[0] = safe_angle_between(b - a, c - a);
    angles[1] = safe_angle_between(a - b, c - b);
    angles[2] = safe_angle_between(a - c, b - c);
    return angles;
}

inline std::array<double, 3> triangle_angles_2d(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    std::array<double, 3> angles;
    angles[0] = safe_angle_between(b - a, c - a);
    angles[1] = safe_angle_between(a - b, c - b);
    angles[2] = safe_angle_between(a - c, b - c);
    return angles;
}

inline void validate_metric_inputs(
    const ::Ruzino::MeshComponent& parameterized_mesh,
    const ::Ruzino::MeshComponent& reference_mesh)
{
    if (parameterized_mesh.get_vertices().size() !=
        reference_mesh.get_vertices().size()) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: vertex count mismatch.");
    }

    if (parameterized_mesh.get_face_vertex_counts() !=
            reference_mesh.get_face_vertex_counts() ||
        parameterized_mesh.get_face_vertex_indices() !=
            reference_mesh.get_face_vertex_indices()) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: face topology mismatch.");
    }

    for (int count : parameterized_mesh.get_face_vertex_counts()) {
        if (count != 3) {
            throw std::runtime_error(
                "hw5_parameterization_metrics: only triangle meshes are supported.");
        }
    }
}

inline std::vector<std::array<int, 3>> collect_triangle_indices(
    const ::Ruzino::MeshComponent& mesh_component)
{
    std::vector<std::array<int, 3>> triangles;
    const auto face_counts = mesh_component.get_face_vertex_counts();
    const auto face_indices = mesh_component.get_face_vertex_indices();

    size_t cursor = 0;
    triangles.reserve(face_counts.size());
    for (int count : face_counts) {
        if (count != 3) {
            throw std::runtime_error(
                "hw5_parameterization_metrics: non-triangular face encountered.");
        }

        triangles.push_back(
            {face_indices[cursor], face_indices[cursor + 1], face_indices[cursor + 2]});
        cursor += 3;
    }

    return triangles;
}

inline FaceMetrics evaluate_face_metrics(
    const Eigen::Vector3d& p0_3d,
    const Eigen::Vector3d& p1_3d,
    const Eigen::Vector3d& p2_3d,
    const Eigen::Vector2d& p0_uv,
    const Eigen::Vector2d& p1_uv,
    const Eigen::Vector2d& p2_uv)
{
    FaceMetrics face_metrics;
    double area_3d = triangle_area_3d(p0_3d, p1_3d, p2_3d);
    double signed_area_uv = signed_triangle_area_2d(p0_uv, p1_uv, p2_uv);
    bool flipped = signed_area_uv > 0.0;

    std::array<double, 3> angles_3d = triangle_angles_3d(p0_3d, p1_3d, p2_3d);
    std::array<double, 3> angles_uv = triangle_angles_2d(p0_uv, p1_uv, p2_uv);
    
    face_metrics.avg_angle_distortion =
        (std::abs(angles_3d[0] - angles_uv[0]) +
         std::abs(angles_3d[1] - angles_uv[1]) +
         std::abs(angles_3d[2] - angles_uv[2])) / 3.0;
    face_metrics.max_angle_distortion =
        std::max({std::abs(angles_3d[0] - angles_uv[0]),
                  std::abs(angles_3d[1] - angles_uv[1]),
                  std::abs(angles_3d[2] - angles_uv[2])});

    face_metrics.area_3d = area_3d;
    face_metrics.signed_area_uv = signed_area_uv;
    face_metrics.flipped = flipped;

    return face_metrics;
}

inline ParameterizationMetrics evaluate_parameterization_metrics(
    const ::Ruzino::MeshComponent& parameterized_mesh,
    const ::Ruzino::MeshComponent& reference_mesh)
{
    validate_metric_inputs(parameterized_mesh, reference_mesh);

    ParameterizationMetrics metrics;
    const auto triangles = collect_triangle_indices(reference_mesh);
    if (triangles.empty()) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: empty triangle mesh.");
    }

    const auto& reference_vertices = reference_mesh.get_vertices();
    const auto& parameterized_vertices = parameterized_mesh.get_vertices();

    metrics.face_avg_angle_distortion.assign(triangles.size(), 0.0f);
    metrics.face_max_angle_distortion.assign(triangles.size(), 0.0f);
    metrics.face_area_distortion.assign(triangles.size(), 0.0f);
    metrics.face_flip_mask.assign(triangles.size(), 0.0f);

    std::vector<FaceMetrics> face_metrics_buffer(triangles.size());
    double total_area_3d = 0.0;
    double total_area_uv = 0.0;

    for (size_t face_id = 0; face_id < triangles.size(); ++face_id) {
        const auto tri = triangles[face_id];

        const auto p0_3d = glm_to_eigen3(reference_vertices[tri[0]]);
        const auto p1_3d = glm_to_eigen3(reference_vertices[tri[1]]);
        const auto p2_3d = glm_to_eigen3(reference_vertices[tri[2]]);

        const auto p0_uv = glm_to_eigen2_xy(parameterized_vertices[tri[0]]);
        const auto p1_uv = glm_to_eigen2_xy(parameterized_vertices[tri[1]]);
        const auto p2_uv = glm_to_eigen2_xy(parameterized_vertices[tri[2]]);

        face_metrics_buffer[face_id] =
            evaluate_face_metrics(p0_3d, p1_3d, p2_3d, p0_uv, p1_uv, p2_uv);

        metrics.face_avg_angle_distortion[face_id] =
            static_cast<float>(face_metrics_buffer[face_id].avg_angle_distortion);
        metrics.face_max_angle_distortion[face_id] =
            static_cast<float>(face_metrics_buffer[face_id].max_angle_distortion);
        metrics.face_flip_mask[face_id] =
            face_metrics_buffer[face_id].flipped ? 1.0f : 0.0f;

        total_area_3d += face_metrics_buffer[face_id].area_3d;
        total_area_uv += std::abs(face_metrics_buffer[face_id].signed_area_uv);
    }

    if (total_area_3d <= kMetricEps) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: total reference area is zero.");
    }
    if (total_area_uv <= kMetricEps) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: total parameterized area is zero.");
    }

    double weight_angle_distortion = 0.0;
    double weight_area_distortion = 0.0;
    double max_angle_distortion = 0.0;
    int flip_count = 0;

    for (size_t face_id = 0; face_id < triangles.size(); ++face_id) {
        const double area_3d = face_metrics_buffer[face_id].area_3d;
        const double signed_area_uv = face_metrics_buffer[face_id].signed_area_uv;
        const double area_uv = std::abs(signed_area_uv);

        double area_distortion = 0.0;
        if (area_3d > kMetricEps && area_uv > kMetricEps) {
            const double ratio = (area_uv / total_area_uv) / (area_3d / total_area_3d);
            area_distortion = std::abs(std::log(ratio));
        }
        else if (area_3d <= kMetricEps && area_uv <= kMetricEps) {
            area_distortion = 0.0;
        }
        else {
            area_distortion = std::numeric_limits<double>::infinity();
        }

        metrics.face_area_distortion[face_id] = static_cast<float>(area_distortion);
        weight_angle_distortion += area_3d * face_metrics_buffer[face_id].avg_angle_distortion;
        weight_area_distortion += area_3d * area_distortion;
        max_angle_distortion = std::max(max_angle_distortion, face_metrics_buffer[face_id].max_angle_distortion);
        
        if (face_metrics_buffer[face_id].flipped) {
            flip_count++;
        }
    }
    metrics.avg_angle_distortion = weight_angle_distortion / total_area_3d;
    metrics.avg_area_distortion = weight_area_distortion / total_area_3d;
    metrics.max_angle_distortion = max_angle_distortion;
    metrics.flip_count = flip_count;
    metrics.flip_ratio = static_cast<double>(flip_count) / static_cast<double>(triangles.size());

    return metrics;
}

inline ::Ruzino::Geometry annotate_face_metrics(
    ::Ruzino::Geometry geometry,
    const ParameterizationMetrics& metrics)
{
    auto mesh_component = geometry.get_component<::Ruzino::MeshComponent>();
    if (!mesh_component) {
        throw std::runtime_error(
            "hw5_parameterization_metrics: missing mesh component in output geometry.");
    }

    mesh_component->add_face_scalar_quantity(
        "avg_angle_distortion", metrics.face_avg_angle_distortion);
    mesh_component->add_face_scalar_quantity(
        "max_angle_distortion", metrics.face_max_angle_distortion);
    mesh_component->add_face_scalar_quantity(
        "area_distortion", metrics.face_area_distortion);
    mesh_component->add_face_scalar_quantity(
        "flip_mask", metrics.face_flip_mask);

    return geometry;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_parameterization_metrics)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference Mesh");

    b.add_output<Geometry>("Annotated Mesh");
    b.add_output<float>("Max Angle Distortion");
    b.add_output<float>("Avg Angle Distortion");
    b.add_output<float>("Avg Area Distortion");
    b.add_output<int>("Flip Count");
    b.add_output<float>("Flip Ratio");
}

NODE_EXECUTION_FUNCTION(hw5_parameterization_metrics)
{
    try {
        auto input = params.get_input<Geometry>("Input");
        auto reference_input = params.get_input<Geometry>("Reference Mesh");

        auto parameterized_mesh = input.get_component<::Ruzino::MeshComponent>();
        auto reference_mesh = reference_input.get_component<::Ruzino::MeshComponent>();
        if (!parameterized_mesh) {
            params.set_error(
                "hw5_parameterization_metrics: Input must contain a mesh.");
            return false;
        }
        if (!reference_mesh) {
            params.set_error(
                "hw5_parameterization_metrics: Reference Mesh must contain a mesh.");
            return false;
        }

        const auto metrics =
            evaluate_parameterization_metrics(*parameterized_mesh, *reference_mesh);
        auto annotated_mesh = annotate_face_metrics(input, metrics);

        params.set_output("Annotated Mesh", std::move(annotated_mesh));
        params.set_output(
            "Max Angle Distortion",
            static_cast<float>(metrics.max_angle_distortion));
        params.set_output(
            "Avg Angle Distortion",
            static_cast<float>(metrics.avg_angle_distortion));
        params.set_output(
            "Avg Area Distortion",
            static_cast<float>(metrics.avg_area_distortion));
        params.set_output("Flip Count", metrics.flip_count);
        params.set_output("Flip Ratio", static_cast<float>(metrics.flip_ratio));
        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

NODE_DECLARATION_UI(hw5_parameterization_metrics);

NODE_DEF_CLOSE_SCOPE
