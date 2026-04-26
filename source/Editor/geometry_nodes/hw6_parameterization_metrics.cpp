#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

namespace {

constexpr double kMetricEps = 1e-12;

struct FaceMetrics
{
    double reference_area = 0.0;
    double signed_uv_area = 0.0;
    double signed_sigma_1 = 0.0;
    double signed_sigma_2 = 0.0;
    double angular_distortion = 0.0;
    double area_distortion = 0.0;
    bool flipped = false;
};

struct ParameterizationMetrics
{
    double angular_distortion = 0.0;
    double area_distortion = 0.0;
    int flip_count = 0;
    double flip_ratio = 0.0;

    std::vector<float> face_angular_distortion;
    std::vector<float> face_area_distortion;
    std::vector<float> face_signed_sigma_1;
    std::vector<float> face_signed_sigma_2;
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
    return 0.5 * ((b - a).x() * (c - a).y() -
                  (b - a).y() * (c - a).x());
}

inline std::array<Eigen::Vector2d, 3> build_reference_triangle(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2)
{
    const Eigen::Vector3d e01 = p1 - p0;
    const Eigen::Vector3d e02 = p2 - p0;
    const double len01 = e01.norm();
    if (len01 <= kMetricEps) {
        throw std::runtime_error("hw6_parameterization_metrics: degenerate reference edge.");
    }

    const double x2 = e01.dot(e02) / len01;
    const double y2_sq = e02.squaredNorm() - x2 * x2;
    if (y2_sq < -1e-10) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: failed to build local reference triangle.");
    }

    const double y2 = std::sqrt(std::max(0.0, y2_sq));
    if (y2 <= kMetricEps) {
        throw std::runtime_error("hw6_parameterization_metrics: degenerate reference triangle.");
    }

    return {
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(len01, 0.0),
        Eigen::Vector2d(x2, y2),
    };
}

inline void validate_metric_inputs(
    const ::Ruzino::MeshComponent& parameterized_mesh,
    const ::Ruzino::MeshComponent& reference_mesh)
{
    if (parameterized_mesh.get_vertices().size() !=
        reference_mesh.get_vertices().size()) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: vertex count mismatch.");
    }

    if (parameterized_mesh.get_face_vertex_counts() !=
            reference_mesh.get_face_vertex_counts() ||
        parameterized_mesh.get_face_vertex_indices() !=
            reference_mesh.get_face_vertex_indices()) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: face topology mismatch.");
    }

    for (int count : parameterized_mesh.get_face_vertex_counts()) {
        if (count != 3) {
            throw std::runtime_error(
                "hw6_parameterization_metrics: only triangle meshes are supported.");
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
                "hw6_parameterization_metrics: non-triangular face encountered.");
        }

        triangles.push_back({
            face_indices[cursor],
            face_indices[cursor + 1],
            face_indices[cursor + 2],
        });
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
    face_metrics.reference_area = triangle_area_3d(p0_3d, p1_3d, p2_3d);
    if (face_metrics.reference_area <= kMetricEps) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: zero-area reference triangle.");
    }

    const auto reference_triangle =
        build_reference_triangle(p0_3d, p1_3d, p2_3d);

    Eigen::Matrix2d reference_edges;
    reference_edges.col(0) = reference_triangle[1] - reference_triangle[0];
    reference_edges.col(1) = reference_triangle[2] - reference_triangle[0];
    if (std::abs(reference_edges.determinant()) <= kMetricEps) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: degenerate reference edge matrix.");
    }

    Eigen::Matrix2d uv_edges;
    uv_edges.col(0) = p1_uv - p0_uv;
    uv_edges.col(1) = p2_uv - p0_uv;

    const Eigen::Matrix2d jacobian = uv_edges * reference_edges.inverse();
    const double jacobian_det = jacobian.determinant();

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(jacobian);
    const auto singular_values = svd.singularValues();

    face_metrics.signed_sigma_1 = singular_values[0];
    face_metrics.signed_sigma_2 =
        jacobian_det < 0.0 ? -singular_values[1] : singular_values[1];
    face_metrics.signed_uv_area =
        signed_triangle_area_2d(p0_uv, p1_uv, p2_uv);
    face_metrics.flipped = jacobian_det <= 0.0;

    if (std::abs(face_metrics.signed_sigma_1) <= kMetricEps ||
        std::abs(face_metrics.signed_sigma_2) <= kMetricEps) {
        face_metrics.angular_distortion =
            std::numeric_limits<double>::infinity();
    }
    else {
        face_metrics.angular_distortion =
            face_metrics.signed_sigma_1 / face_metrics.signed_sigma_2 +
            face_metrics.signed_sigma_2 / face_metrics.signed_sigma_1;
    }

    const double signed_area_scale =
        face_metrics.signed_sigma_1 * face_metrics.signed_sigma_2;
    if (std::abs(signed_area_scale) <= kMetricEps) {
        face_metrics.area_distortion =
            std::numeric_limits<double>::infinity();
    }
    else {
        face_metrics.area_distortion =
            signed_area_scale + 1.0 / signed_area_scale;
    }

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
            "hw6_parameterization_metrics: empty triangle mesh.");
    }

    const auto& reference_vertices = reference_mesh.get_vertices();
    const auto& parameterized_vertices = parameterized_mesh.get_vertices();

    metrics.face_angular_distortion.assign(triangles.size(), 0.0f);
    metrics.face_area_distortion.assign(triangles.size(), 0.0f);
    metrics.face_signed_sigma_1.assign(triangles.size(), 0.0f);
    metrics.face_signed_sigma_2.assign(triangles.size(), 0.0f);
    metrics.face_flip_mask.assign(triangles.size(), 0.0f);

    std::vector<FaceMetrics> face_metrics_buffer(triangles.size());
    double total_reference_area = 0.0;

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

        total_reference_area += face_metrics_buffer[face_id].reference_area;
    }

    if (total_reference_area <= kMetricEps) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: total reference area is zero.");
    }

    int flip_count = 0;
    for (size_t face_id = 0; face_id < triangles.size(); ++face_id) {
        const auto& face_metrics = face_metrics_buffer[face_id];
        const double rho = face_metrics.reference_area / total_reference_area;

        metrics.angular_distortion +=
            rho * face_metrics.angular_distortion;
        metrics.area_distortion +=
            rho * face_metrics.area_distortion;

        metrics.face_angular_distortion[face_id] =
            static_cast<float>(face_metrics.angular_distortion);
        metrics.face_area_distortion[face_id] =
            static_cast<float>(face_metrics.area_distortion);
        metrics.face_signed_sigma_1[face_id] =
            static_cast<float>(face_metrics.signed_sigma_1);
        metrics.face_signed_sigma_2[face_id] =
            static_cast<float>(face_metrics.signed_sigma_2);
        metrics.face_flip_mask[face_id] =
            face_metrics.flipped ? 1.0f : 0.0f;

        if (face_metrics.flipped) {
            flip_count++;
        }
    }

    metrics.flip_count = flip_count;
    metrics.flip_ratio =
        static_cast<double>(flip_count) / static_cast<double>(triangles.size());
    return metrics;
}

inline ::Ruzino::Geometry annotate_face_metrics(
    ::Ruzino::Geometry geometry,
    const ParameterizationMetrics& metrics)
{
    auto mesh_component = geometry.get_component<::Ruzino::MeshComponent>();
    if (!mesh_component) {
        throw std::runtime_error(
            "hw6_parameterization_metrics: missing mesh component in output geometry.");
    }

    mesh_component->add_face_scalar_quantity(
        "angular_distortion", metrics.face_angular_distortion);
    mesh_component->add_face_scalar_quantity(
        "area_distortion", metrics.face_area_distortion);
    mesh_component->add_face_scalar_quantity(
        "signed_sigma_1", metrics.face_signed_sigma_1);
    mesh_component->add_face_scalar_quantity(
        "signed_sigma_2", metrics.face_signed_sigma_2);
    mesh_component->add_face_scalar_quantity(
        "flip_mask", metrics.face_flip_mask);

    return geometry;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw6_parameterization_metrics)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference Mesh");

    b.add_output<Geometry>("Annotated Mesh");
    b.add_output<float>("Angular Distortion");
    b.add_output<float>("Area Distortion");
    b.add_output<int>("Flip Count");
    b.add_output<float>("Flip Ratio");
}

NODE_EXECUTION_FUNCTION(hw6_parameterization_metrics)
{
    try {
        auto input = params.get_input<Geometry>("Input");
        auto reference_input = params.get_input<Geometry>("Reference Mesh");

        auto parameterized_mesh =
            input.get_component<::Ruzino::MeshComponent>();
        auto reference_mesh =
            reference_input.get_component<::Ruzino::MeshComponent>();
        if (!parameterized_mesh) {
            params.set_error(
                "hw6_parameterization_metrics: Input must contain a mesh.");
            return false;
        }
        if (!reference_mesh) {
            params.set_error(
                "hw6_parameterization_metrics: Reference Mesh must contain a mesh.");
            return false;
        }

        const auto metrics =
            evaluate_parameterization_metrics(*parameterized_mesh, *reference_mesh);
        auto annotated_mesh = annotate_face_metrics(input, metrics);

        params.set_output("Annotated Mesh", std::move(annotated_mesh));
        params.set_output(
            "Angular Distortion",
            static_cast<float>(metrics.angular_distortion));
        params.set_output(
            "Area Distortion",
            static_cast<float>(metrics.area_distortion));
        params.set_output("Flip Count", metrics.flip_count);
        params.set_output("Flip Ratio", static_cast<float>(metrics.flip_ratio));
        return true;
    }
    catch (const std::exception& e) {
        params.set_error(e.what());
        return false;
    }
}

NODE_DECLARATION_UI(hw6_parameterization_metrics);

NODE_DEF_CLOSE_SCOPE
