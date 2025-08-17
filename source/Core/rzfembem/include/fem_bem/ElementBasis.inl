#pragma once
#include "api.h"
USTC_CG_NAMESPACE_OPEN_SCOPE
namespace fem_bem {
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
ElementBasis<ProblemDimension, Type, T>::ElementBasis()
{
    // Initialize barycentric coordinate variables
    setup_barycentric_variables();
    // Initialize expression variable manager with common variables
    setup_expression_variables();
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::add_edge_expression(
    const std::string& expr_str)
{
    if constexpr (element_dimension >= 2) {
        edge_expressions_.push_back(create_expression(expr_str));
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::set_edge_expressions(
    const std::vector<std::string>& expr_strs)
{
    if constexpr (element_dimension >= 2) {
        edge_expressions_.clear();
        edge_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            edge_expressions_.push_back(create_expression(expr_str));
        }
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
const std::vector<
    typename ElementBasis<ProblemDimension, Type, T>::expression_type>&
ElementBasis<ProblemDimension, Type, T>::get_edge_expressions() const
{
    if constexpr (element_dimension >= 2) {
        return edge_expressions_;
    }
    else {
        static const std::vector<expression_type> empty;
        return empty;
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::clear_edge_expressions()
{
    if constexpr (element_dimension >= 2) {
        edge_expressions_.clear();
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::add_face_expression(
    const std::string& expr_str)
{
    if constexpr (element_dimension >= 3) {
        face_expressions_.push_back(create_expression(expr_str));
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::set_face_expressions(
    const std::vector<std::string>& expr_strs)
{
    if constexpr (element_dimension >= 3) {
        face_expressions_.clear();
        face_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            face_expressions_.push_back(create_expression(expr_str));
        }
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
const std::vector<
    typename ElementBasis<ProblemDimension, Type, T>::expression_type>&
ElementBasis<ProblemDimension, Type, T>::get_face_expressions() const
{
    if constexpr (element_dimension >= 3) {
        return face_expressions_;
    }
    else {
        static const std::vector<expression_type> empty;
        return empty;
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::clear_face_expressions()
{
    if constexpr (element_dimension >= 3) {
        face_expressions_.clear();
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::add_volume_expression(
    const std::string& expr_str)
{
    if constexpr (element_dimension == 3) {
        volume_expressions_.push_back(create_expression(expr_str));
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::set_volume_expressions(
    const std::vector<std::string>& expr_strs)
{
    if constexpr (element_dimension == 3) {
        volume_expressions_.clear();
        volume_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            volume_expressions_.push_back(create_expression(expr_str));
        }
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
const std::vector<
    typename ElementBasis<ProblemDimension, Type, T>::expression_type>&
ElementBasis<ProblemDimension, Type, T>::get_volume_expressions() const
{
    if constexpr (element_dimension == 3) {
        return volume_expressions_;
    }
    else {
        static const std::vector<expression_type> empty;
        return empty;
    }
}
template<unsigned ProblemDimension, ElementBasisType Type, typename T>
void ElementBasis<ProblemDimension, Type, T>::clear_volume_expressions()
{
    if constexpr (element_dimension == 3) {
        volume_expressions_.clear();
    }
}
template<unsigned ProblemDim, ElementBasisType Type>
std::vector<std::string>
ElementBasisWrapper<ProblemDim, Type>::get_vertex_expression_strings() const
{
    std::vector<std::string> strings;
    const auto& expressions =
        ElementBasis<ProblemDim, Type>::get_vertex_expressions();
    strings.reserve(expressions.size());
    for (const auto& expr : expressions) {
        strings.push_back(expr.get_string());
    }
    return strings;
}
template<unsigned ProblemDim, ElementBasisType Type>
auto ElementBasisWrapper<ProblemDim, Type>::create_linear_mapping(
    const std::vector<pxr::GfVec2d>& world_vertices)
{
    return [world_vertices](auto... u_values) -> std::vector<double> {
        std::vector<double> u_vec = { static_cast<double>(u_values)... };
        std::vector<double> result(2, 0.0);  // x, y coordinates

        // For 1D elements (BEM2D), we have 2 vertices and 1 barycentric
        // coordinate u1 The second coordinate is u2 = 1 - u1
        if (u_vec.size() == 1 && world_vertices.size() >= 2) {
            double u1 = u_vec[0];
            double u2 = 1.0 - u1;
            result[0] =
                u1 * world_vertices[0][0] + u2 * world_vertices[1][0];  // x
            result[1] =
                u1 * world_vertices[0][1] + u2 * world_vertices[1][1];  // y
        }
        // For 2D elements (FEM2D), we have 3 vertices and 2 barycentric
        // coordinates u1, u2 The third coordinate is u3 = 1 - u1 - u2
        else if (u_vec.size() == 2 && world_vertices.size() >= 3) {
            double u1 = u_vec[0];
            double u2 = u_vec[1];
            double u3 = 1.0 - u1 - u2;
            result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] +
                        u3 * world_vertices[2][0];  // x
            result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] +
                        u3 * world_vertices[2][1];  // y
        }
        else {
            // Fallback for other cases
            for (size_t i = 0; i < world_vertices.size() && i < u_vec.size();
                 ++i) {
                result[0] += u_vec[i] * world_vertices[i][0];  // x
                result[1] += u_vec[i] * world_vertices[i][1];  // y
            }
        }
        return result;
    };
}
template<unsigned ProblemDim, ElementBasisType Type>
auto ElementBasisWrapper<ProblemDim, Type>::create_linear_mapping(
    const std::vector<pxr::GfVec3d>& world_vertices)
{
    return [world_vertices](auto... u_values) -> std::vector<double> {
        std::vector<double> u_vec = { static_cast<double>(u_values)... };
        std::vector<double> result(3, 0.0);  // x, y, z coordinates

        // For 1D elements (BEM2D), we have 2 vertices and 1 barycentric
        // coordinate u1
        if (u_vec.size() == 1 && world_vertices.size() >= 2) {
            double u1 = u_vec[0];
            double u2 = 1.0 - u1;
            result[0] =
                u1 * world_vertices[0][0] + u2 * world_vertices[1][0];  // x
            result[1] =
                u1 * world_vertices[0][1] + u2 * world_vertices[1][1];  // y
            result[2] =
                u1 * world_vertices[0][2] + u2 * world_vertices[1][2];  // z
        }
        // For 2D elements (BEM3D), we have 3 vertices and 2 barycentric
        // coordinates u1, u2
        else if (u_vec.size() == 2 && world_vertices.size() >= 3) {
            double u1 = u_vec[0];
            double u2 = u_vec[1];
            double u3 = 1.0 - u1 - u2;
            result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] +
                        u3 * world_vertices[2][0];  // x
            result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] +
                        u3 * world_vertices[2][1];  // y
            result[2] = u1 * world_vertices[0][2] + u2 * world_vertices[1][2] +
                        u3 * world_vertices[2][2];  // z
        }
        // For 3D elements (FEM3D), we have 4 vertices and 3 barycentric
        // coordinates u1, u2, u3
        else if (u_vec.size() == 3 && world_vertices.size() >= 4) {
            double u1 = u_vec[0];
            double u2 = u_vec[1];
            double u3 = u_vec[2];
            double u4 = 1.0 - u1 - u2 - u3;
            result[0] = u1 * world_vertices[0][0] + u2 * world_vertices[1][0] +
                        u3 * world_vertices[2][0] +
                        u4 * world_vertices[3][0];  // x
            result[1] = u1 * world_vertices[0][1] + u2 * world_vertices[1][1] +
                        u3 * world_vertices[2][1] +
                        u4 * world_vertices[3][1];  // y
            result[2] = u1 * world_vertices[0][2] + u2 * world_vertices[1][2] +
                        u3 * world_vertices[2][2] +
                        u4 * world_vertices[3][2];  // z
        }
        else {
            // Fallback for other cases
            for (size_t i = 0; i < world_vertices.size() && i < u_vec.size();
                 ++i) {
                result[0] += u_vec[i] * world_vertices[i][0];  // x
                result[1] += u_vec[i] * world_vertices[i][1];  // y
                result[2] += u_vec[i] * world_vertices[i][2];  // z
            }
        }
        return result;
    };
}
template<unsigned ProblemDim, ElementBasisType Type>
std::vector<std::string>
ElementBasisWrapper<ProblemDim, Type>::get_edge_expression_strings() const
{
    if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
        std::vector<std::string> strings;
        const auto& expressions =
            ElementBasis<ProblemDim, Type>::get_edge_expressions();
        strings.reserve(expressions.size());
        for (const auto& expr : expressions) {
            strings.push_back(expr.get_string());
        }
        return strings;
    }
    else {
        return std::vector<std::string>();
    }
}
template<unsigned ProblemDim, ElementBasisType Type>
std::vector<std::string>
ElementBasisWrapper<ProblemDim, Type>::get_face_expression_strings() const
{
    if constexpr (ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
        std::vector<std::string> strings;
        const auto& expressions =
            ElementBasis<ProblemDim, Type>::get_face_expressions();
        strings.reserve(expressions.size());
        for (const auto& expr : expressions) {
            strings.push_back(expr.get_string());
        }
        return strings;
    }
    else {
        return std::vector<std::string>();
    }
}
template<unsigned ProblemDim, ElementBasisType Type>
std::vector<std::string>
ElementBasisWrapper<ProblemDim, Type>::get_volume_expression_strings() const
{
    if constexpr (ElementBasis<ProblemDim, Type>::element_dimension == 3) {
        std::vector<std::string> strings;
        const auto& expressions =
            ElementBasis<ProblemDim, Type>::get_volume_expressions();
        strings.reserve(expressions.size());
        for (const auto& expr : expressions) {
            strings.push_back(expr.get_string());
        }
        return strings;
    }
    else {
        return std::vector<std::string>();
    }
}
}  // namespace fem_bem

USTC_CG_NAMESPACE_CLOSE_SCOPE
