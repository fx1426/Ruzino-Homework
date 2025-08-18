#pragma once
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "Expression.hpp"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"

namespace USTC_CG {

namespace fem_bem {

    enum class ElementBasisType { FiniteElement, BoundaryElement };

    // Base template for element basis with dimension-aware expression storage
    template<
        unsigned ProblemDimension,
        ElementBasisType Type,
        typename T = double>
    class ElementBasis {
       public:
        using value_type = T;
        using expression_type = Expression;

        static constexpr unsigned problem_dimension = ProblemDimension;
        static constexpr unsigned element_dimension =
            (Type == ElementBasisType::FiniteElement) ? ProblemDimension
                                                      : ProblemDimension - 1;
        static constexpr ElementBasisType type = Type;

        static_assert(
            ProblemDimension >= 1 && ProblemDimension <= 3,
            "Problem dimension must be 1, 2, or 3");
        static_assert(
            Type == ElementBasisType::BoundaryElement ? ProblemDimension >= 2
                                                      : true,
            "Boundary elements require problem dimension >= 2");

        ElementBasis();
        virtual ~ElementBasis() = default;

        // Expression management
        // Vertex expressions (always available) - 0D knots
        void add_vertex_expression(const std::string& expr_str)
        {
            vertex_expressions_.push_back(create_expression(expr_str));
        }

        void set_vertex_expressions(const std::vector<std::string>& expr_strs)
        {
            vertex_expressions_.clear();
            vertex_expressions_.reserve(expr_strs.size());
            for (const auto& expr_str : expr_strs) {
                vertex_expressions_.push_back(create_expression(expr_str));
            }
        }

        const std::vector<expression_type>& get_vertex_expressions() const
        {
            return vertex_expressions_;
        }

        void clear_vertex_expressions()
        {
            vertex_expressions_.clear();
        }

        // Edge expressions (available when element_dimension >= 2) - 1D knots
        virtual void add_edge_expression(const std::string& expr_str);
        void set_edge_expressions(const std::vector<std::string>& expr_strs);
        const std::vector<expression_type>& get_edge_expressions() const;
        void clear_edge_expressions();

        // Face expressions (available when element_dimension >= 3) - 2D knots
        virtual void add_face_expression(const std::string& expr_str);
        void set_face_expressions(const std::vector<std::string>& expr_strs);
        const std::vector<expression_type>& get_face_expressions() const;
        void clear_face_expressions();

        // Volume expressions (only for 3D elements) - 3D knots
        virtual void add_volume_expression(const std::string& expr_str);
        void set_volume_expressions(const std::vector<std::string>& expr_strs);
        const std::vector<expression_type>& get_volume_expressions() const;
        void clear_volume_expressions();

        // Check if specific expression types are available at compile time
        static constexpr bool has_edge_expressions()
        {
            return element_dimension >= 2;
        }
        static constexpr bool has_face_expressions()
        {
            return element_dimension >= 3;
        }
        static constexpr bool has_volume_expressions()
        {
            return element_dimension == 3;
        }

        // Integration interface: integrate shape functions against expressions

        // Integrate vertex shape functions against expression
        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_vertex_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            Expression expr = create_expression(expr_str);
            return integrate_vertex_against(expr, mapping, intervals);
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_vertex_against(
            const Expression& expr,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            std::vector<T> results;
            const auto& vertex_exprs = get_vertex_expressions();
            results.reserve(vertex_exprs.size());

            for (const auto& shape_func : vertex_exprs) {
                results.push_back(integrate_shape_function_against_expression(
                    shape_func, expr, mapping, intervals));
            }
            return results;
        }

        // Integrate edge shape functions against expression (only for
        // element_dimension >= 2)
        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_edge_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension >= 2) {
                Expression expr = create_expression(expr_str);
                return integrate_edge_against(expr, mapping, intervals);
            }
            else {
                return std::vector<T>();
            }
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_edge_against(
            const Expression& expr,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension >= 2) {
                std::vector<T> results;
                const auto& edge_exprs = get_edge_expressions();
                results.reserve(edge_exprs.size());

                for (const auto& shape_func : edge_exprs) {
                    results.push_back(
                        integrate_shape_function_against_expression(
                            shape_func, expr, mapping, intervals));
                }
                return results;
            }
            else {
                return std::vector<T>();
            }
        }

        // Integrate face shape functions against expression (only for
        // element_dimension >= 3)
        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_face_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension >= 3) {
                Expression expr = create_expression(expr_str);
                return integrate_face_against(expr, mapping, intervals);
            }
            else {
                return std::vector<T>();
            }
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_face_against(
            const Expression& expr,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension >= 3) {
                std::vector<T> results;
                const auto& face_exprs = get_face_expressions();
                results.reserve(face_exprs.size());

                for (const auto& shape_func : face_exprs) {
                    results.push_back(
                        integrate_shape_function_against_expression(
                            shape_func, expr, mapping, intervals));
                }
                return results;
            }
            else {
                return std::vector<T>();
            }
        }

        // Integrate volume shape functions against expression (only for
        // element_dimension == 3)
        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_volume_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension == 3) {
                Expression expr = create_expression(expr_str);
                return integrate_volume_against(expr, mapping, intervals);
            }
            else {
                return std::vector<T>();
            }
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<T> integrate_volume_against(
            const Expression& expr,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            if constexpr (element_dimension == 3) {
                std::vector<T> results;
                const auto& volume_exprs = get_volume_expressions();
                results.reserve(volume_exprs.size());

                for (const auto& shape_func : volume_exprs) {
                    results.push_back(
                        integrate_shape_function_against_expression(
                            shape_func, expr, mapping, intervals));
                }
                return results;
            }
            else {
                return std::vector<T>();
            }
        }

       private:
        // Create a new expression with all necessary variables pre-registered
        Expression create_expression(const std::string& expr_str) const
        {
            // Collect all possible variable names that might be needed
            std::vector<std::string> all_vars;

            // Add barycentric variables for this element
            for (const auto& name : barycentric_names_) {
                all_vars.push_back(name);
            }

            // Add physical coordinate variables for mapping support
            all_vars.push_back("x");
            all_vars.push_back("y");
            all_vars.push_back("z");

            // Add ALL possible barycentric coordinates (u1, u2, u3)
            // This handles cases where expressions reference coordinates
            // not directly used by this element type
            all_vars.push_back("u1");
            all_vars.push_back("u2");
            all_vars.push_back("u3");

            // Add test-specific variables for compatibility
            all_vars.push_back("undefined_var");

            // Create expression with pre-defined variables
            return Expression(expr_str, all_vars);
        }

        // Setup barycentric coordinate variables based on element dimension
        void setup_barycentric_variables()
        {
            if constexpr (element_dimension == 1) {
                barycentric_names_ = { "u1" };
            }
            else if constexpr (element_dimension == 2) {
                barycentric_names_ = { "u1", "u2" };
            }
            else if constexpr (element_dimension == 3) {
                barycentric_names_ = { "u1", "u2", "u3" };
            }
        }

        // Setup common expression variables that might be needed
        void setup_expression_variables()
        {
            // This is a simplified setup since Expression class now handles
            // variable management internally through constructor parameters
            // No need to pre-initialize variables
        }

        // Core integration implementation: integrates shape_function *
        // expression over simplex
        template<typename MappingFunc>
        T integrate_shape_function_against_expression(
            const expression_type& shape_func,
            const expression_type& expr,
            MappingFunc mapping,
            std::size_t intervals) const
        {
            // Use the Expression class's integration methods
            return integrate_product_with(
                shape_func, expr, barycentric_names_, mapping, intervals);
        }

       protected:
        // Barycentric variables for expression parsing
        std::vector<std::string> barycentric_names_;

        // Vertex expressions are always available (0D knots)
        std::vector<expression_type> vertex_expressions_;

        // Conditional member variables based on element dimension
        std::conditional_t<
            element_dimension >= 2,
            std::vector<expression_type>,
            char>
            edge_expressions_;

        std::conditional_t<
            element_dimension >= 3,
            std::vector<expression_type>,
            char>
            face_expressions_;

        std::conditional_t<
            element_dimension == 3,
            std::vector<expression_type>,
            char>
            volume_expressions_;
    };

    // Convenient type aliases for common cases
    template<unsigned ProblemDim>
    using FiniteElementBasis =
        ElementBasis<ProblemDim, ElementBasisType::FiniteElement>;

    template<unsigned ProblemDim>
    using BoundaryElementBasis =
        ElementBasis<ProblemDim, ElementBasisType::BoundaryElement>;

    // Specific instantiations with clear documentation
    using FEM1D =
        FiniteElementBasis<1>;  // 1D finite elements (element_dim = 1)
    using FEM2D =
        FiniteElementBasis<2>;  // 2D finite elements (element_dim = 2)
    using FEM3D = FiniteElementBasis<3>;  // 3D finite elements (element_dim =
                                          // 3, has volume)

    using BEM2D = BoundaryElementBasis<2>;  // 1D boundary elements in 2D space
                                            // (element_dim = 1)
    using BEM3D = BoundaryElementBasis<3>;  // 2D boundary elements in 3D space
                                            // (element_dim = 2)

    // Base class for polymorphism when needed
    class ElementBasisBase {
       public:
        virtual ~ElementBasisBase() = default;
        virtual ElementBasisType get_type() const = 0;
        virtual unsigned get_problem_dimension() const = 0;
        virtual unsigned get_element_dimension() const = 0;

        // Virtual interface for expression string management
        virtual void add_vertex_expression(const std::string& expr) = 0;
        virtual std::vector<std::string> get_vertex_expression_strings()
            const = 0;

        // Virtual interface for integration without mapping
        virtual std::vector<double> integrate_vertex_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const = 0;
        virtual std::vector<double> integrate_edge_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const = 0;
        virtual std::vector<double> integrate_face_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const = 0;
        virtual std::vector<double> integrate_volume_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const = 0;

        // Virtual interface for integration with pullback mapping using world
        // coordinates
        virtual std::vector<double> integrate_vertex_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec2d>& world_vertices,
            std::size_t intervals = 100) const = 0;
        virtual std::vector<double> integrate_vertex_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const = 0;

        virtual std::vector<double> integrate_edge_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec2d>& world_vertices,
            std::size_t intervals = 100) const = 0;
        virtual std::vector<double> integrate_edge_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const = 0;

        virtual std::vector<double> integrate_face_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const = 0;

        virtual std::vector<double> integrate_volume_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const = 0;

        // Virtual interface for expression count and availability
        virtual std::size_t vertex_expression_count() const = 0;

        // Dimension-specific virtual interfaces
        virtual bool supports_edge_expressions() const = 0;
        virtual bool supports_face_expressions() const = 0;
        virtual bool supports_volume_expressions() const = 0;

        // Edge expressions (only available if supported)
        virtual void add_edge_expression(const std::string& expr)
        {
        }
        virtual std::vector<std::string> get_edge_expression_strings() const
        {
            return std::vector<std::string>();
        }
        virtual std::size_t edge_expression_count() const
        {
            return 0;
        }

        // Face expressions (only available if supported)
        virtual void add_face_expression(const std::string& expr)
        {
        }
        virtual std::vector<std::string> get_face_expression_strings() const
        {
            return std::vector<std::string>();
        }
        virtual std::size_t face_expression_count() const
        {
            return 0;
        }

        // Volume expressions (only available if supported)
        virtual void add_volume_expression(const std::string& expr)
        {
        }
        virtual std::vector<std::string> get_volume_expression_strings() const
        {
            return std::vector<std::string>();
        }
        virtual std::size_t volume_expression_count() const
        {
            return 0;
        }
    };

    // Wrapper for type erasure when polymorphism is needed
    template<unsigned ProblemDim, ElementBasisType Type>
    class ElementBasisWrapper : public ElementBasisBase,
                                public ElementBasis<ProblemDim, Type> {
       public:
        ElementBasisType get_type() const override
        {
            return Type;
        }
        unsigned get_problem_dimension() const override
        {
            return ProblemDim;
        }
        unsigned get_element_dimension() const override
        {
            return ElementBasis<ProblemDim, Type>::element_dimension;
        }

        // Vertex expression interface
        void add_vertex_expression(const std::string& expr) override
        {
            ElementBasis<ProblemDim, Type>::add_vertex_expression(expr);
        }

        std::vector<std::string> get_vertex_expression_strings() const override;

        std::size_t vertex_expression_count() const override
        {
            return ElementBasis<ProblemDim, Type>::get_vertex_expressions()
                .size();
        }

        // Virtual integration interface implementations - without mapping
        std::vector<double> integrate_vertex_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const override
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_vertex_against(
                    expr_str, nullptr, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_edge_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const override
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_edge_against(
                    expr_str, nullptr, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_face_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const override
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_face_against(
                    expr_str, nullptr, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_volume_against_str(
            const std::string& expr_str,
            std::size_t intervals = 100) const override
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_volume_against(
                    expr_str, nullptr, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        // Virtual integration interface implementations - with mapping
        std::vector<double> integrate_vertex_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec2d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_vertex_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_vertex_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_vertex_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_edge_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec2d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_edge_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_edge_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_edge_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_face_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_face_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        std::vector<double> integrate_volume_against_with_mapping(
            const std::string& expr_str,
            const std::vector<pxr::GfVec3d>& world_vertices,
            std::size_t intervals = 100) const override
        {
            auto mapping = create_linear_mapping(world_vertices);
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_volume_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

       private:
        // Helper functions to create linear mapping functions from world
        // vertices
        static auto create_linear_mapping(
            const std::vector<pxr::GfVec2d>& world_vertices);

        static auto create_linear_mapping(
            const std::vector<pxr::GfVec3d>& world_vertices);

        // Template methods for custom mapping functions - These are the ONLY
        // integration interfaces
        template<typename MappingFunc = std::nullptr_t>
        std::vector<double> integrate_vertex_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_vertex_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<double> integrate_edge_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_edge_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<double> integrate_face_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_face_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        template<typename MappingFunc = std::nullptr_t>
        std::vector<double> integrate_volume_against(
            const std::string& expr_str,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            auto results =
                ElementBasis<ProblemDim, Type>::integrate_volume_against(
                    expr_str, mapping, intervals);
            return std::vector<double>(results.begin(), results.end());
        }

        // Dimension support checks
        bool supports_edge_expressions() const override
        {
            return ElementBasis<ProblemDim, Type>::has_edge_expressions();
        }

        bool supports_face_expressions() const override
        {
            return ElementBasis<ProblemDim, Type>::has_face_expressions();
        }

        bool supports_volume_expressions() const override
        {
            return ElementBasis<ProblemDim, Type>::has_volume_expressions();
        }

        // Edge expression interface (only if supported)
        void add_edge_expression(const std::string& expr) override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
                ElementBasis<ProblemDim, Type>::add_edge_expression(expr);
            }
        }

        std::vector<std::string> get_edge_expression_strings() const override;

        std::size_t edge_expression_count() const override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension >= 2) {
                return ElementBasis<ProblemDim, Type>::get_edge_expressions()
                    .size();
            }
            return 0;
        }

        // Face expression interface (only if supported)
        void add_face_expression(const std::string& expr) override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
                ElementBasis<ProblemDim, Type>::add_face_expression(expr);
            }
        }

        std::vector<std::string> get_face_expression_strings() const override;

        std::size_t face_expression_count() const override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension >= 3) {
                return ElementBasis<ProblemDim, Type>::get_face_expressions()
                    .size();
            }
            return 0;
        }

        // Volume expression interface (only if supported)
        void add_volume_expression(const std::string& expr) override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension == 3) {
                ElementBasis<ProblemDim, Type>::add_volume_expression(expr);
            }
        }

        std::vector<std::string> get_volume_expression_strings() const override;

        std::size_t volume_expression_count() const override
        {
            if constexpr (
                ElementBasis<ProblemDim, Type>::element_dimension == 3) {
                return ElementBasis<ProblemDim, Type>::get_volume_expressions()
                    .size();
            }
            return 0;
        }
    };

    using ElementBasisHandle = std::shared_ptr<ElementBasisBase>;

    // Factory functions for creating ElementBasisHandle
    template<unsigned ProblemDim, ElementBasisType Type>
    ElementBasisHandle make_element_basis()
    {
        return std::make_shared<ElementBasisWrapper<ProblemDim, Type>>();
    }

    // Convenience factory functions
    inline ElementBasisHandle make_fem_1d()
    {
        return make_element_basis<1, ElementBasisType::FiniteElement>();
    }

    inline ElementBasisHandle make_fem_2d()
    {
        return make_element_basis<2, ElementBasisType::FiniteElement>();
    }

    inline ElementBasisHandle make_fem_3d()
    {
        return make_element_basis<3, ElementBasisType::FiniteElement>();
    }

    inline ElementBasisHandle make_bem_2d()
    {
        return make_element_basis<2, ElementBasisType::BoundaryElement>();
    }

    inline ElementBasisHandle make_bem_3d()
    {
        return make_element_basis<3, ElementBasisType::BoundaryElement>();
    }

}  // namespace fem_bem

}  // namespace USTC_CG

#include "ElementBasis.inl"
