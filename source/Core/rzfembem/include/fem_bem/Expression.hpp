#pragma once
#include <exprtk/exprtk.hpp>
#include <memory>
#include <string>
#include <vector>

#include "api.h"
#include "integrate.hpp"
#include "parameter_map.hpp"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"

namespace USTC_CG {
namespace fem_bem {
    using real = float;

    // Forward declarations
    class DerivativeExpression;

    class RZFEMBEM_API Expression {
       public:
        using expression_type = exprtk::expression<real>;
        using symbol_table_type = exprtk::symbol_table<real>;
        using parser_type = exprtk::parser<real>;

        // Constructors
        Expression() = default;
        explicit Expression(const std::string& expr_str);

        Expression(
            const std::string& expr_str,
            const std::vector<std::string>& variable_names);

        // Compound expression constructor
        Expression(
            const Expression& outer_expr,
            const std::vector<std::pair<const char*, Expression>>&
                variable_substitutions);

        Expression(
            const Expression& outer_expr,
            std::initializer_list<std::pair<const char*, Expression>>
                substitutions);
        Expression(const Expression& other);

        Expression& operator=(const Expression& other);

        // Move constructor and assignment
        Expression(Expression&& other) noexcept = default;
        Expression& operator=(Expression&& other) noexcept = default;

        // Virtual destructor for inheritance
        virtual ~Expression() = default;

        // Factory methods
        static Expression from_string(const std::string& expr_str);
        static Expression constant(real value);
        static Expression zero();
        static Expression one();

        // Basic properties
        const std::string& get_string() const;

        // Method to check if this is a string-based expression
        virtual bool is_string_based() const;

        real evaluate_at(const ParameterMap<real>& variable_values) const;

        // Derivative methods
        DerivativeExpression derivative(const std::string& variable_name) const;

        std::vector<DerivativeExpression> gradient(
            const std::vector<std::string>& variable_names) const;

        // Closure methods - bind specific variables to values
        void bind_variables(const ParameterMap<real>& bound_values);
        void bind_variable(const std::string& var_name, real value);

        // Check if expression has bound variables (is a closure)
        bool has_bound_variables() const;

        const ParameterMap<real>& get_bound_variables() const;

        // Access to underlying exprtk objects (for advanced use)
        const expression_type* get_compiled_expression() const;

        const symbol_table_type* get_symbol_table() const;

        // Arithmetic operations
        Expression operator+(const Expression& other) const;
        Expression operator-(const Expression& other) const;
        Expression operator*(const Expression& other) const;
        Expression operator/(const Expression& other) const;
        Expression operator*(real scalar) const;
        Expression operator-() const;

       protected:
        // Protected members for derived classes to access
        std::string expression_string_;

        bool has_bound_variable_ = false;

        // Parsed expression components
        mutable std::unique_ptr<symbol_table_type> symbol_table_;
        mutable std::unique_ptr<expression_type> compiled_expression_;
        mutable bool is_parsed_ = false;

        // Unified variable storage - serves both as temp storage for exprtk and
        // bound variables
        mutable ParameterMap<real> variables_;

        // Compound expression support
        bool is_compound_ = false;
        std::unique_ptr<Expression> outer_expression_;
        std::vector<std::pair<const char*, Expression>> substitution_map_;

        // Support for DerivativeExpression conversion
        std::function<real(const ParameterMap<real>&)> derivative_evaluator_;
        bool has_derivative_evaluator_ = false;

        void ensure_parsed() const;

       private:
        void parse_expression() const;
        void apply_recursive_substitution();

        template<typename MappingFunc>
        friend real integrate_over_simplex(
            const Expression& expr,
            const std::vector<std::string>& barycentric_names,
            const MappingFunc& mapping,
            std::size_t intervals);

        template<typename MappingFunc>
        friend real integrate_simplex_with_mapping(
            const Expression& expr,
            MappingFunc mapping,
            const std::vector<std::string>& barycentric_names,
            std::size_t intervals);
    };

    RZFEMBEM_API Expression operator*(real scalar, const Expression& expr);
    RZFEMBEM_API Expression make_expression(const std::string& expr_str);

    // Numerical derivative class that inherits from Expression
    class DerivativeExpression : public Expression {
       private:
        std::string variable_name_;

       public:
        DerivativeExpression(
            std::function<real(const ParameterMap<real>&)> func,
            const std::string& var_name)
            : Expression(),
              variable_name_(var_name)
        {
            // Set up the base class with the derivative evaluator
            this->derivative_evaluator_ = std::move(func);
            this->has_derivative_evaluator_ = true;
            this->expression_string_ =
                "";  // Derivatives don't have string representation
            this->is_compound_ = false;
        }

        // Get the variable name for this derivative
        const std::string& get_variable_name() const
        {
            return variable_name_;
        }

        // Override is_string_based to return false for derivatives
        bool is_string_based() const override
        {
            return false;
        }

        // Note: Integration methods are inherited from Expression base class
    };

    // Integration methods
    template<typename MappingExpr = std::nullptr_t>
    real integrate_over_simplex(
        const Expression& expr,
        const std::vector<std::string>& barycentric_names,
        const MappingExpr& mapping_expr = nullptr,
        std::size_t intervals = 100)
    {
        Expression final_expr = expr;
        for (const auto& barycentric_name : barycentric_names) {
            final_expr.bind_variable(barycentric_name, real(0));
        }

        // If mapping is provided, compose it with the expression
        if constexpr (!std::is_same_v<MappingExpr, std::nullptr_t>) {
            final_expr =
                compose_with_mapping(expr, mapping_expr, barycentric_names);
        }

        // Always use numerical integration for mapped expressions or complex
        // expressions
        if constexpr (!std::is_same_v<MappingExpr, std::nullptr_t>) {
            // For mapped expressions, always use numerical integration
            auto evaluator = [&final_expr](const ParameterMap<real>& values) {
                return final_expr.evaluate_at(values);
            };
            return integrate_numerical_generic(
                evaluator, barycentric_names, intervals);
        }

        // Handle compound expressions or derivatives using numerical
        // integration
        if (final_expr.has_derivative_evaluator_ ||
            (final_expr.is_compound_ && final_expr.outer_expression_) ||
            final_expr.has_bound_variables()) {
            auto evaluator = [&final_expr](const ParameterMap<real>& values) {
                return final_expr.evaluate_at(values);
            };
            return integrate_numerical_generic(
                evaluator, barycentric_names, intervals);
        }

        // For simple expressions, use existing integration methods
        final_expr.ensure_parsed();
        return integrate_simplex(
            *final_expr.compiled_expression_, barycentric_names, intervals);
    }

    // Helper function to compose expression with mapping
    template<typename MappingExpr>
    Expression compose_with_mapping(
        const Expression& expr,
        const MappingExpr& mapping_expr,
        const std::vector<std::string>& barycentric_names)
    {
        if constexpr (std::is_same_v<MappingExpr, std::nullptr_t>) {
            return expr;
        }
        else if constexpr (std::is_same_v<
                               MappingExpr,
                               ParameterMap<Expression>>) {
            // Handle Expression-based mapping
            return create_mapped_expression_with_coord_mapping(
                expr, mapping_expr, barycentric_names);
        }
        else {
            // For other mapping types, use template specialization
            return create_mapped_expression(
                expr, mapping_expr, barycentric_names);
        }
    }

    // Create coordinate mapping expressions that bind world vertex coordinates
    RZFEMBEM_API ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices);

    RZFEMBEM_API ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices);

    // Helper to create mapped expression using coordinate mapping
    RZFEMBEM_API Expression create_mapped_expression_with_coord_mapping(
        const Expression& expr,
        const ParameterMap<Expression>& coord_mapping,
        const std::vector<std::string>& barycentric_names);

    // Create mapping expression from barycentric to physical coordinates
    // (legacy)
    Expression create_mapping_expression(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices);

    Expression create_mapping_expression(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices);

    // Generic mapping expression creation template
    template<typename MappingExpr>
    Expression create_mapped_expression(
        const Expression& expr,
        const MappingExpr& mapping_expr,
        const std::vector<std::string>& barycentric_names)
    {
        // For now, return the original expression as fallback
        // This would need specific implementation based on MappingExpr type
        return expr;
    }

    // Numerical integration for any expression type
    template<typename EvaluatorFunc>
    real integrate_numerical_generic(
        EvaluatorFunc evaluator,
        const std::vector<std::string>& barycentric_names,
        std::size_t intervals)
    {
        // Create a unified evaluator that handles coordinate conversion
        auto unified_evaluator = [&](const std::vector<real>& coords) -> real {
            ParameterMap<real> values;

            // Set barycentric coordinates
            for (std::size_t i = 0;
                 i < coords.size() && i < barycentric_names.size();
                 ++i) {
                values.insert_unchecked(
                    barycentric_names[i].c_str(), coords[i]);
            }

            //// For missing barycentric coordinates, ensure they are set to 0
            // if (barycentric_names.size() == 1) {
            //     // 1D case: u1, and u2 = 1-u1 (implicitly)
            //     values.insert_unchecked(
            //         "u1", coords.size() > 0 ? coords[0] : real(0));
            // }
            // else if (barycentric_names.size() == 2) {
            //     // 2D case: u1, u2, and u3 = 1-u1-u2 (implicitly)
            //     values.insert_unchecked(
            //         "u1", coords.size() > 0 ? coords[0] : real(0));
            //     values.insert_unchecked(
            //         "u2", coords.size() > 1 ? coords[1] : real(0));
            // }
            // else if (barycentric_names.size() == 3) {
            //     // 3D case: u1, u2, u3, and u4 = 1-u1-u2-u3 (implicitly)
            //     values.insert_unchecked(
            //         "u1", coords.size() > 0 ? coords[0] : real(0));
            //     values.insert_unchecked(
            //         "u2", coords.size() > 1 ? coords[1] : real(0));
            //     values.insert_unchecked(
            //         "u3", coords.size() > 2 ? coords[2] : real(0));
            // }

            return evaluator(values);
        };

        return integrate_simplex_generic<real>(
            unified_evaluator, barycentric_names, intervals);
    }

}  // namespace fem_bem
}  // namespace USTC_CG
