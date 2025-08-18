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
        static Expression from_string(const std::string& expr_str)
        {
            return Expression(expr_str);
        }

        static Expression constant(real value)
        {
            return Expression(std::to_string(value));
        }

        static Expression zero()
        {
            return Expression("0");
        }

        static Expression one()
        {
            return Expression("1");
        }

        // Basic properties
        const std::string& get_string() const
        {
            return expression_string_;
        }

        // Method to check if this is a string-based expression
        virtual bool is_string_based() const
        {
            return true;  // Regular expressions are always string-based
        }

        real evaluate_at(const ParameterMap<real>& variable_values) const;
        // Arithmetic operations
        Expression operator+(const Expression& other) const;

        Expression operator-(const Expression& other) const;

        Expression operator*(const Expression& other) const;

        Expression operator/(const Expression& other) const;

        Expression operator*(real scalar) const;

        Expression operator-() const;

        // Derivative methods
        DerivativeExpression derivative(const std::string& variable_name) const;

        std::vector<DerivativeExpression> gradient(
            const std::vector<std::string>& variable_names) const;

        // Access to underlying exprtk objects (for advanced use)
        const expression_type* get_compiled_expression() const
        {
            ensure_parsed();
            return compiled_expression_.get();
        }

        const symbol_table_type* get_symbol_table() const
        {
            ensure_parsed();
            return symbol_table_.get();
        }

       protected:
        // Protected members for derived classes to access
        std::string expression_string_;
        mutable std::vector<std::string> variable_names_;

        // Parsed expression components
        mutable std::unique_ptr<symbol_table_type> symbol_table_;
        mutable std::unique_ptr<expression_type> compiled_expression_;
        mutable bool is_parsed_ = false;

        // Storage for variables
        mutable ParameterMap<real> temp_variables_;

        // Compound expression support
        bool is_compound_ = false;
        std::unique_ptr<Expression> outer_expression_;
        std::vector<std::pair<const char*, Expression>> substitution_map_;

        // Support for DerivativeExpression conversion
        std::function<real(const ParameterMap<real>&)> derivative_evaluator_;

        void ensure_parsed() const
        {
            if (!is_parsed_ || !compiled_expression_) {
                parse_expression();
            }
        }

       private:
        void parse_expression() const;

        template<typename MappingFunc>
        friend real integrate_over_simplex(
            const Expression& expr,
            const std::vector<std::string>& barycentric_names,
            MappingFunc mapping,
            std::size_t intervals);

        template<typename MappingFunc>
        friend real integrate_product_with(
            const Expression& expr,
            const Expression& other,
            const std::vector<std::string>& barycentric_names,
            MappingFunc mapping,
            std::size_t intervals);

        template<typename EvaluatorFunc, typename MappingFunc>
        friend real integrate_with_mapping(
            const Expression& expr,
            EvaluatorFunc evaluator,
            MappingFunc mapping,
            const std::vector<std::string>& barycentric_names,
            std::size_t intervals);

        template<typename MappingFunc>
        friend real integrate_simplex_with_mapping(
            const Expression& expr,
            MappingFunc mapping,
            const std::vector<std::string>& barycentric_names,
            std::size_t intervals);
    };


    Expression operator*(real scalar, const Expression& expr);
    Expression make_expression(const std::string& expr_str);

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
    template<typename MappingFunc = std::nullptr_t>
    real integrate_over_simplex(
        const Expression& expr,
        const std::vector<std::string>& barycentric_names,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100)
    {
        // Handle compound expressions using numerical integration
        if (expr.derivative_evaluator_ ||
            (expr.is_compound_ && expr.outer_expression_)) {
            auto evaluator = [&expr](const ParameterMap<real>& values) {
                return expr.evaluate_at(values);
            };
            return integrate_numerical_with_mapping(
                evaluator, mapping, barycentric_names, intervals);
        }

        // For simple expressions, use existing integration methods
        expr.ensure_parsed();
        if constexpr (std::is_same_v<MappingFunc, std::nullptr_t>) {
            return integrate_simplex(
                *expr.compiled_expression_, barycentric_names, intervals);
        }
        else {
            return integrate_simplex_with_mapping(
                *expr.compiled_expression_,
                mapping,
                barycentric_names,
                intervals);
        }
    }

    template<typename MappingFunc = std::nullptr_t>
    real integrate_product_with(
        const Expression& expr,
        const Expression& other,
        const std::vector<std::string>& barycentric_names,
        MappingFunc mapping = nullptr,
        std::size_t intervals = 100)
    {
        // If either expression is compound or derivative-based, use
        // numerical integration
        if (expr.derivative_evaluator_ ||
            (expr.is_compound_ && expr.outer_expression_) ||
            (other.derivative_evaluator_ ||
             (other.is_compound_ && other.outer_expression_))) {
            auto product_evaluator =
                [&expr, &other](const ParameterMap<real>& values) {
                    return expr.evaluate_at(values) * other.evaluate_at(values);
                };
            return integrate_numerical_with_mapping(
                product_evaluator, mapping, barycentric_names, intervals);
        }

        // For simple expressions, use existing integration methods
        expr.ensure_parsed();
        other.ensure_parsed();

        if constexpr (std::is_same_v<MappingFunc, std::nullptr_t>) {
            return integrate_product_simplex(
                *expr.compiled_expression_,
                *other.compiled_expression_,
                barycentric_names,
                intervals);
        }
        else {
            return integrate_product_simplex_with_mapping(
                *expr.compiled_expression_,
                *other.compiled_expression_,
                mapping,
                barycentric_names,
                intervals);
        }
    }

    // Numerical integration for compound expressions
    template<typename EvaluatorFunc, typename MappingFunc = std::nullptr_t>
    real integrate_numerical_with_mapping(
        EvaluatorFunc evaluator,
        MappingFunc mapping,
        const std::vector<std::string>& barycentric_names,
        std::size_t intervals)
    {
        // Create a unified evaluator that handles both coordinate types
        auto unified_evaluator = [&](const std::vector<real>& coords) -> real {
            // Convert vector coords to map format
            ParameterMap<real> values;

            // Set barycentric coordinates
            for (std::size_t i = 0;
                 i < coords.size() && i < barycentric_names.size();
                 ++i) {
                values.insert_or_assign(
                    barycentric_names[i].c_str(), coords[i]);
            }

            // Apply mapping if provided
            if constexpr (!std::is_same_v<MappingFunc, std::nullptr_t>) {
                if (coords.size() == 1) {
                    auto mapped_coords = mapping(coords[0]);
                    if (mapped_coords.size() > 0)
                        values.insert_or_assign("x", mapped_coords[0]);
                    if (mapped_coords.size() > 1)
                        values.insert_or_assign("y", mapped_coords[1]);
                    if (mapped_coords.size() > 2)
                        values.insert_or_assign("z", mapped_coords[2]);
                }
                else if (coords.size() == 2) {
                    auto mapped_coords = mapping(coords[0], coords[1]);
                    if (mapped_coords.size() > 0)
                        values.insert_or_assign("x", mapped_coords[0]);
                    if (mapped_coords.size() > 1)
                        values.insert_or_assign("y", mapped_coords[1]);
                    if (mapped_coords.size() > 2)
                        values.insert_or_assign("z", mapped_coords[2]);
                }
                else if (coords.size() == 3) {
                    auto mapped_coords =
                        mapping(coords[0], coords[1], coords[2]);
                    if (mapped_coords.size() > 0)
                        values.insert_or_assign("x", mapped_coords[0]);
                    if (mapped_coords.size() > 1)
                        values.insert_or_assign("y", mapped_coords[1]);
                    if (mapped_coords.size() > 2)
                        values.insert_or_assign("z", mapped_coords[2]);
                }
            }

            return evaluator(values);
        };

        // Use the generic integration framework
        return integrate_simplex_generic<real>(
            unified_evaluator, barycentric_names, intervals);
    }
    template<typename MappingFunc>
    real integrate_simplex_with_mapping(
        const Expression& expr,
        MappingFunc mapping,
        const std::vector<std::string>& barycentric_names,
        std::size_t intervals)
    {
        // For compound expressions, use numerical integration
        if (expr.is_compound_ && expr.outer_expression_) {
            auto compound_evaluator =
                [&expr](const ParameterMap<real>& values) {
                    return expr.evaluate_at(values);
                };
            return integrate_numerical_with_mapping(
                compound_evaluator, mapping, barycentric_names, intervals);
        }

        // For simple expressions, use the mapping-aware integration from
        // integrate.hpp
        Expression unit_expr("1", barycentric_names);
        unit_expr.ensure_parsed();

        return integrate_product_simplex_with_mapping(
            *unit_expr.compiled_expression_,
            expr,
            mapping,
            barycentric_names,
            intervals);
    }

}  // namespace fem_bem
}  // namespace USTC_CG
