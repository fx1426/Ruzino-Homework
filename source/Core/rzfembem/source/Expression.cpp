#include <cstring>
#include <fem_bem/Expression.hpp>

namespace USTC_CG {
namespace fem_bem {
    static exprtk::parser<real> parser;

    // Create a derivative function for compound expressions using chain rule
    std::function<real(const ParameterMap<real>&)>
    create_compound_derivative_function(
        const std::function<real(const ParameterMap<real>&)>&
            compound_evaluator,
        const std::string& variable_name,
        const real& h)
    {
        return [compound_evaluator, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            ParameterMap<real> values_ = values;
            real* var_value = values_.find(variable_name.c_str());
            if (!var_value) {
                return real(0);  // Variable not found
            }

            const real x_init = *var_value;

            // Create modified value maps for derivative computation
            // ParameterMap<real> values_ = values;

            // Use 2-point central difference
            *var_value = x_init + h;
            const real y_plus = compound_evaluator(values_);

            *var_value = x_init - h;
            const real y_minus = compound_evaluator(values_);

            real derivative = (y_plus - y_minus) / (real(2) * h);

            // Check for numerical issues and use adaptive step if needed
            if (std::abs(derivative) > real(1e6)) {
                real smaller_h = h * real(0.1);
                *var_value = x_init + smaller_h;
                const real y_plus_small = compound_evaluator(values_);

                *var_value = x_init - smaller_h;
                const real y_minus_small = compound_evaluator(values_);

                derivative =
                    (y_plus_small - y_minus_small) / (real(2) * smaller_h);
            }

            return derivative;
        };
    }

    Expression::Expression(const std::string& expr_str)
        : expression_string_(expr_str),
          is_compound_(false)
    {
    }
    Expression::Expression(
        const std::string& expr_str,
        const std::vector<std::string>& variable_names)
        : expression_string_(expr_str),
          is_compound_(false)
    {
        for (const auto& var_name : variable_names) {
            variables_.insert_or_assign(var_name.c_str(), real(0));
        }
    }

    Expression::Expression(
        const Expression& outer_expr,
        const std::vector<std::pair<const char*, Expression>>&
            variable_substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(variable_substitutions),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string() +
                             " with "
                             "substitutions: ";
        for (const auto& pair : substitution_map_) {
            expression_string_ += std::string(pair.first) + " = " +
                                  pair.second.get_string() + ", ";
        }

        // Apply recursive substitution logic
        // apply_recursive_substitution();
    }
    Expression::Expression(
        const Expression& outer_expr,
        std::initializer_list<std::pair<const char*, Expression>> substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(substitutions.begin(), substitutions.end()),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string() +
                             " with "
                             "substitutions: ";
        for (const auto& pair : substitution_map_) {
            expression_string_ += std::string(pair.first) + " = " +
                                  pair.second.get_string() + ", ";
        }

        // Apply recursive substitution logic
        // apply_recursive_substitution();
    }

    Expression::Expression(const Expression& other)
        : expression_string_(other.expression_string_),
          is_parsed_(false),  // Force re-parsing with new symbol table
          is_compound_(other.is_compound_),
          outer_expression_(
              other.outer_expression_
                  ? std::make_unique<Expression>(*other.outer_expression_)
                  : nullptr),
          substitution_map_(other.substitution_map_),
          derivative_evaluator_(other.derivative_evaluator_),
          has_derivative_evaluator_(other.has_derivative_evaluator_),
          has_bound_variable_(other.has_bound_variable_)
    {
        if (has_bound_variable_) {
            variables_ = other.variables_;
        }
    }
    Expression& Expression::operator=(const Expression& other)
    {
        if (this != &other) {
            expression_string_ = other.expression_string_;
            is_parsed_ = false;  // Force re-parsing
            compiled_expression_.reset();
            symbol_table_.reset();
            is_compound_ = other.is_compound_;
            outer_expression_ =
                other.outer_expression_
                    ? std::make_unique<Expression>(*other.outer_expression_)
                    : nullptr;
            substitution_map_ = other.substitution_map_;
            derivative_evaluator_ = other.derivative_evaluator_;
            has_derivative_evaluator_ = other.has_derivative_evaluator_;
            has_bound_variable_ = other.has_bound_variable_;
            if (has_bound_variable_)
                variables_ = other.variables_;
        }
        return *this;
    }
    Expression Expression::from_string(const std::string& expr_str)
    {
        return Expression(expr_str);
    }
    Expression Expression::constant(real value)
    {
        return Expression(std::to_string(value));
    }
    Expression Expression::zero()
    {
        return Expression("0");
    }
    Expression Expression::one()
    {
        return Expression("1");
    }
    const std::string& Expression::get_string() const
    {
        return expression_string_;
    }
    bool Expression::is_string_based() const
    {
        return true;  // Regular expressions are always string-based
    }
    void Expression::bind_variables(const ParameterMap<real>& bound_values)
    {
        // Merge bound values with existing ones
        for (std::size_t i = 0; i < bound_values.size(); ++i) {
            const char* name = bound_values.get_name_at(i);
            const real& value = bound_values.get_value_at(i);
            bind_variable(name, value);
        }

        has_bound_variable_ = true;
    }

    void Expression::bind_variable(const std::string& var_name, real value)
    {
        if (is_compound_) {
            for (auto& substitution : substitution_map_) {
                substitution.second.bind_variable(var_name, value);
            }
        }
        else {
            variables_.insert_or_assign(var_name.c_str(), value);
        }

        has_bound_variable_ = true;
    }
    bool Expression::has_bound_variables() const
    {
        return has_bound_variable_;
    }
    const ParameterMap<real>& Expression::get_bound_variables() const
    {
        return variables_;
    }
    const Expression::expression_type* Expression::get_compiled_expression()
        const
    {
        ensure_parsed();
        return compiled_expression_.get();
    }
    const Expression::symbol_table_type* Expression::get_symbol_table() const
    {
        ensure_parsed();
        return symbol_table_.get();
    }

    real Expression::evaluate_at(
        const ParameterMap<real>& variable_values) const
    {
        ++g_evaluate_calls;

        // Handle compound expressions
        if (is_compound_ && outer_expression_) {
            // Merge bound variables with provided values for substitution
            // evaluation

            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);

                if (std::find_if(
                        substitution_map_.begin(),
                        substitution_map_.end(),
                        [&name](const auto& pair) {
                            return std::strcmp(pair.first, name) == 0;
                        }) == substitution_map_.end()) {
                    const real& value = variable_values.get_value_at(i);
                    outer_expression_->bind_variable(name, value);
                }
            }

            // Evaluate substitutions
            ParameterMap<real> outer_values = variables_;
            for (std::size_t i = 0; i < substitution_map_.size(); ++i) {
                const auto& pair = substitution_map_[i];
                real sub_result = pair.second.evaluate_at(variable_values);
                outer_values.insert_or_assign(pair.first, sub_result);
            }

            return outer_expression_->evaluate_at(outer_values);
        }

        if (variables_.empty() || !has_bound_variable_) {
            variables_ = variable_values;
        }
        else
            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);
                const real& value = variable_values.get_value_at(i);
                variables_.insert_or_assign(name, value);
            }

        // Handle expressions created from DerivativeExpression
        if (has_derivative_evaluator_) {
            return derivative_evaluator_(variables_);
        }

        // Standard evaluation for non-compound expressions
        if (!is_parsed_ || !compiled_expression_) {
            parse_expression();
        }

        if (!compiled_expression_) {
            throw std::runtime_error(
                "Expression not properly parsed: " + expression_string_);
        }

        // Merge bound variables with provided values (provided values take
        // precedence)

        // ParameterMap<real> merged_values = variables_;

        // for (std::size_t i = 0; i < variable_values.size(); ++i) {
        //     const char* name = variable_values.get_name_at(i);
        //     const real& value = variable_values.get_value_at(i);
        //     merged_values.insert_or_assign(name, value);
        // }

        // Set all variables in variables_ for exprtk
        // for (std::size_t i = 0; i < merged_values.size(); ++i) {
        //    const char* name = merged_values.get_name_at(i);
        //    const real& value = merged_values.get_value_at(i);
        //    auto ptr = variables_.find(name);
        //    if (ptr)
        //        *ptr = value;
        //}

        real result = compiled_expression_->value();
        return result;
    }

    Expression Expression::operator+(const Expression& other) const
    {
        Expression add_expr("xx_ + yy_", { "xx_", "yy_" });
        return Expression(add_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator-(const Expression& other) const
    {
        Expression sub_expr("xx_ - yy_", { "xx_", "yy_" });
        return Expression(sub_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator*(const Expression& other) const
    {
        Expression mul_expr("xx_ * yy_", { "xx_", "yy_" });
        return Expression(mul_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator/(const Expression& other) const
    {
        Expression div_expr("xx_ / yy_", { "xx_", "yy_" });
        return Expression(div_expr, { { "xx_", *this }, { "yy_", other } });
    }

    Expression Expression::operator*(real scalar) const
    {
        Expression scalar_expr(std::to_string(scalar));
        Expression mul_expr("xx_ * yy_", { "xx_", "yy_" });
        return Expression(
            mul_expr, { { "xx_", scalar_expr }, { "yy_", *this } });
    }

    Expression Expression::operator-() const
    {
        Expression neg_expr("-xx_", { "xx_" });
        return Expression(neg_expr, { { "xx_", *this } });
    }

    DerivativeExpression Expression::derivative(
        const std::string& variable_name) const
    {
        // Use more conservative step sizes for float precision
        real h;
        if (is_compound_ && outer_expression_) {
            // Check if any of the substitutions are derivatives
            bool has_derivative_substitution = false;
            for (const auto& pair : substitution_map_) {
                if (pair.second.has_derivative_evaluator_) {
                    has_derivative_substitution = true;
                    break;
                }
            }
            // For compound expressions with derivatives, use significantly
            // larger step
            h = has_derivative_substitution ? real(5e-3) : real(1e-4);
        }
        else if (has_derivative_evaluator_) {
            // This is already a derivative, so we're computing second
            // derivative
            h = real(5e-3);
        }
        else {
            // Simple expression
            h = real(1e-4);
        }

        // For compound expressions, use numerical chain rule
        if (is_compound_ && outer_expression_) {
            auto compound_evaluator = [this](const ParameterMap<real>& values) {
                return this->evaluate_at(values);
            };
            auto derivative_func = create_compound_derivative_function(
                compound_evaluator, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        // Handle derivative expressions (derivatives of derivatives)
        else if (has_derivative_evaluator_) {
            auto derivative_func = create_compound_derivative_function(
                derivative_evaluator_, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        else {
            auto evaluator = [this](const ParameterMap<real>& values) {
                return this->evaluate_at(values);
            };
            auto derivative_func = create_compound_derivative_function(
                evaluator, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
    }
    void Expression::ensure_parsed() const
    {
        if (!is_parsed_ || !compiled_expression_) {
            parse_expression();
        }
    }
    void Expression::parse_expression() const
    {
        symbol_table_ = std::make_unique<symbol_table_type>();
        compiled_expression_ = std::make_unique<expression_type>();

        // Enable constants and standard functions
        symbol_table_->add_constants();

        // Register specified variables only
        if (!variables_.empty()) {
            for (int i = 0; i < variables_.size(); i++) {
                auto var_name = variables_.get_name_at(i);
                auto* var_ptr = &variables_.get_value_at(i);

                symbol_table_->add_variable(
                    var_name, const_cast<real&>(*var_ptr));
            }
        }

        // Register symbol table with expression
        compiled_expression_->register_symbol_table(*symbol_table_);

        if (!parser.compile(expression_string_, *compiled_expression_)) {
            throw std::runtime_error(
                "Failed to parse expression: " + expression_string_);
        }

        is_parsed_ = true;
    }

    void Expression::apply_recursive_substitution()
    {
        // If the outer expression is not compound, nothing special to do
        if (!outer_expression_ || !outer_expression_->is_compound_) {
            return;
        }

        // This method implements the recursive substitution logic you
        // described:
        // 1. Variables that are re-substituted don't get double-substituted
        // 2. New substitutions are applied to existing expressions recursively
        // Example: if outer has {xx: u1+u2} and new has {xx: u2, u2: u3}
        // Result should be {u2: u3, yy: whatever_yy_was} (xx is gone, u2
        // becomes u3)

        std::vector<std::pair<const char*, Expression>> final_substitutions;
        const auto& outer_substitutions = outer_expression_->substitution_map_;

        // Process each variable from the outer expression's substitution map
        for (const auto& outer_pair : outer_substitutions) {
            const char* var_name = outer_pair.first;
            Expression outer_expr =
                outer_pair.second;  // Make a copy for modification

            // Check if this variable is being re-substituted in our new
            // substitution map
            bool is_being_resubstituted = false;
            for (const auto& new_pair : substitution_map_) {
                if (std::strcmp(new_pair.first, var_name) == 0) {
                    // This variable is being re-substituted, so we skip the
                    // outer substitution (completed substitutions shouldn't be
                    // re-substituted)
                    is_being_resubstituted = true;
                    break;
                }
            }

            if (!is_being_resubstituted) {
                // Apply any relevant new substitutions to this outer expression
                // recursively
                for (const auto& new_pair : substitution_map_) {
                    const char* new_var = new_pair.first;
                    const Expression& new_expr = new_pair.second;

                    // Check if the outer expression contains the variable we're
                    // substituting
                    if (outer_expr.is_compound_) {
                        // For compound expressions, check if any substitution
                        // uses this variable
                        bool contains_var = false;
                        for (const auto& sub_pair :
                             outer_expr.substitution_map_) {
                            if (std::strcmp(sub_pair.first, new_var) == 0) {
                                contains_var = true;
                                break;
                            }
                        }
                        if (contains_var) {
                            // Apply the substitution recursively
                            outer_expr = Expression(
                                outer_expr, { { new_var, new_expr } });
                        }
                    }
                    else {
                        // For simple expressions, check if the variable appears
                        // in the string
                        if (outer_expr.expression_string_.find(new_var) !=
                            std::string::npos) {
                            // Create a compound expression with the
                            // substitution
                            outer_expr = Expression(
                                outer_expr, { { new_var, new_expr } });
                        }
                    }
                }

                // Add the (possibly recursively modified) outer substitution
                final_substitutions.emplace_back(var_name, outer_expr);
            }
        }

        // Add our new substitutions
        for (const auto& new_pair : substitution_map_) {
            final_substitutions.emplace_back(new_pair.first, new_pair.second);
        }

        // Update our substitution map with the final results
        substitution_map_ = std::move(final_substitutions);

        // Update the outer expression to be the base expression (without its
        // substitutions) This ensures we don't double-apply the outer
        // substitutions
        if (outer_expression_->outer_expression_) {
            outer_expression_ = std::make_unique<Expression>(
                *outer_expression_->outer_expression_);
        }
        else {
            // Create a simple expression with just the string
            Expression simple_expr(outer_expression_->expression_string_);
            outer_expression_ = std::make_unique<Expression>(simple_expr);
            outer_expression_->is_compound_ = false;
        }
    }

    Expression operator*(real scalar, const Expression& expr)
    {
        return expr * scalar;
    }
    Expression make_expression(const std::string& expr_str)
    {
        return Expression::from_string(expr_str);
    }

    std::vector<DerivativeExpression> Expression::gradient(
        const std::vector<std::string>& variable_names) const
    {
        std::vector<DerivativeExpression> grad;
        grad.reserve(variable_names.size());
        for (const auto& var : variable_names) {
            grad.push_back(derivative(var));
        }
        return grad;
    }

}  // namespace fem_bem
}  // namespace USTC_CG
