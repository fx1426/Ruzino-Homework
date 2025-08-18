#include <fem_bem/Expression.hpp>
namespace USTC_CG {
namespace fem_bem {

    real numerical_derivative(
        const exprtk::expression<real>& expr,
        exprtk::details::variable_node<real>* var,
        const real& h)
    {
        const real x_init = var->ref();
        const real _2h = real(2) * h;

        var->ref() = x_init + _2h;
        const real y0 = expr.value();
        var->ref() = x_init + h;
        const real y1 = expr.value();
        var->ref() = x_init - h;
        const real y2 = expr.value();
        var->ref() = x_init - _2h;
        const real y3 = expr.value();
        var->ref() = x_init;

        return (-y0 + real(8) * (y1 - y2) + y3) / (real(12) * h);
    }
    std::function<real(const ParameterMap<real>&)> create_derivative_function(
        const std::string& expression_string,
        const std::string& variable_name,
        const real& h = real(1e-6))
    {
        return [expression_string, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            // Create a temporary expression for derivative computation
            exprtk::symbol_table<real> symbol_table;
            exprtk::expression<real> expr;
            exprtk::parser<real> parser;

            // Add all variables to symbol table
            ParameterMap<real> temp_values = values;

            for (std::size_t i = 0; i < temp_values.size(); ++i) {
                const char* name = temp_values.get_name_at(i);
                const real& value = temp_values.get_value_at(i);
                symbol_table.add_variable(
                    name, const_cast<real&>(temp_values.get_value_at(i)));
            }

            symbol_table.add_constants();
            expr.register_symbol_table(symbol_table);

            if (!parser.compile(expression_string, expr)) {
                return real(0);  // Return 0 for invalid expressions
            }

            // Find the variable node for differentiation
            auto* var_node = symbol_table.get_variable(variable_name);
            if (!var_node) {
                return real(0);  // Variable not found
            }

            return numerical_derivative(expr, var_node, h);
        };
    }

    // Create a derivative function for compound expressions using chain rule
    std::function<real(const ParameterMap<real>&)>
    create_compound_derivative_function(
        const std::function<real(const ParameterMap<real>&)>&
            compound_evaluator,
        const std::string& variable_name,
        const real& h = real(1e-6))
    {
        return [compound_evaluator, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            const real* var_value = values.find(variable_name.c_str());
            if (!var_value) {
                return real(0);  // Variable not found
            }

            const real x_init = *var_value;

            // Create modified value maps for derivative computation
            ParameterMap<real> values_h = values;
            values_h.insert_or_assign(variable_name.c_str(), x_init + h);
            const real y_plus = compound_evaluator(values_h);

            values_h.insert_or_assign(variable_name.c_str(), x_init - h);

            // Use simple 2-point central difference for better numerical
            // stability
            const real y_minus = compound_evaluator(values_h);

            return (y_plus - y_minus) / (real(2) * h);
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
          variable_names_(variable_names),
          is_compound_(false)
    {
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
        expression_string_ = outer_expr.get_string();
        for (const auto& pair : variable_substitutions) {
            expression_string_ += " with " + std::string(pair.first) + "=(" +
                                  pair.second.get_string() + ")";
        }
    }
    Expression::Expression(
        const Expression& outer_expr,
        std::initializer_list<std::pair<const char*, Expression>> substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(substitutions.begin(), substitutions.end()),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string();
        for (const auto& pair : substitution_map_) {
            expression_string_ += " with " + std::string(pair.first) + "=(" +
                                  pair.second.get_string() + ")";
        }
    }
    Expression::Expression(const Expression& other)
        : expression_string_(other.expression_string_),
          variable_names_(other.variable_names_),
          is_parsed_(false),  // Force re-parsing with new symbol table
          is_compound_(other.is_compound_),
          outer_expression_(
              other.outer_expression_
                  ? std::make_unique<Expression>(*other.outer_expression_)
                  : nullptr),
          substitution_map_(other.substitution_map_),
          derivative_evaluator_(other.derivative_evaluator_)
    {
    }
    Expression& Expression::operator=(const Expression& other)
    {
        if (this != &other) {
            expression_string_ = other.expression_string_;
            variable_names_ = other.variable_names_;
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
        }
        return *this;
    }
    real Expression::evaluate_at(
        const ParameterMap<real>& variable_values) const
    {
        // Handle expressions created from DerivativeExpression
        if (derivative_evaluator_) {
            return derivative_evaluator_(variable_values);
        }

        // Handle compound expressions
        if (is_compound_ && outer_expression_) {
            // Evaluate substitutions first
            ParameterMap<real> outer_values;

            for (std::size_t i = 0; i < substitution_map_.size(); ++i) {
                const auto& pair = substitution_map_[i];
                real sub_result = pair.second.evaluate_at(variable_values);
                outer_values.insert_or_assign(pair.first, sub_result);
            }

            return outer_expression_->evaluate_at(outer_values);
        }

        // Standard evaluation for non-compound expressions
        if (!is_parsed_ || !compiled_expression_) {
            // If no variables specified, discover them from the values
            // provided
            if (variable_names_.empty()) {
                for (std::size_t i = 0; i < variable_values.size(); ++i) {
                    variable_names_.push_back(variable_values.get_name_at(i));
                }
            }
            parse_expression();
        }

        if (!compiled_expression_) {
            throw std::runtime_error(
                "Expression not properly parsed: " + expression_string_);
        }

        // Store original values

        const exprtk::symbol_table<real>& sym_table =
            compiled_expression_->get_symbol_table();

        for (std::size_t i = 0; i < variable_values.size(); ++i) {
            const char* name = variable_values.get_name_at(i);
            const real& value = variable_values.get_value_at(i);
            *temp_variables_.find(name) = value;
        }

        real result = compiled_expression_->value();

        return result;
    }

    Expression Expression::operator+(const Expression& other) const
    {
        Expression add_expr("x + y", { "x", "y" });
        return Expression(add_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator-(const Expression& other) const
    {
        Expression sub_expr("x - y", { "x", "y" });
        return Expression(sub_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator*(const Expression& other) const
    {
        Expression mul_expr("x * y", { "x", "y" });
        return Expression(mul_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator/(const Expression& other) const
    {
        Expression div_expr("x / y", { "x", "y" });
        return Expression(div_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator*(real scalar) const
    {
        Expression scalar_expr(std::to_string(scalar));
        Expression mul_expr("x * y", { "x", "y" });
        return Expression(mul_expr, { { "x", scalar_expr }, { "y", *this } });
    }

    Expression Expression::operator-() const
    {
        Expression neg_expr("-x", { "x" });
        return Expression(neg_expr, { { "x", *this } });
    }

    DerivativeExpression Expression::derivative(
        const std::string& variable_name) const
    {
        // For compound expressions, use numerical chain rule
        if (is_compound_ && outer_expression_) {
            auto compound_evaluator = [this](const ParameterMap<real>& values) {
                return this->evaluate_at(values);
            };
            auto derivative_func = create_compound_derivative_function(
                compound_evaluator, variable_name);
            return DerivativeExpression(derivative_func, variable_name);
        }
        // Handle derivative expressions (derivatives of derivatives)
        else if (derivative_evaluator_) {
            auto derivative_func = create_compound_derivative_function(
                derivative_evaluator_, variable_name);
            return DerivativeExpression(derivative_func, variable_name);
        }
        else {
            // For simple expressions, use string-based derivative
            auto derivative_func =
                create_derivative_function(expression_string_, variable_name);
            return DerivativeExpression(derivative_func, variable_name);
        }
    }
    void Expression::parse_expression() const
    {
        symbol_table_ = std::make_unique<symbol_table_type>();
        compiled_expression_ = std::make_unique<expression_type>();

        // Enable constants and standard functions
        symbol_table_->add_constants();

        // Register specified variables only
        if (!variable_names_.empty()) {
            for (const auto& var_name : variable_names_) {
                auto* temp_var_ptr = temp_variables_.find(var_name.c_str());
                if (!temp_var_ptr) {
                    temp_variables_.insert_or_assign(var_name.c_str(), real(0));
                    temp_var_ptr = temp_variables_.find(var_name.c_str());
                }
                symbol_table_->add_variable(
                    var_name, const_cast<real&>(*temp_var_ptr));
            }
        }

        // Register symbol table with expression
        compiled_expression_->register_symbol_table(*symbol_table_);

        parser_type parser;
        if (!parser.compile(expression_string_, *compiled_expression_)) {
            throw std::runtime_error(
                "Failed to parse expression: " + expression_string_);
        }

        is_parsed_ = true;
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
