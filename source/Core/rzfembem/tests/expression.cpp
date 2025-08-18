
#include "fem_bem/Expression.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace USTC_CG::fem_bem;

// Test Expression class specific functionality - Factory methods
TEST(ExpressionFocusedTest, FactoryMethods)
{
    auto expr1 = Expression::from_string("x + y");
    EXPECT_EQ(expr1.get_string(), "x + y");

    auto expr2 = Expression::constant(42.0);
    EXPECT_EQ(expr2.get_string(), "42.000000");

    auto expr3 = Expression::zero();
    EXPECT_EQ(expr3.get_string(), "0");

    auto expr4 = Expression::one();
    EXPECT_EQ(expr4.get_string(), "1");
}

// Test Expression class specific functionality - Variable management API
TEST(ExpressionFocusedTest, VariableManagementAPI)
{
    Expression expr("a + b * c");

    // Test evaluate_at method since add_variable is removed
    ParameterMap<float> values = { { "a", 1.0 }, { "b", 2.0 }, { "c", 3.0 } };

    EXPECT_DOUBLE_EQ(expr.evaluate_at(values), 7.0);

    // Test with different values
    values.insert_or_assign("a", 5.0);
    EXPECT_DOUBLE_EQ(expr.evaluate_at(values), 11.0);
}

// Test Expression class specific functionality - evaluate_at method
TEST(ExpressionFocusedTest, EvaluateAtMethod)
{
    Expression expr("x^2 + y^2");

    // Test evaluate_at with variable values
    ParameterMap<float> temp_values = { { "x", 3.0 }, { "y", 4.0 } };

    double result = expr.evaluate_at(temp_values);
    EXPECT_DOUBLE_EQ(result, 25.0);  // 3^2 + 4^2 = 25

    // Test with different values
    temp_values = { { "x", 1.0 }, { "y", 1.0 } };
    EXPECT_DOUBLE_EQ(expr.evaluate_at(temp_values), 2.0);  // 1^2 + 1^2 = 2
}

// Test Expression class specific functionality - String modification API
TEST(ExpressionFocusedTest, StringModificationAPI)
{
    Expression expr("x + 1");

    // Test evaluate_at method instead of set_string
    EXPECT_DOUBLE_EQ(expr.evaluate_at({ { "x", 5.0 } }), 6.0);

    // Create a new expression for different behavior
    Expression expr2("x * 2");
    EXPECT_EQ(expr2.get_string(), "x * 2");
    EXPECT_DOUBLE_EQ(expr2.evaluate_at({ { "x", 5.0 } }), 10.0);
}

// Test Expression class specific functionality - Utility functions
TEST(ExpressionFocusedTest, UtilityFunctions)
{
    auto expr1 = make_expression("x + y");

    EXPECT_EQ(expr1.get_string(), "x + y");

    // Verify types
    static_assert(std::is_same_v<decltype(expr1), Expression>);
}

// Test Expression class specific functionality - Copy semantics
TEST(ExpressionFocusedTest, CopySemantics)
{
    Expression expr1("x * y + 5");

    // Copy constructor should work
    Expression expr2(expr1);
    EXPECT_EQ(expr2.get_string(), "x * y + 5");

    // Both expressions should evaluate to the same result with same inputs
    ParameterMap<float> values = { { "x", 2.0 }, { "y", 3.0 } };
    EXPECT_DOUBLE_EQ(expr1.evaluate_at(values), 11.0);  // 2*3+5
    EXPECT_DOUBLE_EQ(expr2.evaluate_at(values), 11.0);  // 2*3+5
}

// Test Expression class specific functionality - Integration interface
TEST(ExpressionFocusedTest, IntegrationInterface)
{
    Expression expr("x*x + y*y");

    // Test that integration methods exist and can be called
    std::vector<std::string> barycentric_vars;
    barycentric_vars.push_back("u");
    barycentric_vars.push_back("v");
    barycentric_vars.push_back("w");

    // These methods should exist even if not fully implemented
    try {
        expr.integrate_over_simplex(barycentric_vars, nullptr, 10);
    }
    catch (const std::exception&) {
        // Expected if not implemented - the important thing is the interface
        // exists
    }

    // Test integration with another expression
    Expression other("z");
    try {
        expr.integrate_product_with(other, barycentric_vars, nullptr, 10);
    }
    catch (const std::exception&) {
        // Expected if not implemented
    }
}

// Test Expression class specific functionality - Template specializations
TEST(ExpressionFocusedTest, TemplateSpecializations)
{
    // Test double specialization
    Expression double_expr("x + y");
    EXPECT_DOUBLE_EQ(
        double_expr.evaluate_at({ { "x", 1.5 }, { "y", 2.5 } }), 4.0);

    // Test float specialization
    Expression float_expr("a * b");
    EXPECT_NEAR(
        float_expr.evaluate_at({ { "a", 2.0f }, { "b", 3.0f } }), 6.0f, 1e-6f);
}

// Test Expression class specific functionality - Compilation caching
TEST(ExpressionFocusedTest, CompilationCaching)
{
    Expression expr("x + y + z");

    // First evaluation should compile the expression
    double result1 =
        expr.evaluate_at({ { "x", 1.0 }, { "y", 2.0 }, { "z", 3.0 } });
    EXPECT_DOUBLE_EQ(result1, 6.0);

    // Second evaluation should use cached compilation
    double result2 =
        expr.evaluate_at({ { "x", 1.0 }, { "y", 2.0 }, { "z", 3.0 } });
    EXPECT_DOUBLE_EQ(result2, 6.0);

    // Test with different values
    double result3 =
        expr.evaluate_at({ { "x", 10.0 }, { "y", 2.0 }, { "z", 3.0 } });
    EXPECT_DOUBLE_EQ(result3, 15.0);
}

// Test Expression class specific functionality - Error handling
TEST(ExpressionFocusedTest, ErrorHandling)
{
    // Test expression with syntax error - will fail during evaluate_at
    Expression invalid_expr("x + * y");
    EXPECT_THROW(
        invalid_expr.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }),
        std::runtime_error);

    // Test empty expression
    Expression empty_expr;
    EXPECT_EQ(empty_expr.get_string(), "");
    EXPECT_THROW(
        empty_expr.evaluate_at(ParameterMap<double>{}), std::runtime_error);
}

// Test Expression class specific functionality - Derivative interface
TEST(ExpressionFocusedTest, DerivativeInterface)
{
    Expression expr("x^2 + y");

    // Test derivative method (placeholder implementation)
    auto dx = expr.derivative("x");

    EXPECT_EQ(expr.is_string_based(), true);
    EXPECT_EQ(dx.is_string_based(), false);
    EXPECT_EQ(dx.get_string(), "");

    EXPECT_NEAR(dx.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }), 2.0, 1e-6);

    Expression expr2("x^2 + y^2 + z");
    auto grad = expr2.gradient({ "x", "y" });
    EXPECT_EQ(grad.size(), 2);

    auto& dx2 = grad[0];
    auto& dy2 = grad[1];
    EXPECT_EQ(dx2.get_string(), "");
    EXPECT_EQ(dy2.get_string(), "");
    EXPECT_NEAR(
        dx2.evaluate_at({ { "x", 1.0 }, { "y", 2.0 }, { "z", 3.0 } }),
        2.0,
        1e-6);
    EXPECT_NEAR(
        dy2.evaluate_at({ { "x", 1.0 }, { "y", 2.0 }, { "z", 3.0 } }),
        4.0,
        1e-6);
}

// Test Expression class specific functionality - Derivative interface
TEST(ExpressionFocusedTest, CompoundExpression)
{
    Expression expr2("x + y");
    Expression element1("u+v");
    Expression element2("(u-v)^2");
    Expression compound(expr2, { { "x", element1 }, { "y", element2 } });

    auto eval = compound.evaluate_at({ { "u", 1.0 }, { "v", 2.0 } });
    EXPECT_DOUBLE_EQ(
        eval, 1.0 + 2.0 + (1.0 - 2.0) * (1.0 - 2.0));  // u + v + (u - v)^2

    auto derivative = compound.derivative("u");  // 1 + 2 * (u - v)

    auto eval_derivative =
        derivative.evaluate_at({ { "u", 1.0 }, { "v", 2.0 } });
    EXPECT_NEAR(eval_derivative, 1.0 + 2.0 * (1.0 - 2.0), 1e-9);

    // Test that derivative can be used in compound expressions
    auto compound2 = Expression(
        expr2,
        { { "x", element1 }, { "y", derivative } });  // u + v + 1 + 2 * (u - v)
    auto eval2 = compound2.evaluate_at({ { "u", 1.0 }, { "v", 2.0 } });
    EXPECT_NEAR(
        eval2,
        1.0 + 2.0 + 1.0 + 2.0 * (1.0 - 2.0),
        1e-9);  // u + v + 1 + 2 * (u - v) = 3 * u - v + 1

    for (float u = 0.0f; u <= 1.0f; u += 0.1f) {
        for (float v = 0.0f; v <= 1.0f; v += 0.1f) {
            auto eval_at_uv = compound2.evaluate_at({ { "u", u }, { "v", v } });
            EXPECT_NEAR(
                eval_at_uv,
                3 * u - v + 1,
                1e-6);  // Check against expected linear form
        }
    }

    auto rst = compound2.integrate_over_simplex({ "u", "v" }, nullptr, 60);

    EXPECT_NEAR(rst, 5.0 / 3.0, 1e-3);  // Integral over simplex should be 5/6

    auto dc2_du = compound2.derivative("u");
    EXPECT_EQ(dc2_du.get_string(), "");
    EXPECT_EQ(dc2_du.is_string_based(), false);
    EXPECT_NEAR(dc2_du.evaluate_at({ { "u", 1.0 }, { "v", 2.0 } }), 3.0, 1e-4);
    EXPECT_NEAR(dc2_du.evaluate_at({ { "u", 0.0 }, { "v", 0.0 } }), 3.0, 1e-4);
}

// Add, sub, basic ops
TEST(ExpressionFocusedTest, BasicArithmeticOperations)
{
    Expression expr1("x + 2");
    Expression expr2("y * 3");

    // Test addition
    auto sum = expr1 + expr2;
    EXPECT_EQ(sum.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }), 1.0 + 2 + 2 * 3);

    // Test subtraction
    auto diff = expr1 - expr2;
    EXPECT_EQ(
        diff.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }), 1.0 + 2 - 2 * 3);

    // Test multiplication
    auto prod = expr1 * expr2;
    EXPECT_EQ(
        prod.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }), (1.0 + 2) * (2 * 3));

    // Test division
    auto quot = expr1 / expr2;
    EXPECT_EQ(
        quot.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }), (1.0 + 2) / (2 * 3));

    // More compound

    auto rss = sum + diff * prod / quot;
    EXPECT_EQ(
        rss.evaluate_at({ { "x", 1.0 }, { "y", 2.0 } }),
        (1.0 + 2 + 2 * 3) + ((1.0 + 2 - 2 * 3) * ((1.0 + 2) * (2 * 3))) /
                                ((1.0 + 2) / (2 * 3)));
}
