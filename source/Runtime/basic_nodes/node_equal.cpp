#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(equal)
{
    b.add_input<entt::meta_any>("A");
    b.add_input<entt::meta_any>("B");
    b.add_output<bool>("Equal");
};

NODE_EXECUTION_FUNCTION(equal)
{
    auto a = params.get_input<entt::meta_any>("A");
    auto b = params.get_input<entt::meta_any>("B");

    if (a.type() != b.type()) {
        return false;
    }

    auto type = a.type();

    if (type == entt::resolve<float>()) {
        float a_val = a.cast<float>();
        float b_val = b.cast<float>();
        // Compare floating point numbers with epsilon
        params.set_output("Equal", std::abs(a_val - b_val) < 1e-6);
        return true;
    }

    if (type == entt::resolve<double>()) {
        double a_val = a.cast<double>();
        double b_val = b.cast<double>();
        // Compare floating point numbers with epsilon
        params.set_output("Equal", std::abs(a_val - b_val) < 1e-6);
        return true;
    }

    params.set_output("Equal", a == b);
    return true;
};

NODE_DECLARATION_UI(equal);

NODE_DEF_CLOSE_SCOPE
