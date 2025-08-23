#include <pxr/base/vt/array.h>

#include <fem_bem/fem_bem.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/algorithms/delauney.h"
#include "nodes/core/def/node_def.hpp"

using namespace USTC_CG::fem_bem;
using namespace USTC_CG;

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(fem_solver)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<int>("Problem Dimension").default_val(2).min(2).max(3);
    b.add_input<int>("Element Family")
        .default_val(0)
        .min(0)
        .max(3);  // 0 for P_minus
    b.add_input<int>("K").default_val(0).min(0).max(3);
    b.add_input<int>("Equation Type")
        .default_val(0)
        .min(0)
        .max(3);  // 0 for Laplacian
    b.add_input<std::string>("Boundary Condition")
        .default_val("sin(atan2(y,x)*2)");
    b.add_input<int>("Boundary Type")
        .default_val(0)
        .min(0)
        .max(1);  // 0 for Dirichlet
    b.add_input<int>("Boundary Index")
        .default_val(0)
        .min(0)
        .max(1);  // Assuming a reasonable max index
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(fem_solver)
{
    // 获取输入参数
    Geometry input_geometry = params.get_input<Geometry>("Geometry");
    int problem_dim = params.get_input<int>("Problem Dimension");
    int element_family = params.get_input<int>("Element Family");
    int k = params.get_input<int>("K");
    int equation_type = params.get_input<int>("Equation Type");
    std::string boundary_condition =
        params.get_input<std::string>("Boundary Condition").c_str();
    int boundary_type = params.get_input<int>("Boundary Type");
    int boundary_index = params.get_input<int>("Boundary Index");

    // 对输入几何体进行Delaunay三角化
    auto delauneyed = geom_algorithm::delaunay(input_geometry);

    // 创建求解器描述
    ElementSolverDesc desc;
    desc.set_problem_dim(problem_dim);

    // 设置单元族
    if (element_family == 0) {
        desc.set_element_family(ElementFamily::P_minus);
    }

    desc.set_k(k);

    // 设置方程类型
    if (equation_type == 0) {
        desc.set_equation_type(EquationType::Laplacian);
    }

    // 创建单元求解器
    auto solver = create_element_solver(desc);

    // 设置几何体
    solver->set_geometry(delauneyed);

    // 获取边界数量
    auto n = solver->get_boundary_count();

    // 设置边界条件
    BoundaryCondition bc_type;
    if (boundary_type == 0) {
        bc_type = BoundaryCondition::Dirichlet;
    }
    else {
        bc_type = BoundaryCondition::Neumann;
    }

    solver->set_boundary_condition(boundary_condition, bc_type, boundary_index);

    // 求解
    try {
        pxr::VtArray<float> solution = solver->solve();
        // 将解添加到网格组件中
        auto mesh_component = delauneyed.get_component<MeshComponent>();
        mesh_component->add_vertex_scalar_quantity("solution", solution);

        // 输出结果几何体
        params.set_output("Geometry", delauneyed);

        return true;
    }
    catch (const std::exception& e) {
        // 处理异常，例如记录错误信息
        std::cerr << "Error during FEM solve: " << e.what() << std::endl;
        return false;  // 或者其他适当的错误处理
    }
}

NODE_DECLARATION_UI(fem_solver);
NODE_DEF_CLOSE_SCOPE
