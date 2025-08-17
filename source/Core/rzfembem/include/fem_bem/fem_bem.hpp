#pragma once
#include <pxr/base/vt/array.h>

#include <memory>
#include <string>

#include "ElementBasis.hpp"
#include "GCore/GOP.h"
#include "api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

enum class ElementFamily { P_minus, P, Q_minus, S };

enum class EquationType { Laplacian, Heat, Wave, Elasticity };

enum class BoundaryCondition { Dirichlet, Neumann, Robin };

class RZFEMBEM_API ElementSolverDesc {
   public:
    ElementSolverDesc& set_problem_dim(unsigned dim)
    {
        problem_dim_ = dim;
        return *this;
    }
    ElementSolverDesc& set_element_family(ElementFamily family)
    {
        family_ = family;
        return *this;
    }
    ElementSolverDesc& set_k(unsigned k)
    {
        k_ = k;
        return *this;
    }
    ElementSolverDesc& set_equation_type(EquationType type)
    {
        equation_type_ = type;
        return *this;
    }

    unsigned get_problem_dim() const
    {
        return problem_dim_;
    }
    ElementFamily get_element_family() const
    {
        return family_;
    }
    unsigned get_k() const
    {
        return k_;
    }
    EquationType get_equation_type() const
    {
        return equation_type_;
    }

   private:
    unsigned problem_dim_ = 2;
    ElementFamily family_ = ElementFamily::P_minus;
    unsigned k_ = 0;
    EquationType equation_type_ = EquationType::Laplacian;
};

class RZFEMBEM_API ElementSolver {
   public:
    virtual ~ElementSolver() = default;

    virtual void set_geometry(const Geometry& geom) = 0;
    virtual unsigned get_boundary_count() const = 0;
    virtual void set_boundary_condition(
        const std::string& expr,
        BoundaryCondition type,
        unsigned boundary_id) = 0;
    virtual pxr::VtArray<float> solve() = 0;
};

RZFEMBEM_API std::shared_ptr<ElementSolver> create_element_solver(
    const ElementSolverDesc& desc);

USTC_CG_NAMESPACE_CLOSE_SCOPE
