#pragma once

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <chrono>
#include <memory>
#include <string>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace Solver {

enum class SolverType {
    CUDA_CG,
    CUDA_BICGSTAB,
    CUDA_GMRES,   // 新增
    CUSOLVER_QR,  // cuSOLVER-based QR direct solver
    EIGEN_ITERATIVE_CG,
    EIGEN_ITERATIVE_BICGSTAB,
    EIGEN_DIRECT_LU,
    EIGEN_DIRECT_CHOLESKY,
    EIGEN_DIRECT_QR
};

struct SolverConfig {
    float tolerance = 1e-6f;
    int max_iterations = 1000;
    bool use_preconditioner = true;
    bool verbose = false;
};

struct SolverResult {
    bool converged = false;
    int iterations = 0;
    float final_residual = 0.0f;
    std::chrono::microseconds solve_time{ 0 };
    std::chrono::microseconds setup_time{ 0 };
    std::string error_message;
};

class RZSOLVER_API LinearSolver {
   public:
    virtual ~LinearSolver() = default;

    virtual SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}) = 0;

    // GPU-only interface: solve directly with GPU buffers (CSR format)
    // A: CSR sparse matrix (row_offsets, col_indices, values)
    // b: dense vector on GPU
    // x: dense vector on GPU (solution output)
    virtual SolverResult solveGPU(
        int n,
        int nnz,
        const int* d_row_offsets,
        const int* d_col_indices,
        const float* d_values,
        const float* d_b,
        float* d_x,
        const SolverConfig& config = SolverConfig{})
    {
        throw std::runtime_error(
            getName() + " does not support GPU-only interface");
    }

    virtual std::string getName() const = 0;
    virtual bool isIterative() const = 0;
    virtual bool requiresGPU() const = 0;
};

class RZSOLVER_API SolverFactory {
   public:
    static std::unique_ptr<LinearSolver> create(SolverType type);
    static std::vector<SolverType> getAvailableTypes();
    static std::string getTypeName(SolverType type);
};

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
