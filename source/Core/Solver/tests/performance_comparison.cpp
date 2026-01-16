#include <gtest/gtest.h>

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "RZSolver/Solver.hpp"

using namespace Ruzino::Solver;

// 创建拉普拉斯矩阵测试问题
Eigen::SparseMatrix<float> createLaplacian(int n)
{
    Eigen::SparseMatrix<float> A(n, n);
    std::vector<Eigen::Triplet<float>> triplets;

    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 2.0f);
        if (i > 0)
            triplets.emplace_back(i, i - 1, -1.0f);
        if (i < n - 1)
            triplets.emplace_back(i, i + 1, -1.0f);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

TEST(PerformanceComparison, CuSolverVsCudaCG)
{
    try {
        std::vector<int> sizes = { 100, 500, 1000, 2000, 5000 };

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "性能对比: cuSOLVER QR (直接法) vs CUDA CG (迭代法)\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << std::setw(10) << "矩阵大小" << std::setw(15)
                  << "cuSOLVER (μs)" << std::setw(15) << "CUDA CG (μs)"
                  << std::setw(12) << "CG 迭代数" << std::setw(15) << "加速比"
                  << std::setw(15) << "残差 (QR)" << std::setw(15)
                  << "残差 (CG)" << "\n";
        std::cout << std::string(97, '-') << "\n";

        for (int n : sizes) {
            auto A = createLaplacian(n);
            Eigen::VectorXf b = Eigen::VectorXf::Random(n);

            // cuSOLVER QR
            auto cusolver = SolverFactory::create(SolverType::CUSOLVER_QR);
            Eigen::VectorXf x_cusolver = Eigen::VectorXf::Zero(n);
            SolverConfig config;
            config.verbose = false;
            config.tolerance = 1e-6f;

            auto result_cusolver = cusolver->solve(A, b, x_cusolver, config);
            float residual_cusolver = (A * x_cusolver - b).norm();

            // CUDA CG
            auto cuda_cg = SolverFactory::create(SolverType::CUDA_CG);
            Eigen::VectorXf x_cg = Eigen::VectorXf::Zero(n);
            config.max_iterations = 1000;

            auto result_cg = cuda_cg->solve(A, b, x_cg, config);
            float residual_cg = (A * x_cg - b).norm();

            float speedup =
                static_cast<float>(result_cg.solve_time.count()) /
                static_cast<float>(result_cusolver.solve_time.count());

            std::cout << std::setw(10) << n << std::setw(15)
                      << result_cusolver.solve_time.count() << std::setw(15)
                      << result_cg.solve_time.count() << std::setw(12)
                      << result_cg.iterations << std::setw(14) << std::fixed
                      << std::setprecision(2) << speedup << "x" << std::setw(15)
                      << std::scientific << std::setprecision(2)
                      << residual_cusolver << std::setw(15) << residual_cg
                      << "\n";
        }

        std::cout << std::string(97, '-') << "\n\n";

        std::cout << "结论:\n";
        std::cout << "  • cuSOLVER QR: 直接法，单次求解，适合中小规模问题\n";
        std::cout
            << "  • CUDA CG: 迭代法，需要多次迭代，但对大规模稀疏矩阵更高效\n";
        std::cout
            << "  • 加速比 > 1: cuSOLVER更快；加速比 < 1: CUDA CG更快\n\n";
    }
    catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA/cuSOLVER not available: " << e.what();
    }
}

TEST(PerformanceComparison, AllSolvers)
{
    try {
        int n = 1000;
        auto A = createLaplacian(n);
        Eigen::VectorXf b = Eigen::VectorXf::Random(n);

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "所有求解器性能对比 (1000x1000 拉普拉斯矩阵)\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << std::setw(25) << "求解器" << std::setw(15)
                  << "求解时间 (μs)" << std::setw(12) << "迭代数"
                  << std::setw(15) << "残差范数" << "\n";
        std::cout << std::string(67, '-') << "\n";

        SolverConfig config;
        config.verbose = false;
        config.tolerance = 1e-6f;
        config.max_iterations = 1000;

        std::vector<SolverType> solver_types = { SolverType::CUSOLVER_QR,
                                                 SolverType::CUDA_CG,
                                                 SolverType::CUDA_BICGSTAB,
                                                 SolverType::CUDA_GMRES };

        for (auto type : solver_types) {
            try {
                auto solver = SolverFactory::create(type);
                Eigen::VectorXf x = Eigen::VectorXf::Zero(n);

                auto result = solver->solve(A, b, x, config);
                float residual = (A * x - b).norm();

                std::cout << std::setw(25) << solver->getName() << std::setw(15)
                          << result.solve_time.count() << std::setw(12)
                          << result.iterations << std::setw(15)
                          << std::scientific << std::setprecision(2) << residual
                          << "\n";
            }
            catch (const std::exception& e) {
                std::cout << std::setw(25) << "Solver "
                          << static_cast<int>(type) << std::setw(15) << "N/A"
                          << std::setw(12) << "-" << std::setw(15) << e.what()
                          << "\n";
            }
        }

        std::cout << std::string(67, '-') << "\n\n";
    }
    catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}
