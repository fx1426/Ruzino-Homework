#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// CUDA kernel for vector operations (single precision)
__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorScale(const float* a, float alpha, float* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = alpha * a[idx];
    }
}

class CudaCGSolver {
   private:
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

   public:
    CudaCGSolver()
    {
        cusparseCreate(&cusparseHandle);
        cublasCreate(&cublasHandle);
    }

    ~CudaCGSolver()
    {
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
    }

    float solveCG(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        float tol = 1e-6f,
        int maxIter = 1000,
        bool use_preconditioner = true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        int n = A.rows();
        int nnz = A.nonZeros();

        std::cout << "Solving CG system: n=" << n << ", nnz=" << nnz
                  << std::endl;

        // Convert Eigen sparse matrix to CSR format (FIXED: correct row/col
        // mapping)
        std::vector<int> csrRowPtr(n + 1, 0);
        std::vector<int> csrColInd(nnz);
        std::vector<float> csrValues(nnz);
        std::vector<float> diagonal(n, 1.0f);  // For preconditioning

        // First pass: count entries per row
        for (int k = 0; k < A.outerSize(); ++k) {
            for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it; ++it) {
                csrRowPtr[it.row() + 1]++;
            }
        }

        // Convert counts to offsets
        for (int i = 1; i <= n; ++i) {
            csrRowPtr[i] += csrRowPtr[i - 1];
        }

        // Second pass: fill values and column indices
        std::vector<int> current_pos = csrRowPtr;
        for (int k = 0; k < A.outerSize(); ++k) {
            for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it; ++it) {
                int row = it.row();
                int pos = current_pos[row]++;
                csrValues[pos] = it.value();
                csrColInd[pos] = it.col();

                // Extract diagonal for preconditioning
                if (it.row() == it.col()) {
                    diagonal[row] = it.value();
                }
            }
        }

        // GPU setup
        auto d_csrValues = USTC_CG::cuda::create_cuda_linear_buffer(csrValues);
        auto d_csrRowPtr = USTC_CG::cuda::create_cuda_linear_buffer(csrRowPtr);
        auto d_csrColInd = USTC_CG::cuda::create_cuda_linear_buffer(csrColInd);
        auto d_diagonal = USTC_CG::cuda::create_cuda_linear_buffer(diagonal);
        auto d_b = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
        auto d_x = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
        auto d_r = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
        auto d_z = USTC_CG::cuda::create_cuda_linear_buffer<float>(
            n);  // For preconditioning
        auto d_p = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
        auto d_Ap = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);

        // Copy data to GPU
        d_b->assign_host_vector(
            std::vector<float>(b.data(), b.data() + b.size()));
        d_x->assign_host_vector(
            std::vector<float>(x.data(), x.data() + x.size()));

        // Create sparse matrix descriptor
        cusparseSpMatDescr_t matA_desc;
        cusparseCreateCsr(
            &matA_desc,
            n,
            n,
            nnz,
            reinterpret_cast<void*>(d_csrRowPtr->get_device_ptr()),
            reinterpret_cast<void*>(d_csrColInd->get_device_ptr()),
            reinterpret_cast<void*>(d_csrValues->get_device_ptr()),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F);

        // Create dense vector descriptors
        cusparseDnVecDescr_t vecX_desc, vecB_desc, vecR_desc, vecZ_desc,
            vecP_desc, vecAp_desc;
        cusparseCreateDnVec(
            &vecX_desc,
            n,
            reinterpret_cast<void*>(d_x->get_device_ptr()),
            CUDA_R_32F);
        cusparseCreateDnVec(
            &vecB_desc,
            n,
            reinterpret_cast<void*>(d_b->get_device_ptr()),
            CUDA_R_32F);
        cusparseCreateDnVec(
            &vecR_desc,
            n,
            reinterpret_cast<void*>(d_r->get_device_ptr()),
            CUDA_R_32F);
        cusparseCreateDnVec(
            &vecZ_desc,
            n,
            reinterpret_cast<void*>(d_z->get_device_ptr()),
            CUDA_R_32F);
        cusparseCreateDnVec(
            &vecP_desc,
            n,
            reinterpret_cast<void*>(d_p->get_device_ptr()),
            CUDA_R_32F);
        cusparseCreateDnVec(
            &vecAp_desc,
            n,
            reinterpret_cast<void*>(d_Ap->get_device_ptr()),
            CUDA_R_32F);

        // Allocate workspace for SpMV
        size_t bufferSize = 0;
        const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;
        cusparseSpMV_bufferSize(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA_desc,
            vecP_desc,
            &zero,
            vecAp_desc,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize);
        auto dBuffer =
            USTC_CG::cuda::create_cuda_linear_buffer<uint8_t>(bufferSize);

        auto iteration_start_time = std::chrono::high_resolution_clock::now();

        // Preconditioned CG algorithm
        float alpha, beta, rzold, rznew;
        float b_norm;

        // Compute ||b|| for relative residual
        cublasSdot(
            cublasHandle,
            n,
            d_b->get_device_ptr<float>(),
            1,
            d_b->get_device_ptr<float>(),
            1,
            &b_norm);
        b_norm = sqrt(b_norm);

        // r = b - A*x
        cusparseSpMV(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA_desc,
            vecX_desc,
            &zero,
            vecAp_desc,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            reinterpret_cast<void*>(dBuffer->get_device_ptr()));
        cublasScopy(
            cublasHandle,
            n,
            d_b->get_device_ptr<float>(),
            1,
            d_r->get_device_ptr<float>(),
            1);
        cublasSaxpy(
            cublasHandle,
            n,
            &minus_one,
            d_Ap->get_device_ptr<float>(),
            1,
            d_r->get_device_ptr<float>(),
            1);

        // Apply preconditioner: z = M^(-1) * r (diagonal preconditioning)
        if (use_preconditioner) {
            // Element-wise division: z[i] = r[i] / diagonal[i]
            cublasScopy(
                cublasHandle,
                n,
                d_r->get_device_ptr<float>(),
                1,
                d_z->get_device_ptr<float>(),
                1);
            cublasSsbmv(
                cublasHandle,
                CUBLAS_FILL_MODE_UPPER,
                n,
                0,
                &one,
                d_diagonal->get_device_ptr<float>(),
                1,
                d_z->get_device_ptr<float>(),
                1,
                &zero,
                d_z->get_device_ptr<float>(),
                1);
        }
        else {
            cublasScopy(
                cublasHandle,
                n,
                d_r->get_device_ptr<float>(),
                1,
                d_z->get_device_ptr<float>(),
                1);
        }

        // p = z
        cublasScopy(
            cublasHandle,
            n,
            d_z->get_device_ptr<float>(),
            1,
            d_p->get_device_ptr<float>(),
            1);

        // rzold = r^T * z
        cublasSdot(
            cublasHandle,
            n,
            d_r->get_device_ptr<float>(),
            1,
            d_z->get_device_ptr<float>(),
            1,
            &rzold);

        float initial_residual = sqrt(rzold);
        std::cout << "Initial residual norm: " << initial_residual << std::endl;
        std::cout << "RHS norm: " << b_norm << std::endl;
        std::cout << "Using "
                  << (use_preconditioner ? "diagonal preconditioner"
                                         : "no preconditioner")
                  << std::endl;

        // CG iterations
        int converged_iter = maxIter;
        for (int iter = 0; iter < maxIter; ++iter) {
            // Ap = A * p
            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecP_desc,
                &zero,
                vecAp_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                reinterpret_cast<void*>(dBuffer->get_device_ptr()));

            // alpha = rzold / (p^T * Ap)
            float pAp;
            cublasSdot(
                cublasHandle,
                n,
                d_p->get_device_ptr<float>(),
                1,
                d_Ap->get_device_ptr<float>(),
                1,
                &pAp);

            // Check for breakdown
            if (abs(pAp) < 1e-20f) {
                std::cout << "CG breakdown: p^T * A * p = " << pAp
                          << " at iteration " << iter << std::endl;
                break;
            }

            alpha = rzold / pAp;

            // x = x + alpha * p
            cublasSaxpy(
                cublasHandle,
                n,
                &alpha,
                d_p->get_device_ptr<float>(),
                1,
                d_x->get_device_ptr<float>(),
                1);

            // r = r - alpha * Ap
            float neg_alpha = -alpha;
            cublasSaxpy(
                cublasHandle,
                n,
                &neg_alpha,
                d_Ap->get_device_ptr<float>(),
                1,
                d_r->get_device_ptr<float>(),
                1);

            // Apply preconditioner: z = M^(-1) * r
            if (use_preconditioner) {
                cublasScopy(
                    cublasHandle,
                    n,
                    d_r->get_device_ptr<float>(),
                    1,
                    d_z->get_device_ptr<float>(),
                    1);
                cublasSsbmv(
                    cublasHandle,
                    CUBLAS_FILL_MODE_UPPER,
                    n,
                    0,
                    &one,
                    d_diagonal->get_device_ptr<float>(),
                    1,
                    d_z->get_device_ptr<float>(),
                    1,
                    &zero,
                    d_z->get_device_ptr<float>(),
                    1);
            }
            else {
                cublasScopy(
                    cublasHandle,
                    n,
                    d_r->get_device_ptr<float>(),
                    1,
                    d_z->get_device_ptr<float>(),
                    1);
            }

            // rznew = r^T * z
            cublasSdot(
                cublasHandle,
                n,
                d_r->get_device_ptr<float>(),
                1,
                d_z->get_device_ptr<float>(),
                1,
                &rznew);

            // Check convergence (relative residual)
            float relative_residual = sqrt(rznew) / b_norm;
            if (relative_residual < tol) {
                std::cout << "Converged in " << iter + 1 << " iterations"
                          << std::endl;
                std::cout << "Final relative residual: " << relative_residual
                          << std::endl;
                converged_iter = iter + 1;
                break;
            }

            // Print progress for large problems
            if ((iter + 1) % 1000 == 0) {
                std::cout << "Iteration " << iter + 1
                          << ", relative residual: " << relative_residual
                          << std::endl;
            }

            // Check for numerical breakdown
            if (abs(rzold) < 1e-20f) {
                std::cout << "CG breakdown: r^T * z = " << rzold
                          << " at iteration " << iter << std::endl;
                break;
            }

            // beta = rznew / rzold
            beta = rznew / rzold;

            // p = z + beta * p
            cublasSscal(
                cublasHandle, n, &beta, d_p->get_device_ptr<float>(), 1);
            cublasSaxpy(
                cublasHandle,
                n,
                &one,
                d_z->get_device_ptr<float>(),
                1,
                d_p->get_device_ptr<float>(),
                1);

            rzold = rznew;
        }

        auto iteration_end_time = std::chrono::high_resolution_clock::now();

        // Copy result back to host
        auto result_vec = d_x->get_host_vector<float>();
        x = Eigen::Map<Eigen::VectorXf>(result_vec.data(), result_vec.size());

        // Compute final true residual for verification
        cusparseSpMV(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA_desc,
            vecX_desc,
            &zero,
            vecAp_desc,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            reinterpret_cast<void*>(dBuffer->get_device_ptr()));
        cublasScopy(
            cublasHandle,
            n,
            d_b->get_device_ptr<float>(),
            1,
            d_r->get_device_ptr<float>(),
            1);
        cublasSaxpy(
            cublasHandle,
            n,
            &minus_one,
            d_Ap->get_device_ptr<float>(),
            1,
            d_r->get_device_ptr<float>(),
            1);
        auto final_residual_vec = d_r->get_host_vector<float>();
        Eigen::VectorXf final_gpu_residual = Eigen::Map<Eigen::VectorXf>(
            final_residual_vec.data(), final_residual_vec.size());

        auto end_time = std::chrono::high_resolution_clock::now();

        // Print timing information
        auto iteration_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(
                iteration_end_time - iteration_start_time);
        auto total_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

        std::cout << "CG iterations (" << converged_iter
                  << " iters): " << iteration_duration.count() << " μs"
                  << std::endl;
        std::cout << "Total time: " << total_duration.count() << " μs ("
                  << total_duration.count() / 1000.0 << " ms)" << std::endl;
        std::cout << "GPU final residual norm: " << final_gpu_residual.norm()
                  << std::endl;

        // Cleanup
        cusparseDestroySpMat(matA_desc);
        cusparseDestroyDnVec(vecX_desc);
        cusparseDestroyDnVec(vecB_desc);
        cusparseDestroyDnVec(vecR_desc);
        cusparseDestroyDnVec(vecZ_desc);
        cusparseDestroyDnVec(vecP_desc);
        cusparseDestroyDnVec(vecAp_desc);

        return final_gpu_residual.norm();
    }
};

int main()
{
    // Test with multiple matrix sizes for performance comparison
    std::vector<int> matrix_sizes = { 1000, 5000, 10000, 50000, 100000 };

    for (int n : matrix_sizes) {
        std::cout << "\n==============================================="
                  << std::endl;
        std::cout << "Testing with matrix size: " << n << "x" << n << std::endl;
        std::cout << "==============================================="
                  << std::endl;

        auto matrix_creation_start = std::chrono::high_resolution_clock::now();

        // Create a large test sparse matrix (tridiagonal)
        Eigen::SparseMatrix<float> A(n, n);
        std::vector<Eigen::Triplet<float>> triplets;
        triplets.reserve(3 * n - 2);  // Optimize memory allocation

        for (int i = 0; i < n; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, 2.0f));
            if (i > 0)
                triplets.push_back(Eigen::Triplet<float>(i, i - 1, -1.0f));
            if (i < n - 1)
                triplets.push_back(Eigen::Triplet<float>(i, i + 1, -1.0f));
        }
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Create RHS vector
        Eigen::VectorXf b = Eigen::VectorXf::Ones(n);
        Eigen::VectorXf x = Eigen::VectorXf::Zero(n);

        auto matrix_creation_end = std::chrono::high_resolution_clock::now();
        auto matrix_creation_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                matrix_creation_end - matrix_creation_start);

        std::cout << "Matrix creation time: " << matrix_creation_time.count()
                  << " ms" << std::endl;
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Non-zeros: " << A.nonZeros() << std::endl;
        std::cout << "Sparsity: "
                  << (1.0 - (double)A.nonZeros() / (n * n)) * 100 << "%"
                  << std::endl;
        std::cout << "Memory usage (approx): "
                  << (A.nonZeros() * (sizeof(float) + sizeof(int)) +
                      n * sizeof(int)) /
                         (1024.0 * 1024.0)
                  << " MB" << std::endl;

        // Solve using CUDA CG
        CudaCGSolver solver;
        auto solve_start = std::chrono::high_resolution_clock::now();
        float final_gpu_residual = solver.solveCG(
            A,
            b,
            x,
            1e-9f,
            100000,
            true);  // Reduced max iterations, enable preconditioning
        auto solve_end = std::chrono::high_resolution_clock::now();

        auto total_solve_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                solve_end - solve_start);
        std::cout << "Total solve time: " << total_solve_time.count() << " ms"
                  << std::endl;

        // Show first few solution elements for smaller matrices
        std::cout << "First 10 solution elements: ";
        for (int i = 0; i < std::min(10, n); ++i) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;

        // Verify solution
        auto verification_start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXf residual = A * x - b;
        auto verification_end = std::chrono::high_resolution_clock::now();
        auto verification_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                verification_end - verification_start);

        std::cout << "CPU computed residual norm: " << residual.norm()
                  << std::endl;
        std::cout << "Verification time: " << verification_time.count() << " ms"
                  << std::endl;

        // Compare residual calculations
        std::cout << "=== Residual Comparison ===" << std::endl;
        std::cout << "GPU final residual norm (from CG): " << final_gpu_residual
                  << std::endl;
        std::cout << "CPU residual norm (A*x - b): " << residual.norm()
                  << std::endl;
        std::cout << "Relative difference: "
                  << abs(final_gpu_residual - residual.norm()) /
                         residual.norm() * 100.0
                  << "%" << std::endl;

        // Performance metrics
        double flops_per_iteration =
            2.0 * A.nonZeros() + 6.0 * n;  // Approximate FLOPs per CG iteration
        std::cout << "Performance metrics:" << std::endl;
        std::cout << "  Approximate FLOPs per iteration: "
                  << flops_per_iteration << std::endl;

        if (n >= 50000) {
            std::cout << "Skipping larger matrices for this test run..."
                      << std::endl;
            break;
        }
    }

    return 0;
}
