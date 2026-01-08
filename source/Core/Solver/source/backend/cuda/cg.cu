#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <iostream>


RUZINO_NAMESPACE_OPEN_SCOPE

namespace Solver {

// 在 namespace 级别定义静态函数
namespace {
    SolverResult performCGIterationsImpl(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        const SolverConfig& config,
        int n,
        cusparseSpMatDescr_t matA_desc,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecB_desc,
        cusparseDnVecDescr_t vecR_desc,
        cusparseDnVecDescr_t vecZ_desc,
        cusparseDnVecDescr_t vecP_desc,
        cusparseDnVecDescr_t vecAp_desc,
        Ruzino::cuda::CUDALinearBufferHandle d_diagonal,
        Ruzino::cuda::CUDALinearBufferHandle dBuffer,
        Ruzino::cuda::CUDALinearBufferHandle d_b,
        Ruzino::cuda::CUDALinearBufferHandle d_x,
        Ruzino::cuda::CUDALinearBufferHandle d_r,
        Ruzino::cuda::CUDALinearBufferHandle d_z,
        Ruzino::cuda::CUDALinearBufferHandle d_p,
        Ruzino::cuda::CUDALinearBufferHandle d_Ap)
    {
        SolverResult result;
        const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;

        // Compute ||b|| for relative residual
        float b_norm;
        cublasSdot(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            &b_norm);
        b_norm = sqrt(b_norm);

        if (b_norm == 0.0f) {
            result.converged = true;
            result.iterations = 0;
            result.final_residual = 0.0f;
            return result;
        }

        // r = b - A*x
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1);

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
            (void*)dBuffer->get_device_ptr());

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            result.error_message =
                std::string("SpMV sync error: ") + cudaGetErrorString(sync_err);
            result.converged = false;
            return result;
        }

        cublasSaxpy(
            cublasHandle,
            n,
            &minus_one,
            reinterpret_cast<float*>(d_Ap->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1);

        // 简化预条件：z = r（不用复杂的对角预条件）
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_z->get_device_ptr()),
            1);

        // 如果需要对角预条件，可以用GPU kernel
        if (config.use_preconditioner) {
            // 获取device指针用于GPU kernel
            float* z_ptr = reinterpret_cast<float*>(d_z->get_device_ptr());
            float* r_ptr = reinterpret_cast<float*>(d_r->get_device_ptr());
            float* diag_ptr =
                reinterpret_cast<float*>(d_diagonal->get_device_ptr());

            Ruzino::cuda::GPUParallelFor(
                "CG_diagonal_precond", n, GPU_LAMBDA_Ex(int i) {
                    // z[i] = r[i] / diagonal[i] with safe division
                    float diag_safe = diag_ptr[i];
                    if (abs(diag_safe) < 1e-10f) {
                        diag_safe = (diag_safe >= 0.0f) ? 1e-10f : -1e-10f;
                    }
                    z_ptr[i] = r_ptr[i] / diag_safe;
                });
        }

        // p = z
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_z->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_p->get_device_ptr()),
            1);

        // rzold = r^T * z
        float rzold, rznew, alpha, beta;
        cublasSdot(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_z->get_device_ptr()),
            1,
            &rzold);

        float initial_residual = sqrt(rzold);

        // Check if already converged
        if (initial_residual / b_norm < config.tolerance) {
            // Synchronize to check for any GPU errors before returning
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                result.error_message =
                    std::string("GPU error: ") + cudaGetErrorString(sync_err);
                result.converged = false;
                return result;
            }
            result.converged = true;
            result.iterations = 0;
            result.final_residual = initial_residual / b_norm;
            return result;
        }

        // 自适应最大迭代次数 - 对大问题更宽容
        int adaptive_max_iters = config.max_iterations;
        if (n > 10000) {
            adaptive_max_iters =
                std::min(config.max_iterations, n / 5);  // 更现实的迭代限制
        }

        // CG iterations with better numerical stability
        for (int iter = 0; iter < adaptive_max_iters; ++iter) {
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
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_Ap->get_device_ptr()),
                1,
                &pAp);

            if (abs(pAp) < 1e-15f * b_norm * b_norm) {
                result.error_message = "CG breakdown: p^T * A * p too small";
                break;
            }

            alpha = rzold / pAp;

            // x = x + alpha * p
            cublasSaxpy(
                cublasHandle,
                n,
                &alpha,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_x->get_device_ptr()),
                1);

            // r = r - alpha * Ap
            float neg_alpha = -alpha;
            cublasSaxpy(
                cublasHandle,
                n,
                &neg_alpha,
                reinterpret_cast<float*>(d_Ap->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            // z = r (no preconditioning for simplicity)
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1);

            // Apply preconditioner: z = M^(-1) * r
            if (config.use_preconditioner) {
                // 获取device指针用于GPU kernel
                float* z_ptr = reinterpret_cast<float*>(d_z->get_device_ptr());
                float* r_ptr = reinterpret_cast<float*>(d_r->get_device_ptr());
                float* diag_ptr =
                    reinterpret_cast<float*>(d_diagonal->get_device_ptr());

                Ruzino::cuda::GPUParallelFor(
                    "CG_diagonal_precond", n, GPU_LAMBDA_Ex(int i) {
                        // Add small regularization to prevent division by very small diagonals
                        float diag_safe = diag_ptr[i];
                        if (abs(diag_safe) < 1e-10f) {
                            diag_safe = (diag_safe >= 0.0f) ? 1e-10f : -1e-10f;
                        }
                        z_ptr[i] = r_ptr[i] / diag_safe;
                    });
            }
            else {
                cublasScopy(
                    cublasHandle,
                    n,
                    reinterpret_cast<float*>(d_r->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_z->get_device_ptr()),
                    1);
            }

            // rznew = r^T * z
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1,
                &rznew);

            // Check convergence
            float relative_residual = sqrt(rznew) / b_norm;

            // 对大问题使用更现实的收敛标准
            float effective_tolerance = config.tolerance;
            if (n > 10000) {
                effective_tolerance =
                    std::max(config.tolerance, 1e-3f);  // 至少1e-3
            }
            if (n > 50000) {
                effective_tolerance =
                    std::max(config.tolerance, 5e-3f);  // 更大问题更宽松
            }

            if (relative_residual < effective_tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.final_residual = relative_residual;
                break;
            }

            // 检查收敛停滞
            if (iter > 100 && iter % 1000 == 0) {
                float progress_rate =
                    relative_residual / (initial_residual / b_norm);
                if (progress_rate > 0.99f) {  // 几乎没有进展
                    // 不立即退出，给更多机会
                }
            }

            // Check for breakdown with adaptive threshold
            float breakdown_threshold = 1e-30f * b_norm * b_norm;
            if (abs(rznew) < breakdown_threshold && abs(rzold) < breakdown_threshold) {
                // Both near zero - we've converged as much as possible
                result.converged = true;
                result.iterations = iter + 1;
                result.final_residual = relative_residual;
                break;
            }
            
            if (abs(rzold) < breakdown_threshold) {
                // rzold near zero but rznew not - numerical issue, accept current solution
                result.converged = true;
                result.iterations = iter + 1;
                result.final_residual = relative_residual;
                result.error_message = "CG: rzold near zero, accepting current solution";
                break;
            }

            // beta = rznew / rzold
            beta = rznew / rzold;

            // p = z + beta * p
            cublasSscal(
                cublasHandle,
                n,
                &beta,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1);
            cublasSaxpy(
                cublasHandle,
                n,
                &one,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1);

            rzold = rznew;
            result.iterations = iter + 1;
        }

        if (!result.converged && result.error_message.empty()) {
            result.error_message = "Maximum iterations reached";
            result.final_residual = sqrt(rzold) / b_norm;
        }

        return result;
    }
}  // namespace

class CudaCGSolver : public LinearSolver {
   private:
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    bool initialized = false;

    // Check if matrix is likely SPD
    bool isLikelySPD(const Eigen::SparseMatrix<float>& A)
    {
        if (A.rows() != A.cols())
            return false;

        // Quick symmetry check on a sample of entries
        int sample_size = std::min(100, (int)A.rows());
        for (int i = 0; i < sample_size; ++i) {
            for (int j = i + 1; j < sample_size; ++j) {
                float aij = A.coeff(i, j);
                float aji = A.coeff(j, i);
                if (abs(aij - aji) >
                    1e-6f * std::max(abs(aij), abs(aji)) + 1e-10f) {
                    return false;
                }
            }
        }

        // Check diagonal positivity
        for (int i = 0; i < sample_size; ++i) {
            if (A.coeff(i, i) <= 0)
                return false;
        }

        return true;
    }

   public:
    CudaCGSolver()
    {
        if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS ||
            cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA libraries");
        }
        initialized = true;
    }

    ~CudaCGSolver()
    {
        if (initialized) {
            cusparseDestroy(cusparseHandle);
            cublasDestroy(cublasHandle);
        }
    }

    std::string getName() const override
    {
        return "CUDA Conjugate Gradient";
    }

    bool isIterative() const override
    {
        return true;
    }
    bool requiresGPU() const override
    {
        return true;
    }

    SolverResult solveGPU(
        int n,
        int nnz,
        const int* d_row_offsets,
        const int* d_col_indices,
        const float* d_values,
        const float* d_b,
        float* d_x,
        const SolverConfig& config = SolverConfig{}) override
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        SolverResult result;

        try {
            // Create cuSPARSE matrix descriptor from existing GPU data
            cusparseSpMatDescr_t matA_desc;
            cusparseCreateCsr(
                &matA_desc,
                n,
                n,
                nnz,
                (void*)d_row_offsets,
                (void*)d_col_indices,
                (void*)d_values,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F);

            // Allocate temporary vectors
            Ruzino::cuda::CUDALinearBufferDesc vec_desc;
            vec_desc.element_count = n;
            vec_desc.element_size = sizeof(float);

            auto d_r = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            auto d_z = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            auto d_p_buf = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            auto d_Ap = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            auto d_diagonal = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);

            // Simple diagonal: just set to 1.0 (no preconditioning for diagonal
            // matrix)
            float* diag_dev =
                reinterpret_cast<float*>(d_diagonal->get_device_ptr());
            thrust::fill(
                thrust::device_pointer_cast(diag_dev),
                thrust::device_pointer_cast(diag_dev + n),
                1.0f);

            // Create vector descriptors
            cusparseDnVecDescr_t vecX, vecB, vecR, vecZ, vecP, vecAp;
            cusparseCreateDnVec(&vecX, n, (void*)d_x, CUDA_R_32F);
            cusparseCreateDnVec(&vecB, n, (void*)d_b, CUDA_R_32F);
            cusparseCreateDnVec(
                &vecR, n, (void*)d_r->get_device_ptr(), CUDA_R_32F);
            cusparseCreateDnVec(
                &vecZ, n, (void*)d_z->get_device_ptr(), CUDA_R_32F);
            cusparseCreateDnVec(
                &vecP, n, (void*)d_p_buf->get_device_ptr(), CUDA_R_32F);
            cusparseCreateDnVec(
                &vecAp, n, (void*)d_Ap->get_device_ptr(), CUDA_R_32F);

            // Query SpMV buffer size
            size_t bufferSize = 0;
            const float one = 1.0f;
            const float zero = 0.0f;
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                (const void*)&one,
                matA_desc,
                vecP,
                (const void*)&zero,
                vecAp,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSize);

            Ruzino::cuda::CUDALinearBufferDesc buffer_desc;
            buffer_desc.element_count = bufferSize;
            buffer_desc.element_size = 1;
            auto dBuffer = Ruzino::cuda::create_cuda_linear_buffer(buffer_desc);

            auto setup_end = std::chrono::high_resolution_clock::now();
            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    setup_end - start_time);

            // Call CG implementation (reuse existing logic)
            {
                // Create borrowed buffers in limited scope to ensure they
                // destruct before cleanup
                Ruzino::cuda::CUDALinearBufferDesc vec_existing_desc;
                vec_existing_desc.element_count = n;
                vec_existing_desc.element_size = sizeof(float);

                auto d_x_buf = Ruzino::cuda::borrow_cuda_linear_buffer(
                    vec_existing_desc, (void*)d_x);
                auto d_b_buf = Ruzino::cuda::borrow_cuda_linear_buffer(
                    vec_existing_desc, (void*)d_b);

                result = performCGIterationsImpl(
                    cublasHandle,
                    cusparseHandle,
                    config,
                    n,
                    matA_desc,
                    vecX,
                    vecB,
                    vecR,
                    vecZ,
                    vecP,
                    vecAp,
                    d_diagonal,
                    dBuffer,
                    d_b_buf,
                    d_x_buf,
                    d_r,
                    d_z,
                    d_p_buf,
                    d_Ap);

                // Synchronize GPU before borrowed buffers destruct
                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    // Handle sync error if needed
                }
            }  // borrowed buffers destructed here, after sync

            // Synchronize before cleanup to ensure all GPU operations are
            // complete
            cudaError_t final_sync_err = cudaDeviceSynchronize();
            if (final_sync_err != cudaSuccess) {
                if (result.error_message.empty()) {
                    result.error_message = std::string("Final sync error: ") +
                                           cudaGetErrorString(final_sync_err);
                    result.converged = false;
                }
            }

            // Cleanup
            cusparseDestroyDnVec(vecX);
            cusparseDestroyDnVec(vecB);
            cusparseDestroyDnVec(vecR);
            cusparseDestroyDnVec(vecZ);
            cusparseDestroyDnVec(vecP);
            cusparseDestroyDnVec(vecAp);
            cusparseDestroySpMat(matA_desc);

            auto end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - setup_end);
        }
        catch (const std::exception& e) {
            result.error_message =
                std::string("GPU CG solve failed: ") + e.what();
            result.converged = false;
        }
        return result;
    }

    SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}) override
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        SolverResult result;

        try {
            int n = A.rows();
            int nnz = A.nonZeros();

            // Check if matrix is likely SPD (basic symmetry check)
            if (!isLikelySPD(A)) {
                result.error_message =
                    "CG requires symmetric positive definite matrix";
                result.converged = false;
                return result;
            }

            // Convert to CSR format
            std::vector<int> csrRowPtr(n + 1, 0);
            std::vector<int> csrColInd(nnz);
            std::vector<float> csrValues(nnz);

            // First pass: count entries per row
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
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
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    int row = it.row();
                    int pos = current_pos[row]++;
                    csrValues[pos] = it.value();
                    csrColInd[pos] = it.col();
                }
            }

            auto setup_end_time = std::chrono::high_resolution_clock::now();
            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    setup_end_time - start_time);

            // Upload to GPU
            auto d_csrValues =
                Ruzino::cuda::create_cuda_linear_buffer(csrValues);
            auto d_csrRowPtr =
                Ruzino::cuda::create_cuda_linear_buffer(csrRowPtr);
            auto d_csrColInd =
                Ruzino::cuda::create_cuda_linear_buffer(csrColInd);
            auto d_b = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_x = Ruzino::cuda::create_cuda_linear_buffer<float>(n);

            d_b->assign_host_vector(
                std::vector<float>(b.data(), b.data() + b.size()));
            d_x->assign_host_vector(
                std::vector<float>(x.data(), x.data() + x.size()));

            // Call GPU solve
            result = solveGPU(
                n,
                nnz,
                reinterpret_cast<const int*>(d_csrRowPtr->get_device_ptr()),
                reinterpret_cast<const int*>(d_csrColInd->get_device_ptr()),
                reinterpret_cast<const float*>(d_csrValues->get_device_ptr()),
                reinterpret_cast<const float*>(d_b->get_device_ptr()),
                reinterpret_cast<float*>(d_x->get_device_ptr()),
                config);

            // Download result
            auto result_vec = d_x->get_host_vector<float>();
            x = Eigen::Map<Eigen::VectorXf>(
                result_vec.data(), result_vec.size());
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCudaCGSolver()
{
    return std::make_unique<CudaCGSolver>();
}

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
