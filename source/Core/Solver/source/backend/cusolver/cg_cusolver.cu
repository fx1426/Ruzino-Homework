#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <iostream>

RUZINO_NAMESPACE_OPEN_SCOPE

namespace Solver {

namespace {
    // cuSOLVER QR 直接法求解器实现
    SolverResult performCuSolverQRSolveImpl(
        cusolverSpHandle_t cusolverHandle,
        cusparseHandle_t cusparseHandle,
        const SolverConfig& config,
        int n,
        int nnz,
        Ruzino::cuda::CUDALinearBufferHandle d_csrVal,
        Ruzino::cuda::CUDALinearBufferHandle d_csrRowPtr,
        Ruzino::cuda::CUDALinearBufferHandle d_csrColInd,
        Ruzino::cuda::CUDALinearBufferHandle d_b,
        Ruzino::cuda::CUDALinearBufferHandle d_x)
    {
        SolverResult result;

        // Create cuSOLVER matrix descriptor
        cusparseMatDescr_t descrA;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

        int singularity = 0;
        
        // Call cuSOLVER QR solver
        cusolverStatus_t status = cusolverSpScsrlsvqr(
            cusolverHandle,
            n,
            nnz,
            descrA,
            reinterpret_cast<const float*>(d_csrVal->get_device_ptr()),
            reinterpret_cast<const int*>(d_csrRowPtr->get_device_ptr()),
            reinterpret_cast<const int*>(d_csrColInd->get_device_ptr()),
            reinterpret_cast<const float*>(d_b->get_device_ptr()),
            config.tolerance,
            0,  // no reordering
            reinterpret_cast<float*>(d_x->get_device_ptr()),
            &singularity);

        if (status != CUSOLVER_STATUS_SUCCESS) {
            result.converged = false;
            result.error_message = "cuSOLVER QR failed with status " + std::to_string(status);
            cusparseDestroyMatDescr(descrA);
            return result;
        }

        if (singularity >= 0) {
            result.converged = false;
            result.error_message = "Matrix is singular at column " + std::to_string(singularity);
            if (config.verbose) {
                std::cout << result.error_message << std::endl;
            }
        } else {
            result.converged = true;
            result.iterations = 1;  // Direct solver, single "iteration"
            result.final_residual = 0.0f;  // Direct solver
            
            if (config.verbose) {
                std::cout << "cuSOLVER QR direct solve completed successfully" << std::endl;
            }
        }

        cusparseDestroyMatDescr(descrA);
        return result;
    }
}  // namespace

class CuSolverQRSolver : public LinearSolver {
   private:
    cusolverSpHandle_t cusolverHandle;
    cusparseHandle_t cusparseHandle;
    bool initialized = false;

   public:
    CuSolverQRSolver()
    {
        if (cusolverSpCreate(&cusolverHandle) != CUSOLVER_STATUS_SUCCESS ||
            cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSOLVER/cuSPARSE handles");
        }
        initialized = true;
    }

    ~CuSolverQRSolver()
    {
        if (initialized) {
            cusolverSpDestroy(cusolverHandle);
            cusparseDestroy(cusparseHandle);
        }
    }

    std::string getName() const override
    {
        return "cuSOLVER QR (Direct)";
    }

    bool isIterative() const override
    {
        return false;  // 直接法!
    }

    bool requiresGPU() const override
    {
        return true;
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

            // Convert to CSR format
            Eigen::SparseMatrix<float, Eigen::RowMajor> A_csr = A;
            
            std::vector<float> csrVal(nnz);
            std::vector<int> csrRowPtr(n + 1);
            std::vector<int> csrColInd(nnz);

            // Copy CSR data
            int idx = 0;
            csrRowPtr[0] = 0;
            for (int i = 0; i < n; ++i) {
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A_csr, i); it; ++it) {
                    csrVal[idx] = it.value();
                    csrColInd[idx] = it.col();
                    idx++;
                }
                csrRowPtr[i + 1] = idx;
            }
            
            // Debug output
            if (config.verbose) {
                std::cout << "Matrix size: " << n << "x" << n << std::endl;
                std::cout << "Total nnz: " << nnz << std::endl;
                std::cout << "First few values: ";
                for (int i = 0; i < std::min(5, nnz); ++i) {
                    std::cout << csrVal[i] << " ";
                }
                std::cout << std::endl;
            }

            // Allocate device memory using proper API
            Ruzino::cuda::CUDALinearBufferDesc val_desc, rowptr_desc, colind_desc, vec_desc;
            val_desc.element_count = nnz;
            val_desc.element_size = sizeof(float);
            rowptr_desc.element_count = n + 1;
            rowptr_desc.element_size = sizeof(int);
            colind_desc.element_count = nnz;
            colind_desc.element_size = sizeof(int);
            vec_desc.element_count = n;
            vec_desc.element_size = sizeof(float);

            auto d_csrVal = Ruzino::cuda::create_cuda_linear_buffer(val_desc);
            auto d_csrRowPtr = Ruzino::cuda::create_cuda_linear_buffer(rowptr_desc);
            auto d_csrColInd = Ruzino::cuda::create_cuda_linear_buffer(colind_desc);
            auto d_b = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            auto d_x = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);

            // Copy to device
            cudaMemcpy(
                (void*)d_csrVal->get_device_ptr(),
                csrVal.data(),
                nnz * sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_csrRowPtr->get_device_ptr(),
                csrRowPtr.data(),
                (n + 1) * sizeof(int),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_csrColInd->get_device_ptr(),
                csrColInd.data(),
                nnz * sizeof(int),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_b->get_device_ptr(),
                b.data(),
                n * sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_x->get_device_ptr(),
                x.data(),
                n * sizeof(float),
                cudaMemcpyHostToDevice);

            // Call cuSOLVER QR direct solver
            result = performCuSolverQRSolveImpl(
                cusolverHandle,
                cusparseHandle,
                config,
                n,
                nnz,
                d_csrVal,
                d_csrRowPtr,
                d_csrColInd,
                d_b,
                d_x);

            // Copy result back
            cudaMemcpy(
                x.data(),
                (void*)d_x->get_device_ptr(),
                n * sizeof(float),
                cudaMemcpyDeviceToHost);

        } catch (const std::exception& e) {
            result.converged = false;
            result.error_message = std::string("cuSOLVER Cholesky error: ") + e.what();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

        return result;
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCuSolverCGSolver()
{
    return std::make_unique<CuSolverQRSolver>();
}

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
