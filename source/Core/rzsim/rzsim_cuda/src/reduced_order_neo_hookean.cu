#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <device_launch_parameters.h>

#include <Eigen/Eigen>
#include <cmath>
#include <cstdio>  // For printf as error reporting
#include <vector>

#include "RHI/internal/cuda_extension.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/reduced_order_neo_hookean.cuh"

// Include glm after CUDA headers to avoid conflicts
#ifndef __CUDACC__
#include <glm/glm.hpp>
#else
// For CUDA device code, we need basic vec3 definition
namespace glm {
struct vec3 {
    float x, y, z;
};
}  // namespace glm
#endif

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// ============================================================================
// Kernel: Build reduced order data
// ============================================================================

ReducedOrderData build_reduced_order_data_gpu(
    const void* basis_data,
    const void* rest_positions_data)
{
    // Cast back to actual types
    const auto& basis =
        *reinterpret_cast<const std::vector<Eigen::VectorXf>*>(basis_data);
    const auto& rest_positions =
        *reinterpret_cast<const std::vector<glm::vec3>*>(rest_positions_data);

    ReducedOrderData ro_data;
    ro_data.num_basis = basis.size();
    ro_data.num_particles = rest_positions.size();

    if (ro_data.num_basis == 0 || ro_data.num_particles == 0) {
        printf("[ReducedOrder] Empty basis or rest positions\n");
        return ro_data;
    }

    // Build basis weights matrix [num_particles, num_basis]
    // Each entry is the eigenvector weight for that vertex and basis
    std::vector<float> weights(ro_data.num_particles * ro_data.num_basis);

    for (int i = 0; i < ro_data.num_basis; ++i) {
        const auto& eigenvec = basis[i];
        if (eigenvec.size() != ro_data.num_particles) {
            printf(
                "[ReducedOrder] Basis %d size mismatch: %d vs %d\n",
                i,
                (int)eigenvec.size(),
                ro_data.num_particles);
            continue;
        }

        for (int v = 0; v < ro_data.num_particles; ++v) {
            weights[v * ro_data.num_basis + i] = eigenvec(v);
        }
    }

    ro_data.basis_weights = cuda::create_cuda_linear_buffer(weights);
    ro_data.rest_positions = cuda::create_cuda_linear_buffer(rest_positions);

    return ro_data;
}

// ============================================================================
// Kernel: Map reduced coordinates to full space
// x[v] = rest[v] + Σ_i weight[v,i] * (R_i * rest[v] + t_i - rest[v])
// ============================================================================

void map_reduced_to_full_gpu(
    cuda::CUDALinearBufferHandle q_reduced,
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle x_full)
{
    int num_particles = ro_data.num_particles;
    int num_basis = ro_data.num_basis;

    const float* q_reduced_ptr = q_reduced->get_device_ptr<float>();
    const float* basis_weights_ptr =
        ro_data.basis_weights->get_device_ptr<float>();
    const float* rest_positions_ptr =
        ro_data.rest_positions->get_device_ptr<float>();
    float* x_full_ptr = x_full->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "map_reduced_to_full", num_particles, [=] __device__(int v) {
            // Load rest position
            float rest_x = rest_positions_ptr[3 * v + 0];
            float rest_y = rest_positions_ptr[3 * v + 1];
            float rest_z = rest_positions_ptr[3 * v + 2];

            // Initialize with rest position
            float x = rest_x;
            float y = rest_y;
            float z = rest_z;

            // Add contributions from each basis
            for (int i = 0; i < num_basis; ++i) {
                float weight = basis_weights_ptr[v * num_basis + i];

                // Load affine transform parameters for basis i
                // Rotation matrix R (row-major, 3x3)
                float R00 = q_reduced_ptr[i * 12 + 0];
                float R01 = q_reduced_ptr[i * 12 + 1];
                float R02 = q_reduced_ptr[i * 12 + 2];
                float R10 = q_reduced_ptr[i * 12 + 3];
                float R11 = q_reduced_ptr[i * 12 + 4];
                float R12 = q_reduced_ptr[i * 12 + 5];
                float R20 = q_reduced_ptr[i * 12 + 6];
                float R21 = q_reduced_ptr[i * 12 + 7];
                float R22 = q_reduced_ptr[i * 12 + 8];

                // Translation t
                float tx = q_reduced_ptr[i * 12 + 9];
                float ty = q_reduced_ptr[i * 12 + 10];
                float tz = q_reduced_ptr[i * 12 + 11];

                // Apply affine transform: R * rest_pos + t
                float transformed_x =
                    R00 * rest_x + R01 * rest_y + R02 * rest_z + tx;
                float transformed_y =
                    R10 * rest_x + R11 * rest_y + R12 * rest_z + ty;
                float transformed_z =
                    R20 * rest_x + R21 * rest_y + R22 * rest_z + tz;

                // Add weighted contribution
                x += weight * (transformed_x - rest_x);
                y += weight * (transformed_y - rest_y);
                z += weight * (transformed_z - rest_z);
            }

            x_full_ptr[3 * v + 0] = x;
            x_full_ptr[3 * v + 1] = y;
            x_full_ptr[3 * v + 2] = z;
        });
}

// ============================================================================
// Kernel: Compute Jacobian matrix
// J[3v+d, 12i+p] = weight[v,i] * ∂(R_i*rest[v] + t_i)[d] / ∂q[12i+p]
// ============================================================================

void compute_jacobian_gpu(
    cuda::CUDALinearBufferHandle q_reduced,
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle jacobian)
{
    int num_particles = ro_data.num_particles;
    int num_basis = ro_data.num_basis;

    // Zero out jacobian first
    size_t jacobian_size = num_particles * 3 * num_basis * 12 * sizeof(float);
    cudaMemset(
        reinterpret_cast<void*>(jacobian->get_device_ptr()), 0, jacobian_size);

    const float* basis_weights_ptr =
        ro_data.basis_weights->get_device_ptr<float>();
    const float* rest_positions_ptr =
        ro_data.rest_positions->get_device_ptr<float>();
    float* jacobian_ptr = jacobian->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "compute_jacobian", num_particles, [=] __device__(int v) {
            float rest_x = rest_positions_ptr[3 * v + 0];
            float rest_y = rest_positions_ptr[3 * v + 1];
            float rest_z = rest_positions_ptr[3 * v + 2];

            // For each basis mode
            for (int i = 0; i < num_basis; ++i) {
                float weight = basis_weights_ptr[v * num_basis + i];

                // Derivatives w.r.t. rotation matrix elements
                // x_component: d(x)/d(R[row][col]) = weight * rest[col] if row
                // == 0 For x-component of vertex v
                int row_x = 3 * v + 0;
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 0] =
                    weight * rest_x;  // dR00
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 1] =
                    weight * rest_y;  // dR01
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 2] =
                    weight * rest_z;  // dR02
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 3] =
                    0.0f;  // dR10
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 4] =
                    0.0f;  // dR11
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 5] =
                    0.0f;  // dR12
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 6] =
                    0.0f;  // dR20
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 7] =
                    0.0f;  // dR21
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 8] =
                    0.0f;  // dR22
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 9] =
                    weight;  // dtx
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 10] =
                    0.0f;  // dty
                jacobian_ptr[row_x * (num_basis * 12) + i * 12 + 11] =
                    0.0f;  // dtz

                // For y-component
                int row_y = 3 * v + 1;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 0] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 1] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 2] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 3] =
                    weight * rest_x;  // dR10
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 4] =
                    weight * rest_y;  // dR11
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 5] =
                    weight * rest_z;  // dR12
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 6] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 7] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 8] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 9] = 0.0f;
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 10] =
                    weight;  // dty
                jacobian_ptr[row_y * (num_basis * 12) + i * 12 + 11] = 0.0f;

                // For z-component
                int row_z = 3 * v + 2;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 0] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 1] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 2] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 3] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 4] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 5] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 6] =
                    weight * rest_x;  // dR20
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 7] =
                    weight * rest_y;  // dR21
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 8] =
                    weight * rest_z;  // dR22
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 9] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 10] = 0.0f;
                jacobian_ptr[row_z * (num_basis * 12) + i * 12 + 11] =
                    weight;  // dtz
            }
        });
}

// ============================================================================
// Kernel: Compute reduced gradient (J^T * grad_x)
// ============================================================================

void compute_reduced_gradient_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle grad_x,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle grad_q)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* grad_x_ptr = grad_x->get_device_ptr<float>();
    float* grad_q_ptr = grad_q->get_device_ptr<float>();

    // Use cuBLAS for matrix-vector multiply: grad_q = J^T * grad_x
    // J is row-major [full_dof, reduced_dof]
    // For row-major matrix, J^T * x uses CUBLAS_OP_N
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuBLAS handle: %d\n", status);
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // grad_q[reduced_dof] = J^T[reduced_dof, full_dof] * grad_x[full_dof]
    status = cublasSgemv(
        handle,
        CUBLAS_OP_N,  // No transpose (for row-major J^T)
        reduced_dof,  // m: rows of op(A)
        full_dof,     // n: cols of op(A)
        &alpha,
        jacobian_ptr,  // A: J in row-major
        reduced_dof,   // lda: leading dimension
        grad_x_ptr,    // x
        1,             // incx
        &beta,
        grad_q_ptr,  // y
        1);          // incy

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] cublasSgemv failed in compute_reduced_gradient: "
            "%d\n",
            status);
    }

    cublasDestroy(handle);
}

void compute_reduced_neg_gradient_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle grad_x,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle neg_grad_q)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* grad_x_ptr = grad_x->get_device_ptr<float>();
    float* neg_grad_q_ptr = neg_grad_q->get_device_ptr<float>();

    // Use cuBLAS for matrix-vector multiply: neg_grad_q = -J^T * grad_x
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuBLAS handle: %d\n", status);
        return;
    }

    float alpha = -1.0f;  // Negative for negation
    float beta = 0.0f;

    // neg_grad_q[reduced_dof] = -J^T[reduced_dof, full_dof] * grad_x[full_dof]
    status = cublasSgemv(
        handle,
        CUBLAS_OP_N,  // No transpose (for row-major J^T)
        reduced_dof,  // m: rows of op(A)
        full_dof,     // n: cols of op(A)
        &alpha,
        jacobian_ptr,  // A: J in row-major
        reduced_dof,   // lda: leading dimension
        grad_x_ptr,    // x
        1,             // incx
        &beta,
        neg_grad_q_ptr,  // y
        1);              // incy

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] cublasSgemv failed in "
            "compute_reduced_neg_gradient: %d\n",
            status);
    }

    cublasDestroy(handle);
}

// ============================================================================
// Kernel: Map reduced velocities to full space (v_full = J * q_dot)
// ============================================================================

void map_reduced_velocities_to_full_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    cuda::CUDALinearBufferHandle q_dot,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle v_full)
{
    int reduced_dof = num_basis * 12;
    int full_dof = num_particles * 3;

    const float* jacobian_ptr = jacobian->get_device_ptr<float>();
    const float* q_dot_ptr = q_dot->get_device_ptr<float>();
    float* v_full_ptr = v_full->get_device_ptr<float>();

    // Use cuBLAS for matrix-vector multiply: v_full = J * q_dot
    // J is row-major [full_dof, reduced_dof]
    // For row-major matrix, J * x uses CUBLAS_OP_T
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuBLAS handle: %d\n", status);
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // v_full[full_dof] = J[full_dof, reduced_dof] * q_dot[reduced_dof]
    status = cublasSgemv(
        handle,
        CUBLAS_OP_T,  // Transpose (for row-major J)
        reduced_dof,  // m: cols of J (before transpose)
        full_dof,     // n: rows of J (before transpose)
        &alpha,
        jacobian_ptr,  // A: J in row-major
        reduced_dof,   // lda: leading dimension
        q_dot_ptr,     // x
        1,             // incx
        &beta,
        v_full_ptr,  // y
        1);          // incy

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] cublasSgemv failed in map_reduced_velocities: %d\n",
            status);
    }

    cublasDestroy(handle);
}

// ============================================================================
// Kernel: Sparse matrix-dense matrix multiply using cuSPARSE (CSR * dense)
// ============================================================================

void compute_reduced_hessian_gpu(
    const NeoHookeanCSRStructure& hessian_structure,
    cuda::CUDALinearBufferHandle hessian_values,
    cuda::CUDALinearBufferHandle jacobian,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle temp_buffer,
    cuda::CUDALinearBufferHandle H_q)
{
    int full_dof = num_particles * 3;
    int reduced_dof = num_basis * 12;

    // Create cuSPARSE handle
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t status = cusparseCreate(&cusparseHandle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuSPARSE handle: %d\n", status);
        return;
    }

    const int* row_offsets_ptr =
        hessian_structure.row_offsets->get_device_ptr<int>();
    const int* col_indices_ptr =
        hessian_structure.col_indices->get_device_ptr<int>();
    const float* hessian_values_ptr = hessian_values->get_device_ptr<float>();
    float* jacobian_ptr = jacobian->get_device_ptr<float>();
    float* temp_buffer_ptr = temp_buffer->get_device_ptr<float>();
    float* H_q_ptr = H_q->get_device_ptr<float>();

    // Step 1: temp = H_x * J using cuSPARSE SpMM
    // H_x is [full_dof, full_dof] sparse CSR
    // J is [full_dof, reduced_dof] dense (column-major for cuSPARSE)
    // temp is [full_dof, reduced_dof] dense (column-major)

    // Create sparse matrix descriptor for H_x (CSR format)
    cusparseSpMatDescr_t matH_desc;
    status = cusparseCreateCsr(
        &matH_desc,
        full_dof,               // rows
        full_dof,               // cols
        hessian_structure.nnz,  // nnz
        const_cast<int*>(row_offsets_ptr),
        const_cast<int*>(col_indices_ptr),
        const_cast<float*>(hessian_values_ptr),
        CUSPARSE_INDEX_32I,  // row offsets type
        CUSPARSE_INDEX_32I,  // col indices type
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);  // data type

    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] Failed to create sparse matrix descriptor: %d\n",
            status);
        cusparseDestroy(cusparseHandle);
        return;
    }

    // Create dense matrix descriptor for J [full_dof, reduced_dof]
    // (column-major)
    cusparseDnMatDescr_t matJ_desc;
    status = cusparseCreateDnMat(
        &matJ_desc,
        full_dof,     // rows
        reduced_dof,  // cols
        full_dof,     // leading dimension (column-major stride)
        jacobian_ptr,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL);  // column-major

    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] Failed to create Jacobian dense matrix descriptor: "
            "%d\n",
            status);
        cusparseDestroySpMat(matH_desc);
        cusparseDestroy(cusparseHandle);
        return;
    }

    // Create dense matrix descriptor for temp [full_dof, reduced_dof]
    // (column-major)
    cusparseDnMatDescr_t matTemp_desc;
    status = cusparseCreateDnMat(
        &matTemp_desc,
        full_dof,     // rows
        reduced_dof,  // cols
        full_dof,     // leading dimension
        temp_buffer_ptr,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL);  // column-major

    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] Failed to create temp dense matrix descriptor: "
            "%d\n",
            status);
        cusparseDestroyDnMat(matJ_desc);
        cusparseDestroySpMat(matH_desc);
        cusparseDestroy(cusparseHandle);
        return;
    }

    // Allocate workspace for SpMM
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;

    status = cusparseSpMM_bufferSize(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,  // H_x not transposed
        CUSPARSE_OPERATION_NON_TRANSPOSE,  // J not transposed
        &alpha,
        matH_desc,
        matJ_desc,
        &beta,
        matTemp_desc,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to get SpMM buffer size: %d\n", status);
        cusparseDestroyDnMat(matTemp_desc);
        cusparseDestroyDnMat(matJ_desc);
        cusparseDestroySpMat(matH_desc);
        cusparseDestroy(cusparseHandle);
        return;
    }

    void* dBuffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }

    // Perform SpMM: temp = H_x * J
    status = cusparseSpMM(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matH_desc,
        matJ_desc,
        &beta,
        matTemp_desc,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("[ReducedOrder] SpMM failed: %d\n", status);
    }

    if (dBuffer) {
        cudaFree(dBuffer);
    }

    // Step 2: H_q = J^T * temp using cuBLAS gemm
    // J^T is [reduced_dof, full_dof]
    // temp is [full_dof, reduced_dof]
    // H_q is [reduced_dof, reduced_dof]

    // Create cuBLAS handle for dense matrix multiply
    cublasHandle_t cublasHandle;
    cublasStatus_t cublas_status = cublasCreate(&cublasHandle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf(
            "[ReducedOrder] Failed to create cuBLAS handle: %d\n",
            cublas_status);
        cusparseDestroyDnMat(matTemp_desc);
        cusparseDestroyDnMat(matJ_desc);
        cusparseDestroySpMat(matH_desc);
        cusparseDestroy(cusparseHandle);
        return;
    }

    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
    // We want: H_q = J^T * temp
    // In column-major: H_q = J^T * temp
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
    // C, ldc) C[m,n] = op(A)[m,k] * op(B)[k,n] H_q[reduced_dof, reduced_dof] =
    // J^T[reduced_dof, full_dof] * temp[full_dof, reduced_dof]
    cublas_status = cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,  // J transposed
        CUBLAS_OP_N,  // temp not transposed
        reduced_dof,  // m: rows of J^T
        reduced_dof,  // n: cols of temp
        full_dof,     // k: cols of J^T = rows of temp
        &alpha,
        jacobian_ptr,     // J [full_dof, reduced_dof] in column-major
        full_dof,         // lda: leading dimension of J
        temp_buffer_ptr,  // temp [full_dof, reduced_dof] in column-major
        full_dof,         // ldb: leading dimension of temp
        &beta,
        H_q_ptr,       // H_q [reduced_dof, reduced_dof] in column-major
        reduced_dof);  // ldc: leading dimension of H_q

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] cublasSgemm failed: %d\n", cublas_status);
    }

    // Cleanup
    cublasDestroy(cublasHandle);
    cusparseDestroyDnMat(matTemp_desc);
    cusparseDestroyDnMat(matJ_desc);
    cusparseDestroySpMat(matH_desc);
    cusparseDestroy(cusparseHandle);
}

// ============================================================================
// Initialize reduced coordinates to identity (R=I, t=0 for each basis)
// ============================================================================

void initialize_reduced_coords_identity_gpu(
    int num_basis,
    cuda::CUDALinearBufferHandle q)
{
    int reduced_dof = num_basis * 12;
    std::vector<float> host_q(reduced_dof, 0.0f);

    // For each basis: set rotation to identity (R00=R11=R22=1), translation to
    // 0
    for (int i = 0; i < num_basis; ++i) {
        host_q[i * 12 + 0] = 1.0f;  // R00
        host_q[i * 12 + 4] = 1.0f;  // R11
        host_q[i * 12 + 8] = 1.0f;  // R22
        // Rest are already 0
    }

    cudaMemcpy(
        reinterpret_cast<void*>(q->get_device_ptr()),
        host_q.data(),
        reduced_dof * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] CUDA error in "
            "initialize_reduced_coords_identity_gpu: %s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Explicit step in reduced space
// ============================================================================

void explicit_step_reduced_gpu(
    cuda::CUDALinearBufferHandle q,
    cuda::CUDALinearBufferHandle q_dot,
    float dt,
    int num_basis,
    cuda::CUDALinearBufferHandle q_tilde)
{
    int reduced_dof = num_basis * 12;

    const float* q_ptr = q->get_device_ptr<float>();
    const float* q_dot_ptr = q_dot->get_device_ptr<float>();
    float* q_tilde_ptr = q_tilde->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "explicit_step_reduced", reduced_dof, [=] __device__(int idx) {
            q_tilde_ptr[idx] = q_ptr[idx] + dt * q_dot_ptr[idx];
        });
}

// ============================================================================
// Kernel: Update reduced velocities
// ============================================================================

void update_reduced_velocities_gpu(
    cuda::CUDALinearBufferHandle q_new,
    cuda::CUDALinearBufferHandle q_old,
    float dt,
    float damping,
    int num_basis,
    cuda::CUDALinearBufferHandle q_dot)
{
    int reduced_dof = num_basis * 12;

    const float* q_new_ptr = q_new->get_device_ptr<float>();
    const float* q_old_ptr = q_old->get_device_ptr<float>();
    float* q_dot_ptr = q_dot->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "update_reduced_velocities", reduced_dof, [=] __device__(int idx) {
            q_dot_ptr[idx] = (q_new_ptr[idx] - q_old_ptr[idx]) / dt * damping;
        });
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
