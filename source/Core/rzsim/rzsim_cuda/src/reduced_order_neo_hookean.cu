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
// Compute reduced gradient using cuBLAS: grad_q = J^T * grad_x
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

    // Use cuBLAS gemv for matrix-vector multiply: y = J^T * x
    // J is row-major [full_dof, reduced_dof] with ld=reduced_dof
    // In column-major view, this is J^T[reduced_dof, full_dof] with
    // ld=reduced_dof
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuBLAS handle: %d\n", status);
        return;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // cublasSgemv: y = alpha * op(A) * x + beta * y
    // A = J^T (column-major view of row-major J)
    // Shape: [reduced_dof, full_dof], lda = reduced_dof
    status = cublasSgemv(
        handle,
        CUBLAS_OP_N,  // No transpose (already have J^T in col-major view)
        reduced_dof,  // m: rows of A
        full_dof,     // n: cols of A
        &alpha,
        jacobian_ptr,  // A: row-major J = col-major J^T
        reduced_dof,   // lda: leading dimension
        grad_x_ptr,    // x
        1,             // incx
        &beta,
        grad_q_ptr,  // y
        1);          // incy

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] cublasSgemv failed: %d\n", status);
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

    // Use cuBLAS gemv: y = -J^T * x (alpha = -1)
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] Failed to create cuBLAS handle: %d\n", status);
        return;
    }

    float alpha = -1.0f;  // Negative to get -J^T * grad_x
    float beta = 0.0f;

    status = cublasSgemv(
        handle,
        CUBLAS_OP_N,
        reduced_dof,
        full_dof,
        &alpha,
        jacobian_ptr,
        reduced_dof,
        grad_x_ptr,
        1,
        &beta,
        neg_grad_q_ptr,
        1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[ReducedOrder] cublasSgemv failed: %d\n", status);
    }

    cublasDestroy(handle);
}

// ============================================================================
// Kernel: Map reduced velocities to full space (v_full = J * q_dot)
// ============================================================================

__global__ void map_reduced_to_full_velocities_kernel(
    const float* jacobian,  // [full_dof × reduced_dof] row-major
    const float* q_dot,     // [reduced_dof]
    float* v_full,          // [full_dof]
    int full_dof,
    int reduced_dof)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= full_dof)
        return;

    // Compute v_full[i] = sum_j J[i,j] * q_dot[j]
    // Row-major: J[i,j] = jacobian[i * reduced_dof + j]
    float sum = 0.0f;
    for (int j = 0; j < reduced_dof; ++j) {
        sum += jacobian[i * reduced_dof + j] * q_dot[j];
    }
    v_full[i] = sum;
}

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

    // Launch kernel: v_full = J * q_dot
    int threads_per_block = 256;
    int num_blocks = (full_dof + threads_per_block - 1) / threads_per_block;

    map_reduced_to_full_velocities_kernel<<<num_blocks, threads_per_block>>>(
        jacobian_ptr, q_dot_ptr, v_full_ptr, full_dof, reduced_dof);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] map_reduced_to_full_velocities kernel failed: %s\n",
            cudaGetErrorString(err));
    }
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
    // Jacobian is stored in ROW-MAJOR format: J[i,j] = jacobian[i*reduced_dof +
    // j]
    cusparseDnMatDescr_t matJ_desc;
    status = cusparseCreateDnMat(
        &matJ_desc,
        full_dof,     // rows
        reduced_dof,  // cols
        reduced_dof,  // leading dimension (row-major stride)
        jacobian_ptr,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW);  // row-major storage

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
    // temp will also be in ROW-MAJOR format to match the output of SpMM
    cusparseDnMatDescr_t matTemp_desc;
    status = cusparseCreateDnMat(
        &matTemp_desc,
        full_dof,     // rows
        reduced_dof,  // cols
        reduced_dof,  // leading dimension (row-major stride)
        temp_buffer_ptr,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW);  // row-major storage

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

    // cuBLAS computes: C = alpha * op(A) * op(B) + beta * C
    // We use: H_q = J^T_col * temp^T_col^T = J^T * temp
    // A = J (row-major [f,r] → column-major J^T [r,f])
    // B = temp (row-major [f,r] → column-major temp^T [r,f])
    // op(A) = A (no transpose needed, already have J^T)
    // op(B) = B^T (transpose to get temp from temp^T)
    cublas_status = cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,  // A = J^T (no transpose needed)
        CUBLAS_OP_T,  // B = temp^T, use transpose to get temp
        reduced_dof,  // m: rows of A = rows of J^T
        reduced_dof,  // n: cols of B^T = cols of temp
        full_dof,     // k: cols of A = rows of B^T
        &alpha,
        jacobian_ptr,     // A: J row-major [f,r] = J^T col-major [r,f]
        reduced_dof,      // lda: leading dimension (row stride in row-major J)
        temp_buffer_ptr,  // B: temp row-major [f,r] = temp^T col-major [r,f]
        reduced_dof,  // ldb: leading dimension (row stride in row-major temp)
        &beta,
        H_q_ptr,       // C: H_q row-major [r,r]
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
// Compute J^T * J (Gram matrix)
// ============================================================================

__global__ void compute_gram_matrix_kernel(
    const float* J,  // [full_dof × reduced_dof] row-major
    float* JtJ,      // [reduced_dof × reduced_dof] row-major output
    int full_dof,
    int reduced_dof)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= reduced_dof || col >= reduced_dof)
        return;

    // Compute (J^T * J)[row, col] = sum_k J[k, row] * J[k, col]
    // Row-major: J[k, row] = J[k * reduced_dof + row]
    float sum = 0.0f;
    for (int k = 0; k < full_dof; ++k) {
        sum += J[k * reduced_dof + row] * J[k * reduced_dof + col];
    }

    JtJ[row * reduced_dof + col] = sum;
}

void compute_jacobian_gram_matrix_gpu(
    cuda::CUDALinearBufferHandle jacobian,
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle JtJ)
{
    int full_dof = num_particles * 3;
    int reduced_dof = num_basis * 12;

    const float* J_ptr = jacobian->get_device_ptr<float>();
    float* JtJ_ptr = JtJ->get_device_ptr<float>();

    // Launch kernel to compute J^T * J
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (reduced_dof + threads_per_block.x - 1) / threads_per_block.x,
        (reduced_dof + threads_per_block.y - 1) / threads_per_block.y);

    compute_gram_matrix_kernel<<<num_blocks, threads_per_block>>>(
        J_ptr, JtJ_ptr, full_dof, reduced_dof);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] CUDA error in compute_jacobian_gram_matrix_gpu: "
            "%s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Apply Dirichlet BC to velocities
// ============================================================================

__global__ void apply_bc_to_velocities_kernel(
    const int* bc_dofs,  // [num_bc_dofs] - DOF indices
    int num_bc_dofs,
    glm::vec3* velocities,  // [num_particles]
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bc_dofs)
        return;

    int dof = bc_dofs[idx];
    int vertex = dof / 3;
    int component = dof % 3;

    if (vertex < num_particles) {
        // Access glm::vec3 components using pointer arithmetic
        float* vel_ptr = reinterpret_cast<float*>(&velocities[vertex]);
        vel_ptr[component] = 0.0f;
    }
}

void apply_dirichlet_bc_to_velocities_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle velocities,
    int num_particles)
{
    const int* bc_dofs_ptr = bc_dofs->get_device_ptr<int>();
    glm::vec3* velocities_ptr = velocities->get_device_ptr<glm::vec3>();

    int threads_per_block = 256;
    int num_blocks = (num_bc_dofs + threads_per_block - 1) / threads_per_block;

    apply_bc_to_velocities_kernel<<<num_blocks, threads_per_block>>>(
        bc_dofs_ptr, num_bc_dofs, velocities_ptr, num_particles);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] apply_bc_to_velocities kernel failed: %s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Apply Dirichlet BC to positions (set to rest pose)
// ============================================================================

__global__ void apply_bc_to_positions_kernel(
    const int* bc_dofs,  // [num_bc_dofs] - DOF indices
    int num_bc_dofs,
    float* positions,             // [num_particles * 3]
    const float* rest_positions,  // [num_particles * 3]
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bc_dofs)
        return;

    int dof = bc_dofs[idx];
    if (dof < num_particles * 3) {
        positions[dof] = rest_positions[dof];
    }
}

void apply_bc_to_positions_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle rest_positions,
    int num_particles)
{
    const int* bc_dofs_ptr = bc_dofs->get_device_ptr<int>();
    float* positions_ptr = positions->get_device_ptr<float>();
    const float* rest_positions_ptr = rest_positions->get_device_ptr<float>();

    int threads_per_block = 256;
    int num_blocks = (num_bc_dofs + threads_per_block - 1) / threads_per_block;

    apply_bc_to_positions_kernel<<<num_blocks, threads_per_block>>>(
        bc_dofs_ptr,
        num_bc_dofs,
        positions_ptr,
        rest_positions_ptr,
        num_particles);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] apply_bc_to_positions kernel failed: %s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Compute barrier energy for boundary conditions
// E_barrier = -k * sum_bc log(1 - ||x - x_rest||^2 / d^2)
// ============================================================================

__global__ void compute_barrier_energy_kernel(
    const int* bc_dofs,           // [num_bc_dofs]
    int num_bc_dofs,
    const float* positions,       // [num_particles * 3]
    const float* rest_positions,  // [num_particles * 3]
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    float* energy_buffer)  // [num_bc_vertices] output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process every 3 DOFs (one vertex)
    int vertex_idx = idx * 3;
    if (vertex_idx >= num_bc_dofs)
        return;
    
    // Get the vertex index from the first DOF of this vertex
    int dof = bc_dofs[vertex_idx];
    int vertex = dof / 3;
    
    if (vertex >= num_particles)
        return;
    
    // Compute squared distance
    float dx = positions[vertex * 3 + 0] - rest_positions[vertex * 3 + 0];
    float dy = positions[vertex * 3 + 1] - rest_positions[vertex * 3 + 1];
    float dz = positions[vertex * 3 + 2] - rest_positions[vertex * 3 + 2];
    float dist_sq = dx * dx + dy * dy + dz * dz;
    
    float d_sq = allowed_width * allowed_width;
    float ratio = dist_sq / d_sq;
    
    // Clamp to prevent log(0) or log(negative)
    if (ratio >= 0.99f) {
        ratio = 0.99f;
    }
    
    // E = -k * log(1 - ratio)
    energy_buffer[idx] = -barrier_stiffness * logf(1.0f - ratio);
}

float compute_barrier_energy_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle rest_positions,
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    cuda::CUDALinearBufferHandle energy_buffer)
{
    if (num_bc_dofs == 0)
        return 0.0f;
    
    const int* bc_dofs_ptr = bc_dofs->get_device_ptr<int>();
    const float* positions_ptr = positions->get_device_ptr<float>();
    const float* rest_positions_ptr = rest_positions->get_device_ptr<float>();
    float* energy_buffer_ptr = energy_buffer->get_device_ptr<float>();
    
    int num_bc_vertices = num_bc_dofs / 3;
    int threads_per_block = 256;
    int num_blocks = (num_bc_vertices + threads_per_block - 1) / threads_per_block;
    
    compute_barrier_energy_kernel<<<num_blocks, threads_per_block>>>(
        bc_dofs_ptr,
        num_bc_dofs,
        positions_ptr,
        rest_positions_ptr,
        barrier_stiffness,
        allowed_width,
        num_particles,
        energy_buffer_ptr);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] compute_barrier_energy kernel failed: %s\n",
            cudaGetErrorString(err));
        return 0.0f;
    }
    
    // Sum up energies
    auto energy_vec = energy_buffer->get_host_vector<float>();
    float total_energy = 0.0f;
    for (int i = 0; i < num_bc_vertices; ++i) {
        total_energy += energy_vec[i];
    }
    
    return total_energy;
}

// ============================================================================
// Kernel: Add barrier gradient
// grad += k * 2 * (x - x_rest) / (d^2 - ||x - x_rest||^2)
// ============================================================================

__global__ void add_barrier_gradient_kernel(
    const int* bc_dofs,           // [num_bc_dofs]
    int num_bc_dofs,
    const float* positions,       // [num_particles * 3]
    const float* rest_positions,  // [num_particles * 3]
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    float* gradient)  // [num_particles * 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bc_dofs)
        return;
    
    int dof = bc_dofs[idx];
    int vertex = dof / 3;
    int component = dof % 3;
    
    if (vertex >= num_particles)
        return;
    
    // Compute squared distance
    float dx = positions[vertex * 3 + 0] - rest_positions[vertex * 3 + 0];
    float dy = positions[vertex * 3 + 1] - rest_positions[vertex * 3 + 1];
    float dz = positions[vertex * 3 + 2] - rest_positions[vertex * 3 + 2];
    float dist_sq = dx * dx + dy * dy + dz * dz;
    
    float d_sq = allowed_width * allowed_width;
    float denominator = d_sq - dist_sq;
    
    // Clamp to prevent division by zero
    if (denominator < 0.01f * d_sq) {
        denominator = 0.01f * d_sq;
    }
    
    // grad[dof] += k * 2 * diff[component] / denominator
    float diff_component;
    if (component == 0) diff_component = dx;
    else if (component == 1) diff_component = dy;
    else diff_component = dz;
    
    gradient[dof] += barrier_stiffness * 2.0f * diff_component / denominator;
}

void add_barrier_gradient_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle rest_positions,
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    cuda::CUDALinearBufferHandle gradient)
{
    if (num_bc_dofs == 0)
        return;
    
    const int* bc_dofs_ptr = bc_dofs->get_device_ptr<int>();
    const float* positions_ptr = positions->get_device_ptr<float>();
    const float* rest_positions_ptr = rest_positions->get_device_ptr<float>();
    float* gradient_ptr = gradient->get_device_ptr<float>();
    
    int threads_per_block = 256;
    int num_blocks = (num_bc_dofs + threads_per_block - 1) / threads_per_block;
    
    add_barrier_gradient_kernel<<<num_blocks, threads_per_block>>>(
        bc_dofs_ptr,
        num_bc_dofs,
        positions_ptr,
        rest_positions_ptr,
        barrier_stiffness,
        allowed_width,
        num_particles,
        gradient_ptr);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] add_barrier_gradient kernel failed: %s\n",
            cudaGetErrorString(err));
    }
}

// ============================================================================
// Kernel: Add barrier Hessian diagonal contributions
// H_ii += k * 2 * (d^2 - ||diff||^2 + 2*diff_i^2) / (d^2 - ||diff||^2)^2
// ============================================================================

__global__ void add_barrier_hessian_diagonal_kernel(
    const int* bc_dofs,           // [num_bc_dofs]
    int num_bc_dofs,
    const float* positions,       // [num_particles * 3]
    const float* rest_positions,  // [num_particles * 3]
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    float* hessian_diagonal)  // [num_particles * 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bc_dofs)
        return;
    
    int dof = bc_dofs[idx];
    int vertex = dof / 3;
    int component = dof % 3;
    
    if (vertex >= num_particles)
        return;
    
    // Compute squared distance
    float dx = positions[vertex * 3 + 0] - rest_positions[vertex * 3 + 0];
    float dy = positions[vertex * 3 + 1] - rest_positions[vertex * 3 + 1];
    float dz = positions[vertex * 3 + 2] - rest_positions[vertex * 3 + 2];
    float dist_sq = dx * dx + dy * dy + dz * dz;
    
    float d_sq = allowed_width * allowed_width;
    float denominator = d_sq - dist_sq;
    
    // Clamp to prevent division by zero
    if (denominator < 0.01f * d_sq) {
        denominator = 0.01f * d_sq;
    }
    
    float diff_component;
    if (component == 0) diff_component = dx;
    else if (component == 1) diff_component = dy;
    else diff_component = dz;
    
    float diff_sq = diff_component * diff_component;
    
    // H_ii += k * 2 * (d^2 - dist_sq + 2*diff_i^2) / (d^2 - dist_sq)^2
    float numerator = denominator + 2.0f * diff_sq;
    float hessian_contrib = barrier_stiffness * 2.0f * numerator / (denominator * denominator);
    
    hessian_diagonal[dof] += hessian_contrib;
}

void add_barrier_hessian_diagonal_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle rest_positions,
    float barrier_stiffness,
    float allowed_width,
    int num_particles,
    cuda::CUDALinearBufferHandle hessian_diagonal)
{
    if (num_bc_dofs == 0)
        return;
    
    const int* bc_dofs_ptr = bc_dofs->get_device_ptr<int>();
    const float* positions_ptr = positions->get_device_ptr<float>();
    const float* rest_positions_ptr = rest_positions->get_device_ptr<float>();
    float* hessian_diagonal_ptr = hessian_diagonal->get_device_ptr<float>();
    
    int threads_per_block = 256;
    int num_blocks = (num_bc_dofs + threads_per_block - 1) / threads_per_block;
    
    add_barrier_hessian_diagonal_kernel<<<num_blocks, threads_per_block>>>(
        bc_dofs_ptr,
        num_bc_dofs,
        positions_ptr,
        rest_positions_ptr,
        barrier_stiffness,
        allowed_width,
        num_particles,
        hessian_diagonal_ptr);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(
            "[ReducedOrder] add_barrier_hessian_diagonal kernel failed: %s\n",
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
