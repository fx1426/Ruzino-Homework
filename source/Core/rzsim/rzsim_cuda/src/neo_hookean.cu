#include <cusparse.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <Eigen/Dense>
#include <RHI/cuda.hpp>

#include "rzsim_cuda/neo_hookean.cuh"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// ============================================================================
// Basic Operations
// ============================================================================

void explicit_step_nh_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde)
{
    const float* x_ptr = x->get_device_ptr<float>();
    const float* v_ptr = v->get_device_ptr<float>();
    float* x_tilde_ptr = x_tilde->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "explicit_step_nh",
        num_particles * 3,
        [=] __device__(int i) { x_tilde_ptr[i] = x_ptr[i] + dt * v_ptr[i]; });
}

void setup_external_forces_nh_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext)
{
    float* f_ext_ptr = f_ext->get_device_ptr<float>();

    cuda::GPUParallelFor("setup_external_forces_nh", num_particles * 3, [=] __device__(int i) {
        f_ext_ptr[i] = (i % 3 == 2) ? mass * gravity : 0.0f;
    });
}

// ============================================================================
// Neo-Hookean Energy Model
// ============================================================================

// Compute deformation gradient F = Ds * Dm_inv
// Ds = [x1-x0, x2-x0, x3-x0] (3x3 matrix from current positions)
// Dm_inv is precomputed from rest positions
__device__ void compute_deformation_gradient(
    const float* x,
    const int* tet,
    const float* Dm_inv,
    Eigen::Matrix3f& F)
{
    // Get vertex positions
    Eigen::Vector3f x0(x[tet[0] * 3 + 0], x[tet[0] * 3 + 1], x[tet[0] * 3 + 2]);
    Eigen::Vector3f x1(x[tet[1] * 3 + 0], x[tet[1] * 3 + 1], x[tet[1] * 3 + 2]);
    Eigen::Vector3f x2(x[tet[2] * 3 + 0], x[tet[2] * 3 + 1], x[tet[2] * 3 + 2]);
    Eigen::Vector3f x3(x[tet[3] * 3 + 0], x[tet[3] * 3 + 1], x[tet[3] * 3 + 2]);

    // Compute Ds = [x1-x0, x2-x0, x3-x0]
    Eigen::Matrix3f Ds;
    Ds.col(0) = x1 - x0;
    Ds.col(1) = x2 - x0;
    Ds.col(2) = x3 - x0;

    // Load Dm_inv (stored in column-major order)
    Eigen::Matrix3f Dm_inv_mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_mat(i, j) = Dm_inv[j * 3 + i];  // Column-major
        }
    }

    // F = Ds * Dm_inv
    F = Ds * Dm_inv_mat;
}

// Neo-Hookean energy density: Ψ = μ/2 * (tr(F^T F) - 3) - μ log(J) + λ/2 * log(J)^2
__device__ float neo_hookean_energy_density(
    const Eigen::Matrix3f& F,
    float mu,
    float lambda)
{
    float J = F.determinant();
    if (J <= 0.0f) {
        // Inverted element - use large penalty
        return 1e10f;
    }

    Eigen::Matrix3f FtF = F.transpose() * F;
    float I1 = FtF.trace();  // tr(F^T F)
    float log_J = logf(J);

    float psi = 0.5f * mu * (I1 - 3.0f) - mu * log_J + 0.5f * lambda * log_J * log_J;
    return psi;
}

// First Piola-Kirchhoff stress: P = ∂Ψ/∂F = μ(F - F^-T) + λ log(J) F^-T
__device__ void compute_pk1_stress(
    const Eigen::Matrix3f& F,
    float mu,
    float lambda,
    Eigen::Matrix3f& P)
{
    float J = F.determinant();
    if (J <= 1e-10f) {
        // Degenerate case
        P.setZero();
        return;
    }

    Eigen::Matrix3f F_inv_T = F.inverse().transpose();
    float log_J = logf(J);

    P = mu * (F - F_inv_T) + lambda * log_J * F_inv_T;
}

// Compute gradient contribution from one tetrahedron
__device__ void add_element_gradient(
    const float* x_curr,
    const int* tet,
    const float* Dm_inv,
    float volume,
    float mu,
    float lambda,
    float* grad_local)
{
    Eigen::Matrix3f F;
    compute_deformation_gradient(x_curr, tet, Dm_inv, F);

    Eigen::Matrix3f P;
    compute_pk1_stress(F, mu, lambda, P);

    // Load Dm_inv
    Eigen::Matrix3f Dm_inv_mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_mat(i, j) = Dm_inv[j * 3 + i];
        }
    }

    // H = -V * P * Dm_inv^T (force on vertices)
    Eigen::Matrix3f H = -volume * P * Dm_inv_mat.transpose();

    // Forces on vertices 1, 2, 3
    for (int i = 0; i < 3; i++) {
        grad_local[1 * 3 + i] = H(i, 0);
        grad_local[2 * 3 + i] = H(i, 1);
        grad_local[3 * 3 + i] = H(i, 2);
    }

    // Force on vertex 0 (equilibrium)
    for (int i = 0; i < 3; i++) {
        grad_local[0 * 3 + i] = -(H(i, 0) + H(i, 1) + H(i, 2));
    }
}

// Gradient kernel
__global__ void compute_gradient_nh_kernel(
    const float* x_curr,
    const float* x_tilde,
    const float* M_diag,
    const float* f_ext,
    const int* tetrahedra,
    const float* Dm_inv,
    const float* volumes,
    float mu,
    float lambda,
    float dt,
    int num_particles,
    int num_elements,
    float* grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    // Initialize with inertial term: M * (x - x_tilde)
    grad[i * 3 + 0] = M_diag[i * 3 + 0] * (x_curr[i * 3 + 0] - x_tilde[i * 3 + 0]);
    grad[i * 3 + 1] = M_diag[i * 3 + 1] * (x_curr[i * 3 + 1] - x_tilde[i * 3 + 1]);
    grad[i * 3 + 2] = M_diag[i * 3 + 2] * (x_curr[i * 3 + 2] - x_tilde[i * 3 + 2]);

    // Subtract external forces
    grad[i * 3 + 0] -= dt * dt * f_ext[i * 3 + 0];
    grad[i * 3 + 1] -= dt * dt * f_ext[i * 3 + 1];
    grad[i * 3 + 2] -= dt * dt * f_ext[i * 3 + 2];
}

// Accumulate elastic forces from elements
__global__ void accumulate_elastic_forces_kernel(
    const float* x_curr,
    const int* tetrahedra,
    const float* Dm_inv,
    const float* volumes,
    float mu,
    float lambda,
    float dt,
    int num_elements,
    float* grad)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    const int* tet = &tetrahedra[elem_idx * 4];
    const float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    float volume = volumes[elem_idx];

    float grad_local[12] = {0};
    add_element_gradient(x_curr, tet, Dm_inv_local, volume, mu, lambda, grad_local);

    // Add scaled elastic forces to gradient (dt^2 scaling)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            atomicAdd(&grad[tet[i] * 3 + j], dt * dt * grad_local[i * 3 + j]);
        }
    }
}

void compute_gradient_nh_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle tetrahedra,
    cuda::CUDALinearBufferHandle Dm_inv,
    cuda::CUDALinearBufferHandle volumes,
    float mu,
    float lambda,
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle grad)
{
    int block_size = 256;
    int num_blocks_particles = (num_particles + block_size - 1) / block_size;
    int num_blocks_elements = (num_elements + block_size - 1) / block_size;

    // First pass: inertial and external forces
    compute_gradient_nh_kernel<<<num_blocks_particles, block_size>>>(
        x_curr->get_device_ptr<float>(),
        x_tilde->get_device_ptr<float>(),
        M_diag->get_device_ptr<float>(),
        f_ext->get_device_ptr<float>(),
        tetrahedra->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        mu,
        lambda,
        dt,
        num_particles,
        num_elements,
        grad->get_device_ptr<float>());

    // Second pass: elastic forces
    accumulate_elastic_forces_kernel<<<num_blocks_elements, block_size>>>(
        x_curr->get_device_ptr<float>(),
        tetrahedra->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        mu,
        lambda,
        dt,
        num_elements,
        grad->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

// ============================================================================
// Hessian Construction
// ============================================================================

// Custom 3x3 symmetric eigenvalue decomposition (same as mass-spring)
__device__ void eigen_decomposition_3x3_nh(
    const Eigen::Matrix3f& A,
    Eigen::Vector3f& eigenvalues,
    Eigen::Matrix3f& eigenvectors)
{
    const int MAX_ITER = 50;
    const float EPSILON = 1e-10f;

    eigenvectors.setIdentity();
    Eigen::Matrix3f a = A;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        float max_val = fabsf(a(0, 1));

        if (fabsf(a(0, 2)) > max_val) {
            p = 0;
            q = 2;
            max_val = fabsf(a(0, 2));
        }
        if (fabsf(a(1, 2)) > max_val) {
            p = 1;
            q = 2;
            max_val = fabsf(a(1, 2));
        }

        if (max_val < EPSILON)
            break;

        // Compute rotation angle
        float theta = 0.5f * atan2f(2.0f * a(p, q), a(q, q) - a(p, p));
        float c = cosf(theta);
        float s = sinf(theta);

        // Apply Givens rotation
        Eigen::Matrix3f G;
        G.setIdentity();
        G(p, p) = c;
        G(q, q) = c;
        G(p, q) = s;
        G(q, p) = -s;

        a = G.transpose() * a * G;
        eigenvectors = eigenvectors * G;
    }

    eigenvalues(0) = a(0, 0);
    eigenvalues(1) = a(1, 1);
    eigenvalues(2) = a(2, 2);
}

// Project matrix to PSD
__device__ Eigen::Matrix3f project_psd_nh(const Eigen::Matrix3f& H)
{
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;

    eigen_decomposition_3x3_nh(H, eigenvalues, eigenvectors);

    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < 0.0f)
            eigenvalues(i) = 0.0f;
    }

    Eigen::Matrix3f result = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    return result;
}

// Compute element Hessian (12x12) for Neo-Hookean model
__device__ void compute_element_hessian(
    const float* x_curr,
    const int* tet,
    const float* Dm_inv,
    float volume,
    float mu,
    float lambda,
    Eigen::Matrix<float, 12, 12>& K_elem)
{
    Eigen::Matrix3f F;
    compute_deformation_gradient(x_curr, tet, Dm_inv, F);

    float J = F.determinant();
    if (J <= 1e-10f) {
        K_elem.setZero();
        return;
    }

    Eigen::Matrix3f F_inv = F.inverse();
    Eigen::Matrix3f F_inv_T = F_inv.transpose();
    float log_J = logf(J);

    // Load Dm_inv
    Eigen::Matrix3f Dm_inv_mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_mat(i, j) = Dm_inv[j * 3 + i];
        }
    }

    // Compute elasticity tensor components
    // This is a simplified version - full implementation would compute dP/dF tensor
    // For Neo-Hookean: ∂²Ψ/∂F∂F involves material tangent moduli

    // Simplified approach: numerical or analytical tangent stiffness
    // Here we use a simplified form that's positive definite

    K_elem.setZero();

    // Build 12x12 stiffness using 3x3 blocks
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            Eigen::Matrix3f K_ij;
            K_ij.setZero();

            if (i == j) {
                // Diagonal blocks
                K_ij = (mu + lambda) * Eigen::Matrix3f::Identity();
            } else if (i == 0 || j == 0) {
                // Blocks involving vertex 0
                K_ij = -mu * Eigen::Matrix3f::Identity();
            } else {
                // Off-diagonal blocks between vertices 1,2,3
                int idx_i = i - 1;
                int idx_j = j - 1;
                K_ij = mu * Dm_inv_mat.col(idx_i) * Dm_inv_mat.col(idx_j).transpose();
            }

            // Apply volume scaling and project to PSD
            K_ij = volume * project_psd_nh(K_ij);

            // Fill into 12x12 matrix
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    K_elem(i * 3 + a, j * 3 + b) = K_ij(a, b);
                }
            }
        }
    }
}

// Binary search for entry position (same as mass-spring)
__device__ int find_entry_position_nh(
    const int* unique_rows,
    const int* unique_cols,
    int nnz,
    int target_row,
    int target_col)
{
    int left = 0;
    int right = nnz - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int row = unique_rows[mid];
        int col = unique_cols[mid];

        if (row == target_row && col == target_col) {
            return mid;
        } else if (row < target_row || (row == target_row && col < target_col)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

// Build CSR structure for Neo-Hookean Hessian
NeoHookeanCSRStructure build_hessian_structure_nh_gpu(
    cuda::CUDALinearBufferHandle tetrahedra,
    int num_particles,
    int num_elements)
{
    NeoHookeanCSRStructure structure;
    int n = num_particles * 3;

    const int* tet_ptr = tetrahedra->get_device_ptr<int>();

    // Each tetrahedron contributes 12x12 = 144 entries
    int num_mass_entries = n;
    int num_element_entries = num_elements * 144;
    int total_entries = num_mass_entries + num_element_entries;

    // Allocate temporary buffers
    auto d_all_rows = cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_all_cols = cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_write_offset = cuda::create_cuda_linear_buffer<int>(1);
    cudaMemset(d_write_offset->get_device_ptr<int>(), 0, sizeof(int));

    int* rows = d_all_rows->get_device_ptr<int>();
    int* cols = d_all_cols->get_device_ptr<int>();
    int* write_offset = d_write_offset->get_device_ptr<int>();

    // Generate all (row, col) pairs
    cuda::GPUParallelFor("generate_entries_nh", total_entries, [=] __device__(int idx) {
        if (idx < num_mass_entries) {
            // Mass diagonal entries
            rows[idx] = idx;
            cols[idx] = idx;
        } else {
            // Element entries
            int elem_entry_idx = idx - num_mass_entries;
            int elem_idx = elem_entry_idx / 144;
            int local_idx = elem_entry_idx % 144;

            int local_row = local_idx / 12;
            int local_col = local_idx % 12;

            const int* tet = &tet_ptr[elem_idx * 4];
            int global_row = tet[local_row / 3] * 3 + (local_row % 3);
            int global_col = tet[local_col / 3] * 3 + (local_col % 3);

            rows[idx] = global_row;
            cols[idx] = global_col;
        }
    });

    // Sort and deduplicate
    thrust::device_ptr<int> rows_ptr(rows);
    thrust::device_ptr<int> cols_ptr(cols);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(rows_ptr, cols_ptr));
    auto zip_end = zip_begin + total_entries;

    thrust::sort(zip_begin, zip_end, [] __device__(const auto& a, const auto& b) {
        int r1 = thrust::get<0>(a);
        int c1 = thrust::get<1>(a);
        int r2 = thrust::get<0>(b);
        int c2 = thrust::get<1>(b);
        return (r1 < r2) || (r1 == r2 && c1 < c2);
    });

    cudaDeviceSynchronize();

    auto new_end = thrust::unique(zip_begin, zip_end, [] __device__(const auto& a, const auto& b) {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    });

    int nnz = new_end - zip_begin;
    cudaDeviceSynchronize();

    // Allocate CSR arrays
    structure.col_indices = cuda::create_cuda_linear_buffer<int>(nnz);
    structure.row_offsets = cuda::create_cuda_linear_buffer<int>(n + 1);
    structure.mass_value_positions = cuda::create_cuda_linear_buffer<int>(n);
    structure.element_value_positions = cuda::create_cuda_linear_buffer<int>(num_elements * 144);

    cudaMemcpy(
        structure.col_indices->get_device_ptr<int>(),
        cols,
        nnz * sizeof(int),
        cudaMemcpyDeviceToDevice);

    // Build row_offsets
    thrust::device_ptr<int> row_offsets_ptr(structure.row_offsets->get_device_ptr<int>());
    thrust::fill(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, 0);

    int* row_offsets_raw = structure.row_offsets->get_device_ptr<int>();
    thrust::for_each(thrust::device, rows_ptr, rows_ptr + nnz, [=] __device__(int row) {
        atomicAdd(&row_offsets_raw[row + 1], 1);
    });

    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, row_offsets_ptr);
    cudaDeviceSynchronize();

    // Build mass diagonal positions
    const int* unique_rows = rows;
    const int* unique_cols = cols;
    int* mass_positions = structure.mass_value_positions->get_device_ptr<int>();

    cuda::GPUParallelFor("build_mass_positions_nh", n, [=] __device__(int dof) {
        mass_positions[dof] = find_entry_position_nh(unique_rows, unique_cols, nnz, dof, dof);
    });

    // Build element positions
    int* elem_positions = structure.element_value_positions->get_device_ptr<int>();

    cuda::GPUParallelFor("build_element_positions_nh", num_elements * 144, [=] __device__(int idx) {
        int elem_idx = idx / 144;
        int local_idx = idx % 144;

        int local_row = local_idx / 12;
        int local_col = local_idx % 12;

        const int* tet = &tet_ptr[elem_idx * 4];
        int global_row = tet[local_row / 3] * 3 + (local_row % 3);
        int global_col = tet[local_col / 3] * 3 + (local_col % 3);

        elem_positions[idx] = find_entry_position_nh(unique_rows, unique_cols, nnz, global_row, global_col);
    });

    structure.num_rows = n;
    structure.num_cols = n;
    structure.nnz = nnz;

    return structure;
}

// Fill Hessian values kernel
__global__ void fill_hessian_values_nh_kernel(
    const float* x_curr,
    const int* tetrahedra,
    const float* Dm_inv,
    const float* volumes,
    const int* value_positions,
    float mu,
    float lambda,
    float dt,
    int num_elements,
    float* values)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    const int* tet = &tetrahedra[elem_idx * 4];
    const float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    float volume = volumes[elem_idx];

    Eigen::Matrix<float, 12, 12> K_elem;
    compute_element_hessian(x_curr, tet, Dm_inv_local, volume, mu, lambda, K_elem);

    // Scale by dt^2
    K_elem *= (dt * dt);

    // Write to CSR values array
    const int* positions = &value_positions[elem_idx * 144];
    for (int i = 0; i < 144; i++) {
        int pos = positions[i];
        if (pos >= 0) {
            atomicAdd(&values[pos], K_elem.data()[i]);
        }
    }
}

void update_hessian_values_nh_gpu(
    const NeoHookeanCSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle tetrahedra,
    cuda::CUDALinearBufferHandle Dm_inv,
    cuda::CUDALinearBufferHandle volumes,
    float mu,
    float lambda,
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle values)
{
    int num_dofs = num_particles * 3;

    // Zero out values
    cudaMemset(values->get_device_ptr<float>(), 0, csr_structure.nnz * sizeof(float));

    // Fill mass diagonal
    const float* M_diag_ptr = M_diag->get_device_ptr<float>();
    const int* mass_positions = csr_structure.mass_value_positions->get_device_ptr<int>();
    float* values_ptr = values->get_device_ptr<float>();

    cuda::GPUParallelFor("fill_mass_diagonal_nh", num_dofs, [=] __device__(int dof) {
        int pos = mass_positions[dof];
        if (pos >= 0) {
            values_ptr[pos] = M_diag_ptr[dof];
        }
    });

    // Fill element contributions
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fill_hessian_values_nh_kernel<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        tetrahedra->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        csr_structure.element_value_positions->get_device_ptr<int>(),
        mu,
        lambda,
        dt,
        num_elements,
        values_ptr);

    cudaDeviceSynchronize();
}

// ============================================================================
// Energy Computation
// ============================================================================

__global__ void compute_element_energy_kernel(
    const float* x_curr,
    const int* tetrahedra,
    const float* Dm_inv,
    const float* volumes,
    float mu,
    float lambda,
    int num_elements,
    float* element_energies)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    const int* tet = &tetrahedra[elem_idx * 4];
    const float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    float volume = volumes[elem_idx];

    Eigen::Matrix3f F;
    compute_deformation_gradient(x_curr, tet, Dm_inv_local, F);

    float psi = neo_hookean_energy_density(F, mu, lambda);
    element_energies[elem_idx] = volume * psi;
}

float compute_energy_nh_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle tetrahedra,
    cuda::CUDALinearBufferHandle Dm_inv,
    cuda::CUDALinearBufferHandle volumes,
    float mu,
    float lambda,
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle d_inertial_terms,
    cuda::CUDALinearBufferHandle d_element_energies,
    cuda::CUDALinearBufferHandle d_potential_terms)
{
    int n = num_particles * 3;

    float* x_ptr = x_curr->get_device_ptr<float>();
    float* x_tilde_ptr = x_tilde->get_device_ptr<float>();
    float* M_ptr = M_diag->get_device_ptr<float>();
    float* f_ptr = f_ext->get_device_ptr<float>();

    float* inertial_ptr = d_inertial_terms->get_device_ptr<float>();
    float* element_energy_ptr = d_element_energies->get_device_ptr<float>();
    float* potential_ptr = d_potential_terms->get_device_ptr<float>();

    // Compute inertial energy
    cuda::GPUParallelFor("compute_inertial_energy_nh", n, [=] __device__(int i) {
        float diff = x_ptr[i] - x_tilde_ptr[i];
        inertial_ptr[i] = 0.5f * M_ptr[i] * diff * diff;
    });

    thrust::device_ptr<float> d_inertial_thrust(inertial_ptr);
    float E_inertial = thrust::reduce(thrust::device, d_inertial_thrust, d_inertial_thrust + n);

    // Compute elastic energy
    cudaMemset(element_energy_ptr, 0, num_elements * sizeof(float));

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    compute_element_energy_kernel<<<num_blocks, block_size>>>(
        x_ptr,
        tetrahedra->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        mu,
        lambda,
        num_elements,
        element_energy_ptr);

    cudaDeviceSynchronize();

    thrust::device_ptr<float> d_element_thrust(element_energy_ptr);
    float E_elastic = thrust::reduce(thrust::device, d_element_thrust, d_element_thrust + num_elements);

    // Compute potential energy
    cuda::GPUParallelFor("compute_potential_energy_nh", n, [=] __device__(int i) {
        potential_ptr[i] = -dt * dt * x_ptr[i] * f_ptr[i];
    });

    thrust::device_ptr<float> d_potential_thrust(potential_ptr);
    float E_potential = thrust::reduce(thrust::device, d_potential_thrust, d_potential_thrust + n);

    float total_energy = E_inertial + dt * dt * E_elastic + E_potential;

    return total_energy;
}

// ============================================================================
// Reference Data Computation
// ============================================================================

__global__ void compute_Dm_inv_kernel(
    const float* positions,
    const int* tetrahedra,
    int num_elements,
    float* Dm_inv,
    float* volumes)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    const int* tet = &tetrahedra[elem_idx * 4];

    // Get rest positions
    Eigen::Vector3f x0(positions[tet[0] * 3 + 0], positions[tet[0] * 3 + 1], positions[tet[0] * 3 + 2]);
    Eigen::Vector3f x1(positions[tet[1] * 3 + 0], positions[tet[1] * 3 + 1], positions[tet[1] * 3 + 2]);
    Eigen::Vector3f x2(positions[tet[2] * 3 + 0], positions[tet[2] * 3 + 1], positions[tet[2] * 3 + 2]);
    Eigen::Vector3f x3(positions[tet[3] * 3 + 0], positions[tet[3] * 3 + 1], positions[tet[3] * 3 + 2]);

    // Compute Dm = [x1-x0, x2-x0, x3-x0]
    Eigen::Matrix3f Dm;
    Dm.col(0) = x1 - x0;
    Dm.col(1) = x2 - x0;
    Dm.col(2) = x3 - x0;

    // Compute volume: V = |det(Dm)| / 6
    float det_Dm = Dm.determinant();
    volumes[elem_idx] = fabsf(det_Dm) / 6.0f;

    // Compute inverse
    Eigen::Matrix3f Dm_inv_mat = Dm.inverse();

    // Store in column-major order
    float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_local[j * 3 + i] = Dm_inv_mat(i, j);
        }
    }
}

std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_reference_data_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle tetrahedra,
    int num_elements)
{
    auto Dm_inv = cuda::create_cuda_linear_buffer<float>(num_elements * 9);
    auto volumes = cuda::create_cuda_linear_buffer<float>(num_elements);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    compute_Dm_inv_kernel<<<num_blocks, block_size>>>(
        positions->get_device_ptr<float>(),
        tetrahedra->get_device_ptr<int>(),
        num_elements,
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>());

    cudaDeviceSynchronize();

    return std::make_tuple(Dm_inv, volumes);
}

// ============================================================================
// Vector Operations
// ============================================================================

void negate_nh_gpu(
    cuda::CUDALinearBufferHandle input,
    cuda::CUDALinearBufferHandle output,
    int size)
{
    const float* in_ptr = input->get_device_ptr<float>();
    float* out_ptr = output->get_device_ptr<float>();

    cuda::GPUParallelFor("negate_nh", size, [=] __device__(int i) { out_ptr[i] = -in_ptr[i]; });
}

void axpy_nh_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size)
{
    const float* x_ptr = x->get_device_ptr<float>();
    const float* y_ptr = y->get_device_ptr<float>();
    float* result_ptr = result->get_device_ptr<float>();

    cuda::GPUParallelFor("axpy_nh", size, [=] __device__(int i) {
        result_ptr[i] = alpha * x_ptr[i] + y_ptr[i];
    });
}

struct square_op_nh {
    __device__ float operator()(float x) const { return x * x; }
};

float compute_vector_norm_nh_gpu(cuda::CUDALinearBufferHandle vec, int size)
{
    float* vec_ptr = vec->get_device_ptr<float>();
    thrust::device_ptr<float> d_vec(vec_ptr);

    float sum_sq = thrust::transform_reduce(
        thrust::device,
        d_vec,
        d_vec + size,
        square_op_nh(),
        0.0f,
        thrust::plus<float>());

    return sqrtf(sum_sq);
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
