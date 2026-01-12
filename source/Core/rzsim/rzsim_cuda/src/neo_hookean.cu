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
        "explicit_step_nh", num_particles * 3, [=] __device__(int i) {
            x_tilde_ptr[i] = x_ptr[i] + dt * v_ptr[i];
        });
}

void setup_external_forces_nh_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext)
{
    float* f_ext_ptr = f_ext->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "setup_external_forces_nh", num_particles * 3, [=] __device__(int i) {
            // Gravity acts on z-component (i % 3 == 2)
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

// Neo-Hookean energy density: Ψ = μ/2 * (tr(F^T F) - 3) - μ log(J) + λ/2 *
// log(J)^2
__device__ float
neo_hookean_energy_density(const Eigen::Matrix3f& F, float mu, float lambda)
{
    float J = F.determinant();
    if (J <= 0.0f) {
        // Inverted element - use large penalty
        return 1e10f;
    }

    Eigen::Matrix3f FtF = F.transpose() * F;
    float I1 = FtF.trace();  // tr(F^T F)
    float log_J = logf(J);

    float psi =
        0.5f * mu * (I1 - 3.0f) - mu * log_J + 0.5f * lambda * log_J * log_J;
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
    if (fabsf(J) <= 1e-6f || !isfinite(J)) {
        // Degenerate/inverted case
        P.setZero();
        return;
    }

    Eigen::Matrix3f F_inv = F.inverse();
    Eigen::Matrix3f F_inv_T = F_inv.transpose();

    // Check inverse validity
    bool valid = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!isfinite(F_inv_T(i, j))) {
                valid = false;
                break;
            }
        }
        if (!valid)
            break;
    }

    if (!valid) {
        P.setZero();
        return;
    }

    float log_J = logf(fabsf(J));

    if (!isfinite(log_J)) {
        P.setZero();
        return;
    }

    P = mu * (F - F_inv_T) + lambda * log_J * F_inv_T;

    // Check result
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!isfinite(P(i, j))) {
                P.setZero();
                return;
            }
        }
    }
}

// Compute gradient contribution from one tetrahedron
// Now using Eigen properly - it works fine in CUDA device code!
__device__ void add_element_gradient(
    const float* x_curr,
    const int* tet,
    const float* Dm_inv,
    float volume,
    float mu,
    float lambda,
    float* grad_local)
{
    // Compute F using Eigen
    Eigen::Matrix3f F;
    compute_deformation_gradient(x_curr, tet, Dm_inv, F);

    // Compute PK1 stress using the existing function
    Eigen::Matrix3f P;
    compute_pk1_stress(F, mu, lambda, P);

    // Check if stress is valid
    bool valid = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!isfinite(P(i, j))) {
                valid = false;
                break;
            }
        }
        if (!valid)
            break;
    }

    if (!valid) {
        for (int i = 0; i < 12; i++)
            grad_local[i] = 0.0f;
        return;
    }

    // Load Dm_inv as Eigen matrix
    Eigen::Matrix3f Dm_inv_mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_mat(i, j) = Dm_inv[j * 3 + i];  // Column-major
        }
    }

    // H = V * P * Dm_inv^T (gradient of elastic energy w.r.t. vertex positions)
    Eigen::Matrix3f H = volume * P * Dm_inv_mat.transpose();

    // Forces on vertices 1, 2, 3 (columns of H)
    for (int i = 0; i < 3; i++) {
        grad_local[1 * 3 + i] = H(i, 0);
        grad_local[2 * 3 + i] = H(i, 1);
        grad_local[3 * 3 + i] = H(i, 2);
    }

    // Force on vertex 0 (equilibrium: sum of forces = 0)
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
    grad[i * 3 + 0] =
        M_diag[i * 3 + 0] * (x_curr[i * 3 + 0] - x_tilde[i * 3 + 0]);
    grad[i * 3 + 1] =
        M_diag[i * 3 + 1] * (x_curr[i * 3 + 1] - x_tilde[i * 3 + 1]);
    grad[i * 3 + 2] =
        M_diag[i * 3 + 2] * (x_curr[i * 3 + 2] - x_tilde[i * 3 + 2]);

    // Subtract external forces (with dt² scaling like mass-spring)
    grad[i * 3 + 0] -= dt * dt * f_ext[i * 3 + 0];
    grad[i * 3 + 1] -= dt * dt * f_ext[i * 3 + 1];
    grad[i * 3 + 2] -= dt * dt * f_ext[i * 3 + 2];
}

// Accumulate elastic forces from elements
__global__ void accumulate_elastic_forces_kernel(
    const float* x_curr,
    const unsigned* adjacency,
    const unsigned* offsets,
    const int* element_to_vertex,
    const int* element_to_local_face,
    const float* Dm_inv,
    const float* volumes,
    float mu,
    float lambda,
    float dt,
    int num_elements,
    int num_particles,
    float* grad)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    // Step 1: Test mapping lookup
    int v_apex = element_to_vertex[elem_idx];
    int local_face_idx = element_to_local_face[elem_idx];

    if (v_apex < 0 || v_apex >= num_particles) {
        return;
    }

    unsigned offset = offsets[v_apex];
    unsigned count = adjacency[offset];

    if (local_face_idx < 0 || local_face_idx >= (int)count) {
        return;
    }

    unsigned face_a = adjacency[offset + 1 + local_face_idx * 3 + 0];
    unsigned face_b = adjacency[offset + 1 + local_face_idx * 3 + 1];
    unsigned face_c = adjacency[offset + 1 + local_face_idx * 3 + 2];

    if (face_a >= (unsigned)num_particles ||
        face_b >= (unsigned)num_particles ||
        face_c >= (unsigned)num_particles) {
        return;
    }

    int tet[4] = { v_apex, (int)face_a, (int)face_b, (int)face_c };
    const float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    float volume = volumes[elem_idx];

    if (volume <= 0.0f || !isfinite(volume)) {
        return;
    }

    // Compute gradient using raw float arrays (Eigen version causes device
    // errors)
    float grad_local[12] = { 0 };
    add_element_gradient(
        x_curr, tet, Dm_inv_local, volume, mu, lambda, grad_local);

    // Check for NaN/Inf in grad_local
    bool valid = true;
    for (int i = 0; i < 12; i++) {
        if (!isfinite(grad_local[i])) {
            valid = false;
            break;
        }
    }

    if (!valid) {
        return;
    }

    // Add elastic forces to gradient
    // Like mass-spring: gradient includes dt² scaling for elastic term
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = tet[i] * 3 + j;
            if (idx >= 0 && idx < num_particles * 3) {
                atomicAdd(&grad[idx], dt * dt * grad_local[i * 3 + j]);
            }
        }
    }
}

void compute_gradient_nh_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
    cuda::CUDALinearBufferHandle element_to_vertex,
    cuda::CUDALinearBufferHandle element_to_local_face,
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
        nullptr,  // Not used in this kernel
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
        adjacency->get_device_ptr<unsigned>(),
        offsets->get_device_ptr<unsigned>(),
        element_to_vertex->get_device_ptr<int>(),
        element_to_local_face->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        mu,
        lambda,
        dt,
        num_elements,
        num_particles,
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

    // Clamp negative eigenvalues to small positive value for regularization
    // This prevents singular matrices while maintaining positive definiteness
    const float min_eigenvalue = 1e-8f;  // Small regularization
    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < min_eigenvalue)
            eigenvalues(i) = min_eigenvalue;
    }

    Eigen::Matrix3f result =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    return result;
}

// Compute element Hessian (12x12) for Neo-Hookean model
// Using St. Venant-Kirchhoff (SVK) Hessian as approximation
// This is standard practice: use Neo-Hookean for gradient (forces)
// but SVK for Hessian (stiffness matrix) because it's simpler and SPD
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
    if (J <= 1e-10f || !isfinite(J)) {
        K_elem.setZero();
        return;
    }

    // Load Dm_inv
    Eigen::Matrix3f Dm_inv_mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_mat(i, j) = Dm_inv[j * 3 + i];
        }
    }

    // St. Venant-Kirchhoff (SVK) elasticity tensor for tetrahedral FEM
    // This gives a symmetric positive-definite Hessian
    // C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl +
    // delta_il * delta_jk)

    K_elem.setZero();

    // Build element stiffness matrix
    // K_ab^(alpha,beta) = V * sum_{i,j,k,l} dF_ij/dx_a^alpha * C_ijkl *
    // dF_kl/dx_b^beta For linear tetrahedron: dF/dx relates to Dm_inv

    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            Eigen::Matrix3f K_ab;
            K_ab.setZero();

            // Shape function gradients in reference configuration
            Eigen::Vector3f grad_Na, grad_Nb;
            if (a == 0) {
                grad_Na = -(
                    Dm_inv_mat.col(0) + Dm_inv_mat.col(1) + Dm_inv_mat.col(2));
            }
            else {
                grad_Na = Dm_inv_mat.col(a - 1);
            }

            if (b == 0) {
                grad_Nb = -(
                    Dm_inv_mat.col(0) + Dm_inv_mat.col(1) + Dm_inv_mat.col(2));
            }
            else {
                grad_Nb = Dm_inv_mat.col(b - 1);
            }

            // Compute 3x3 stiffness block using SVK constitutive model
            // For linear FEM: K_ab^(alpha,beta) = V * sum_{i,j} grad_Na^i C_{i
            // alpha j beta} grad_Nb^j where C_{ijkl} = lambda * delta_ij *
            // delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
            
            // Correct formula for isotropic elasticity:
            // K_ab = V * [lambda * (grad_Na ⊗ grad_Nb) + mu * (grad_Na ⊗ grad_Nb + grad_Nb ⊗ grad_Na)]
            float dot_product = grad_Na.dot(grad_Nb);
            K_ab = volume * lambda * (grad_Na * grad_Nb.transpose());
            K_ab += volume * mu * (grad_Na * grad_Nb.transpose() + 
                                   grad_Nb * grad_Na.transpose());
            
            // Ensure strict symmetry (important for numerical stability)
            K_ab = 0.5f * (K_ab + K_ab.transpose());

            // Project to PSD
            K_ab = project_psd_nh(K_ab);

            // Fill into 12x12 matrix
            for (int alpha = 0; alpha < 3; alpha++) {
                for (int beta = 0; beta < 3; beta++) {
                    K_elem(a * 3 + alpha, b * 3 + beta) = K_ab(alpha, beta);
                }
            }
        }
    }
    
    // Verify symmetry of element stiffness matrix (debug only)
    #ifdef DEBUG_HESSIAN_SYMMETRY
    for (int i = 0; i < 12; i++) {
        for (int j = i + 1; j < 12; j++) {
            float diff = fabsf(K_elem(i,j) - K_elem(j,i));
            float mag = fmaxf(fabsf(K_elem(i,j)), fabsf(K_elem(j,i)));
            if (mag > 1e-10f && diff > 1e-4f * mag) {
                printf("[HESSIAN] Asymmetry at K(%d,%d): %.6e vs K(%d,%d): %.6e, diff=%.6e\\n",
                       i, j, K_elem(i,j), j, i, K_elem(j,i), diff);
            }
        }
    }
    #endif
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
        }
        else if (row < target_row || (row == target_row && col < target_col)) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }

    return -1;
}

// Build CSR structure for Neo-Hookean Hessian
NeoHookeanCSRStructure build_hessian_structure_nh_gpu(
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
    cuda::CUDALinearBufferHandle element_to_vertex,
    cuda::CUDALinearBufferHandle element_to_local_face,
    int num_particles,
    int num_elements)
{
    NeoHookeanCSRStructure structure;
    int n = num_particles * 3;

    const unsigned* adj_ptr = adjacency->get_device_ptr<unsigned>();
    const unsigned* off_ptr = offsets->get_device_ptr<unsigned>();
    const int* elem_to_v_ptr = element_to_vertex->get_device_ptr<int>();
    const int* elem_to_lf_ptr = element_to_local_face->get_device_ptr<int>();

    // Each tetrahedron contributes 12x12 = 144 entries
    // For symmetric matrix, we need both (i,j) and (j,i) if i != j
    // Upper triangle: 12*13/2 = 78 unique entries per element
    // But we'll generate all 144 and rely on deduplication for simplicity
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
    cuda::GPUParallelFor(
        "generate_entries_nh", total_entries, [=] __device__(int idx) {
            if (idx < num_mass_entries) {
                // Mass diagonal entries
                rows[idx] = idx;
                cols[idx] = idx;
            }
            else {
                // Element entries
                int elem_entry_idx = idx - num_mass_entries;
                int elem_idx = elem_entry_idx / 144;
                int local_idx = elem_entry_idx % 144;

                int local_row = local_idx / 12;
                int local_col = local_idx % 12;

                // Use precomputed mapping
                int v_apex = elem_to_v_ptr[elem_idx];
                int local_face_idx = elem_to_lf_ptr[elem_idx];

                unsigned offset = off_ptr[v_apex];
                unsigned face_a = adj_ptr[offset + 1 + local_face_idx * 3 + 0];
                unsigned face_b = adj_ptr[offset + 1 + local_face_idx * 3 + 1];
                unsigned face_c = adj_ptr[offset + 1 + local_face_idx * 3 + 2];
                int tet[4] = { v_apex, (int)face_a, (int)face_b, (int)face_c };

                int global_row = tet[local_row / 3] * 3 + (local_row % 3);
                int global_col = tet[local_col / 3] * 3 + (local_col % 3);

                rows[idx] = global_row;
                cols[idx] = global_col;
            }
        });

    // Sort and deduplicate
    thrust::device_ptr<int> rows_ptr(rows);
    thrust::device_ptr<int> cols_ptr(cols);

    auto zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(rows_ptr, cols_ptr));
    auto zip_end = zip_begin + total_entries;

    thrust::sort(
        zip_begin, zip_end, [] __device__(const auto& a, const auto& b) {
            int r1 = thrust::get<0>(a);
            int c1 = thrust::get<1>(a);
            int r2 = thrust::get<0>(b);
            int c2 = thrust::get<1>(b);
            return (r1 < r2) || (r1 == r2 && c1 < c2);
        });

    cudaDeviceSynchronize();

    auto new_end = thrust::unique(
        zip_begin, zip_end, [] __device__(const auto& a, const auto& b) {
            return thrust::get<0>(a) == thrust::get<0>(b) &&
                   thrust::get<1>(a) == thrust::get<1>(b);
        });

    int nnz = new_end - zip_begin;
    cudaDeviceSynchronize();

    // Debug: print first few (row, col) pairs after deduplication
    std::vector<int> debug_rows(std::min(20, nnz));
    std::vector<int> debug_cols(std::min(20, nnz));
    cudaMemcpy(
        debug_rows.data(),
        rows,
        debug_rows.size() * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        debug_cols.data(),
        cols,
        debug_cols.size() * sizeof(int),
        cudaMemcpyDeviceToHost);
    fprintf(
        stderr,
        "[NeoHookean CSR] After deduplication, nnz=%d, first 20 entries:\n",
        nnz);
    for (size_t i = 0; i < debug_rows.size(); i++) {
        fprintf(stderr, "  [%zu] (%d, %d)\n", i, debug_rows[i], debug_cols[i]);
    }

    // Allocate CSR arrays
    structure.col_indices = cuda::create_cuda_linear_buffer<int>(nnz);
    structure.row_offsets = cuda::create_cuda_linear_buffer<int>(n + 1);
    structure.mass_value_positions = cuda::create_cuda_linear_buffer<int>(n);
    structure.element_value_positions =
        cuda::create_cuda_linear_buffer<int>(num_elements * 144);

    // Copy deduplicated rows and cols to permanent storage
    auto d_unique_rows = cuda::create_cuda_linear_buffer<int>(nnz);
    cudaMemcpy(
        d_unique_rows->get_device_ptr<int>(),
        rows,
        nnz * sizeof(int),
        cudaMemcpyDeviceToDevice);

    cudaMemcpy(
        structure.col_indices->get_device_ptr<int>(),
        cols,
        nnz * sizeof(int),
        cudaMemcpyDeviceToDevice);

    // Build row_offsets using the deduplicated rows
    thrust::device_ptr<int> row_offsets_ptr(
        structure.row_offsets->get_device_ptr<int>());
    thrust::fill(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, 0);
    cudaDeviceSynchronize();

    int* row_offsets_raw = structure.row_offsets->get_device_ptr<int>();
    int* unique_rows_ptr = d_unique_rows->get_device_ptr<int>();

    // Debug: check row_offsets after fill
    std::vector<int> debug_after_fill(n + 1);
    cudaMemcpy(
        debug_after_fill.data(),
        row_offsets_raw,
        (n + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);
    fprintf(
        stderr,
        "[DEBUG] After fill, row_offsets[0]=%d, row_offsets[1]=%d\n",
        debug_after_fill[0],
        debug_after_fill[1]);

    thrust::device_ptr<int> unique_rows_tptr(unique_rows_ptr);
    fprintf(
        stderr, "[DEBUG] About to call thrust::for_each with nnz=%d\n", nnz);

    thrust::for_each(
        thrust::device,
        unique_rows_tptr,
        unique_rows_tptr + nnz,
        [=] __device__(int row) { atomicAdd(&row_offsets_raw[row], 1); });

    cudaDeviceSynchronize();

    // Debug: check row_offsets after for_each
    std::vector<int> debug_after_foreach(n + 1);
    cudaMemcpy(
        debug_after_foreach.data(),
        row_offsets_raw,
        (n + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);
    fprintf(
        stderr,
        "[DEBUG] After for_each, row_offsets[0]=%d, row_offsets[1]=%d, "
        "row_offsets[2]=%d\n",
        debug_after_foreach[0],
        debug_after_foreach[1],
        debug_after_foreach[2]);
    fprintf(stderr, "[DEBUG] Full row_offsets before scan:\n");
    for (int i = 0; i <= n; i++) {
        fprintf(stderr, "  before_scan[%d] = %d\n", i, debug_after_foreach[i]);
    }

    thrust::exclusive_scan(
        thrust::device,
        row_offsets_ptr,
        row_offsets_ptr + n + 1,
        row_offsets_ptr);
    cudaDeviceSynchronize();

    // Debug: check row_offsets after scan
    std::vector<int> debug_after_scan(n + 1);
    cudaMemcpy(
        debug_after_scan.data(),
        row_offsets_raw,
        (n + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);
    fprintf(stderr, "[DEBUG] Full row_offsets after scan:\n");
    for (int i = 0; i <= n; i++) {
        fprintf(stderr, "  after_scan[%d] = %d\n", i, debug_after_scan[i]);
    }

    // Debug: print first 30 (row,col) pairs from d_unique_rows after unique
    fprintf(
        stderr,
        "\n[DEBUG] First 30 (row,col) pairs from d_unique_rows (nnz=%d):\n",
        (int)nnz);
    std::vector<int> debug_rows2(std::min(30, (int)nnz));
    std::vector<int> debug_cols2(std::min(30, (int)nnz));
    cudaMemcpy(
        debug_rows2.data(),
        unique_rows_ptr,
        std::min(30, (int)nnz) * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        debug_cols2.data(),
        structure.col_indices->get_device_ptr<int>(),
        std::min(30, (int)nnz) * sizeof(int),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < std::min(30, (int)nnz); i++) {
        fprintf(
            stderr,
            "  Entry %d: row=%d, col=%d\n",
            i,
            debug_rows2[i],
            debug_cols2[i]);
    }

    // Debug: print row_offsets
    std::vector<int> debug_row_offsets(n + 1);
    cudaMemcpy(
        debug_row_offsets.data(),
        row_offsets_raw,
        (n + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);
    fprintf(stderr, "\n[NeoHookean CSR] row_offsets (num_rows=%d):\n", n);
    for (int i = 0; i <= n; i++) {
        fprintf(stderr, "  row_offsets[%d] = %d\n", i, debug_row_offsets[i]);
    }

    // Build mass diagonal positions
    const int* unique_rows = d_unique_rows->get_device_ptr<int>();
    const int* unique_cols = structure.col_indices->get_device_ptr<int>();
    int* mass_positions = structure.mass_value_positions->get_device_ptr<int>();

    cuda::GPUParallelFor("build_mass_positions_nh", n, [=] __device__(int dof) {
        mass_positions[dof] =
            find_entry_position_nh(unique_rows, unique_cols, nnz, dof, dof);
    });

    // Build element positions
    int* elem_positions =
        structure.element_value_positions->get_device_ptr<int>();

    cuda::GPUParallelFor(
        "build_element_positions_nh",
        num_elements * 144,
        [=] __device__(int idx) {
            int elem_idx = idx / 144;
            int local_idx = idx % 144;

            int local_row = local_idx / 12;
            int local_col = local_idx % 12;

            // Use precomputed mapping
            int v_apex = elem_to_v_ptr[elem_idx];
            int local_face_idx = elem_to_lf_ptr[elem_idx];

            unsigned offset = off_ptr[v_apex];
            unsigned face_a = adj_ptr[offset + 1 + local_face_idx * 3 + 0];
            unsigned face_b = adj_ptr[offset + 1 + local_face_idx * 3 + 1];
            unsigned face_c = adj_ptr[offset + 1 + local_face_idx * 3 + 2];
            int tet[4] = { v_apex, (int)face_a, (int)face_b, (int)face_c };

            int global_row = tet[local_row / 3] * 3 + (local_row % 3);
            int global_col = tet[local_col / 3] * 3 + (local_col % 3);

            elem_positions[idx] = find_entry_position_nh(
                unique_rows, unique_cols, nnz, global_row, global_col);
        });

    structure.num_rows = n;
    structure.num_cols = n;
    structure.nnz = nnz;

    return structure;
}

// Fill Hessian values kernel
__global__ void fill_hessian_values_nh_kernel(
    const float* x_curr,
    const unsigned* adjacency,
    const unsigned* offsets,
    const int* element_to_vertex,
    const int* element_to_local_face,
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

    // Use precomputed mapping
    int v_apex = element_to_vertex[elem_idx];
    int local_face_idx = element_to_local_face[elem_idx];

    unsigned offset = offsets[v_apex];
    unsigned face_a = adjacency[offset + 1 + local_face_idx * 3 + 0];
    unsigned face_b = adjacency[offset + 1 + local_face_idx * 3 + 1];
    unsigned face_c = adjacency[offset + 1 + local_face_idx * 3 + 2];
    int tet[4] = { v_apex, (int)face_a, (int)face_b, (int)face_c };
    const float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    float volume = volumes[elem_idx];

    Eigen::Matrix<float, 12, 12> K_elem;
    compute_element_hessian(
        x_curr, tet, Dm_inv_local, volume, mu, lambda, K_elem);

    // Scale by dt² (same as mass-spring)
    K_elem *= (dt * dt);

    // Write to CSR values array
    // CRITICAL: Eigen uses column-major order, but we need row-major for CSR
    // K_elem.data()[i] gives column-major, but positions[] expects row-major
    // For symmetric matrices: K_elem(row, col) should equal K_elem(col, row)
    const int* positions = &value_positions[elem_idx * 144];
    for (int row = 0; row < 12; row++) {
        for (int col = 0; col < 12; col++) {
            int idx = row * 12 + col;  // Row-major index
            int pos = positions[idx];
            if (pos >= 0) {
                // Use (row, col) to access Eigen matrix correctly
                // For symmetric matrix, K(row,col) should already equal K(col,row)
                // due to the symmetric construction in compute_element_hessian
                atomicAdd(&values[pos], K_elem(row, col));
            }
        }
    }
}

void update_hessian_values_nh_gpu(
    const NeoHookeanCSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
    cuda::CUDALinearBufferHandle element_to_vertex,
    cuda::CUDALinearBufferHandle element_to_local_face,
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

    fprintf(
        stderr,
        "[NeoHookean Hessian] START: num_elements=%d, nnz=%d\n",
        num_elements,
        csr_structure.nnz);
    fflush(stderr);

    // Zero out values
    cudaMemset(
        values->get_device_ptr<float>(), 0, csr_structure.nnz * sizeof(float));

    // Fill mass diagonal (like mass-spring: M + regularization)
    const float* M_diag_ptr = M_diag->get_device_ptr<float>();
    const int* mass_positions =
        csr_structure.mass_value_positions->get_device_ptr<int>();
    float* values_ptr = values->get_device_ptr<float>();

    cuda::GPUParallelFor(
        "fill_mass_diagonal_nh", num_dofs, [=] __device__(int dof) {
            int pos = mass_positions[dof];
            if (pos >= 0) {
                float regularization = 1e-6f;
                values_ptr[pos] = M_diag_ptr[dof] + regularization;
            }
        });

    // Fill element contributions
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fill_hessian_values_nh_kernel<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        adjacency->get_device_ptr<unsigned>(),
        offsets->get_device_ptr<unsigned>(),
        element_to_vertex->get_device_ptr<int>(),
        element_to_local_face->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        csr_structure.element_value_positions->get_device_ptr<int>(),
        mu,
        lambda,
        dt,
        num_elements,
        values_ptr);

    cudaDeviceSynchronize();

    // DEBUG: Check if element_value_positions has -1 values
    auto elem_pos_host =
        csr_structure.element_value_positions->get_host_vector<int>();
    int num_invalid = 0;
    int num_valid = 0;
    for (int i = 0; i < num_elements * 144; i++) {
        if (elem_pos_host[i] < 0) {
            num_invalid++;
        }
        else {
            num_valid++;
        }
    }
    fprintf(
        stderr,
        "[NeoHookean Hessian] element_value_positions: valid=%d, invalid=%d\n",
        num_valid,
        num_invalid);
    fflush(stderr);

    // DEBUG: Check if values are actually filled
    auto values_host = values->get_host_vector<float>();
    int num_nonzero = 0;
    float max_val = 0.0f;
    for (int i = 0; i < csr_structure.nnz; i++) {
        if (values_host[i] != 0.0f) {
            num_nonzero++;
            max_val = std::max(max_val, std::abs(values_host[i]));
        }
    }
    fprintf(
        stderr,
        "[NeoHookean Hessian] nnz=%d, nonzero values=%d, max_abs=%.6e\n",
        csr_structure.nnz,
        num_nonzero,
        max_val);
    fflush(stderr);
}

// ============================================================================
// Energy Computation
// ============================================================================

__global__ void compute_element_energy_kernel(
    const float* x_curr,
    const unsigned* adjacency,
    const unsigned* offsets,
    const int* element_to_vertex,
    const int* element_to_local_face,
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

    // Use precomputed mapping
    int v_apex = element_to_vertex[elem_idx];
    int local_face_idx = element_to_local_face[elem_idx];

    unsigned offset = offsets[v_apex];
    int tet[4];
    tet[0] = v_apex;
    tet[1] = adjacency[offset + 1 + local_face_idx * 3 + 0];
    tet[2] = adjacency[offset + 1 + local_face_idx * 3 + 1];
    tet[3] = adjacency[offset + 1 + local_face_idx * 3 + 2];

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
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
    cuda::CUDALinearBufferHandle element_to_vertex,
    cuda::CUDALinearBufferHandle element_to_local_face,
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

    // Compute inertial energy: 1/2 * M * (x - x_tilde)² (like mass-spring)
    cuda::GPUParallelFor(
        "compute_inertial_energy_nh", n, [=] __device__(int i) {
            float diff = x_ptr[i] - x_tilde_ptr[i];
            inertial_ptr[i] = 0.5f * M_ptr[i] * diff * diff;
        });

    thrust::device_ptr<float> d_inertial_thrust(inertial_ptr);
    float E_inertial = thrust::reduce(
        thrust::device, d_inertial_thrust, d_inertial_thrust + n);

    // Compute elastic energy
    cudaMemset(element_energy_ptr, 0, num_elements * sizeof(float));

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    compute_element_energy_kernel<<<num_blocks, block_size>>>(
        x_ptr,
        adjacency->get_device_ptr<unsigned>(),
        offsets->get_device_ptr<unsigned>(),
        element_to_vertex->get_device_ptr<int>(),
        element_to_local_face->get_device_ptr<int>(),
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>(),
        mu,
        lambda,
        num_elements,
        element_energy_ptr);

    cudaDeviceSynchronize();

    thrust::device_ptr<float> d_element_thrust(element_energy_ptr);
    float E_elastic = thrust::reduce(
        thrust::device, d_element_thrust, d_element_thrust + num_elements);

    // Compute potential energy: -dt² * f^T * x (like mass-spring)
    cuda::GPUParallelFor(
        "compute_potential_energy_nh", n, [=] __device__(int i) {
            potential_ptr[i] = -dt * dt * x_ptr[i] * f_ptr[i];
        });

    thrust::device_ptr<float> d_potential_thrust(potential_ptr);
    float E_potential = thrust::reduce(
        thrust::device, d_potential_thrust, d_potential_thrust + n);

    // Total energy (like mass-spring): E = 1/2*M*(x-x_tilde)² + dt²*Ψ(x) -
    // dt²*f^T*x
    float total_energy = E_inertial + dt * dt * E_elastic + E_potential;

    return total_energy;
}

// ============================================================================
// Reference Data Computation
// ============================================================================

__global__ void compute_Dm_inv_kernel(
    const float* positions,
    const unsigned* adjacency,
    const unsigned* offsets,
    const int* element_to_vertex,
    const int* element_to_local_face,
    int num_elements,
    float* Dm_inv,
    float* volumes)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements)
        return;

    // Use precomputed mapping
    int v_apex = element_to_vertex[elem_idx];
    int local_face_idx = element_to_local_face[elem_idx];

    // Extract face vertices
    unsigned offset = offsets[v_apex];
    unsigned face_a = adjacency[offset + 1 + local_face_idx * 3 + 0];
    unsigned face_b = adjacency[offset + 1 + local_face_idx * 3 + 1];
    unsigned face_c = adjacency[offset + 1 + local_face_idx * 3 + 2];

    // Tetrahedron vertices: apex + face (v_apex, face_a, face_b, face_c)
    int tet[4] = { v_apex, (int)face_a, (int)face_b, (int)face_c };

    // Get rest positions
    Eigen::Vector3f x0(
        positions[tet[0] * 3 + 0],
        positions[tet[0] * 3 + 1],
        positions[tet[0] * 3 + 2]);
    Eigen::Vector3f x1(
        positions[tet[1] * 3 + 0],
        positions[tet[1] * 3 + 1],
        positions[tet[1] * 3 + 2]);
    Eigen::Vector3f x2(
        positions[tet[2] * 3 + 0],
        positions[tet[2] * 3 + 1],
        positions[tet[2] * 3 + 2]);
    Eigen::Vector3f x3(
        positions[tet[3] * 3 + 0],
        positions[tet[3] * 3 + 1],
        positions[tet[3] * 3 + 2]);

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

    // Store in column-major order (Eigen's default, matches loading format)
    float* Dm_inv_local = &Dm_inv[elem_idx * 9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Dm_inv_local[j * 3 + i] = Dm_inv_mat(i, j);  // Column-major
        }
    }
}

std::tuple<
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle>
compute_reference_data_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
    int num_elements)
{
    // num_elements is the unique number of tetrahedra (total_face_counts / 4)
    // We need to select ONE representative element per tetrahedron
    // Strategy: For each tetrahedron, use the vertex with MINIMUM index as apex

    auto Dm_inv = cuda::create_cuda_linear_buffer<float>(num_elements * 9);
    auto volumes = cuda::create_cuda_linear_buffer<float>(num_elements);
    auto element_to_vertex = cuda::create_cuda_linear_buffer<int>(num_elements);
    auto element_to_local_face =
        cuda::create_cuda_linear_buffer<int>(num_elements);

    int block_size = 256;
    int num_vertices = offsets->getDesc().element_count;

    const unsigned* adj_ptr = adjacency->get_device_ptr<unsigned>();
    const unsigned* off_ptr = offsets->get_device_ptr<unsigned>();
    int* elem_to_v_ptr = element_to_vertex->get_device_ptr<int>();
    int* elem_to_lf_ptr = element_to_local_face->get_device_ptr<int>();

    // Atomic counter for unique elements
    auto element_counter = cuda::create_cuda_linear_buffer<int>(1);
    cudaMemset(
        reinterpret_cast<void*>(element_counter->get_device_ptr()),
        0,
        sizeof(int));
    int* counter_ptr = element_counter->get_device_ptr<int>();

    // For each vertex, check all its opposite faces
    // Only create element if this vertex has the MINIMUM index among the 4
    // vertices
    cuda::GPUParallelFor(
        "select_unique_elements", num_vertices, [=] __device__(int v) {
            unsigned offset = off_ptr[v];
            unsigned count = adj_ptr[offset];

            for (unsigned local_idx = 0; local_idx < count; local_idx++) {
                unsigned v0 = adj_ptr[offset + 1 + local_idx * 3 + 0];
                unsigned v1 = adj_ptr[offset + 1 + local_idx * 3 + 1];
                unsigned v2 = adj_ptr[offset + 1 + local_idx * 3 + 2];

                // Check if v is the minimum among {v, v0, v1, v2}
                bool is_min = (v < v0) && (v < v1) && (v < v2);

                if (is_min) {
                    // This is the unique representative for this tetrahedron
                    int elem_idx = atomicAdd(counter_ptr, 1);
                    elem_to_v_ptr[elem_idx] = v;
                    elem_to_lf_ptr[elem_idx] = local_idx;
                }
            }
        });
    cudaDeviceSynchronize();

    // Verify we got exactly num_elements
    int actual_count = 0;
    cudaMemcpy(&actual_count, counter_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (actual_count != num_elements) {
        printf(
            "[NeoHookean ERROR] Expected %d elements but got %d unique "
            "elements!\n",
            num_elements,
            actual_count);
    }

    // Then compute Dm_inv using the selected elements
    int num_blocks = (num_elements + block_size - 1) / block_size;

    compute_Dm_inv_kernel<<<num_blocks, block_size>>>(
        positions->get_device_ptr<float>(),
        adjacency->get_device_ptr<unsigned>(),
        offsets->get_device_ptr<unsigned>(),
        element_to_vertex->get_device_ptr<int>(),
        element_to_local_face->get_device_ptr<int>(),
        num_elements,
        Dm_inv->get_device_ptr<float>(),
        volumes->get_device_ptr<float>());

    cudaDeviceSynchronize();

    return std::make_tuple(
        Dm_inv, volumes, element_to_vertex, element_to_local_face);
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

    cuda::GPUParallelFor(
        "negate_nh", size, [=] __device__(int i) { out_ptr[i] = -in_ptr[i]; });
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
    __device__ float operator()(float x) const
    {
        return x * x;
    }
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
