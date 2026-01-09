#pragma once

#include <RHI/internal/cuda_extension.hpp>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// CSR structure for sparse Hessian matrix (same as mass-spring)
struct NeoHookeanCSRStructure {
    cuda::CUDALinearBufferHandle row_offsets;   // size: (n+1)
    cuda::CUDALinearBufferHandle col_indices;   // size: nnz
    cuda::CUDALinearBufferHandle
        mass_value_positions;  // size: n (positions of diagonal mass entries)
    cuda::CUDALinearBufferHandle element_value_positions;  // size:
                                                           // num_elements * 144
                                                           // (12x12 per tet)
    int num_rows;
    int num_cols;
    int nnz;
};

// Explicit Euler step: x_tilde = x + dt * v
RZSIM_CUDA_API
void explicit_step_nh_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde);

// Setup external forces (gravity)
RZSIM_CUDA_API
void setup_external_forces_nh_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext);

// Compute gradient of Neo-Hookean energy
RZSIM_CUDA_API
void compute_gradient_nh_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle tetrahedra,  // [4 * num_elements] vertex
                                              // indices
    cuda::CUDALinearBufferHandle Dm_inv,      // [9 * num_elements] inverse
                                              // reference shape matrices
    cuda::CUDALinearBufferHandle volumes,     // [num_elements] rest volumes
    float mu,                                 // Lamé parameter
    float lambda,                             // Lamé parameter
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle grad);

// Build CSR sparsity pattern once during initialization
RZSIM_CUDA_API
NeoHookeanCSRStructure build_hessian_structure_nh_gpu(
    cuda::CUDALinearBufferHandle tetrahedra,
    int num_particles,
    int num_elements);

// Update Hessian values (fast, no sorting needed)
RZSIM_CUDA_API
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
    cuda::CUDALinearBufferHandle values);

// Compute total Neo-Hookean energy
RZSIM_CUDA_API
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
    cuda::CUDALinearBufferHandle inertial_terms,
    cuda::CUDALinearBufferHandle element_energies,
    cuda::CUDALinearBufferHandle potential_terms);

// Compute reference shape matrices Dm and their inverses for all tetrahedra
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_reference_data_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle tetrahedra,
    int num_elements);

// Vector operations (reuse from mass-spring where appropriate)
RZSIM_CUDA_API
void negate_nh_gpu(
    cuda::CUDALinearBufferHandle input,
    cuda::CUDALinearBufferHandle output,
    int size);

RZSIM_CUDA_API
void axpy_nh_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size);

RZSIM_CUDA_API
float compute_vector_norm_nh_gpu(cuda::CUDALinearBufferHandle vec, int size);

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
