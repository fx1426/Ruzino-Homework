#pragma once

#include <RHI/internal/cuda_extension.hpp>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Forward declaration
class VolumeAdjacencyMap;

// CSR structure for sparse Hessian matrix (same as mass-spring)
struct NeoHookeanCSRStructure {
    cuda::CUDALinearBufferHandle row_offsets;  // size: (n+1)
    cuda::CUDALinearBufferHandle col_indices;  // size: nnz
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

// Compute lumped mass matrix from density and element volumes
RZSIM_CUDA_API
void compute_lumped_mass_matrix_gpu(
    const VolumeAdjacencyMap& volume_adjacency,
    cuda::CUDALinearBufferHandle volumes,
    float density,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle mass_matrix);

// Setup external forces using FEM integration
RZSIM_CUDA_API
void setup_external_forces_fem_gpu(
    const VolumeAdjacencyMap& volume_adjacency,
    cuda::CUDALinearBufferHandle volumes,
    float density,
    float gravity,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle f_ext);

// Setup external forces (gravity) - old per-vertex method
RZSIM_CUDA_API
void setup_external_forces_nh_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext);

// Compute negative gradient of Neo-Hookean energy (for Newton's method)
RZSIM_CUDA_API
void compute_neg_gradient_nh_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    const VolumeAdjacencyMap& volume_adjacency,
    cuda::CUDALinearBufferHandle Dm_inv,   // [9 * num_elements] inverse
                                           // reference shape matrices
    cuda::CUDALinearBufferHandle volumes,  // [num_elements] rest volumes
    float mu,                              // Lamé parameter
    float lambda,                          // Lamé parameter
    float density,                         // Material density
    float gravity,                         // Gravity magnitude
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle neg_grad);  // Output: negative gradient

// Build CSR sparsity pattern once during initialization
RZSIM_CUDA_API
NeoHookeanCSRStructure build_hessian_structure_nh_gpu(
    const VolumeAdjacencyMap& volume_adjacency,
    int num_particles,
    int num_elements);

// Update Hessian values (fast, no sorting needed)
RZSIM_CUDA_API
void update_hessian_values_nh_gpu(
    const NeoHookeanCSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    const VolumeAdjacencyMap& volume_adjacency,
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
    const VolumeAdjacencyMap& volume_adjacency,
    cuda::CUDALinearBufferHandle Dm_inv,
    cuda::CUDALinearBufferHandle volumes,
    float mu,
    float lambda,
    float density,
    float gravity,
    float dt,
    int num_particles,
    int num_elements,
    cuda::CUDALinearBufferHandle inertial_terms,
    cuda::CUDALinearBufferHandle element_energies);

// Compute reference shape matrices Dm and their inverses for all tetrahedra
// Returns: (Dm_inv, volumes, element_to_vertex, element_to_local_face)
// Compute Neo-Hookean reference data (Dm_inv and volumes)
// Requires element_to_vertex and element_to_local_face from VolumeAdjacencyMap
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_nh_reference_data_gpu(
    cuda::CUDALinearBufferHandle positions,
    const VolumeAdjacencyMap& volume_adjacency,
    int num_elements);

// Legacy function - computes both topology and reference data
RZSIM_CUDA_API
std::tuple<
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle,
    cuda::CUDALinearBufferHandle>
compute_reference_data_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle adjacency,
    cuda::CUDALinearBufferHandle offsets,
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
void scale_vector_gpu(cuda::CUDALinearBufferHandle vec, float scale, int size);

RZSIM_CUDA_API
float compute_vector_norm_nh_gpu(cuda::CUDALinearBufferHandle vec, int size);

// Update velocities and apply damping: v = (x_new - x_old) / dt * damping
RZSIM_CUDA_API
void update_velocities_nh_gpu(
    cuda::CUDALinearBufferHandle x_new,
    cuda::CUDALinearBufferHandle x_old,
    float dt,
    float damping,
    int num_particles,
    cuda::CUDALinearBufferHandle velocities);

// Handle ground collision with restitution
RZSIM_CUDA_API
void handle_ground_collision_nh_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle velocities,
    float restitution,
    int num_particles);

// Apply Dirichlet boundary conditions to Hessian matrix (CSR format)
// Sets rows and columns of BC DOFs appropriately
RZSIM_CUDA_API
void apply_dirichlet_bc_to_hessian_gpu(
    const NeoHookeanCSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle bc_dofs,  // [num_bc_dofs] DOF indices with BC
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle values);  // Modified in-place

// Apply Dirichlet boundary conditions to gradient vector
// Sets gradient to zero for BC DOFs
RZSIM_CUDA_API
void apply_dirichlet_bc_to_gradient_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,  // [num_bc_dofs] DOF indices with BC
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle gradient);  // Modified in-place

// Zero out Newton direction for BC DOFs
RZSIM_CUDA_API
void apply_dirichlet_bc_to_direction_gpu(
    cuda::CUDALinearBufferHandle bc_dofs,
    int num_bc_dofs,
    cuda::CUDALinearBufferHandle direction);

// Add values to Hessian diagonal elements (CSR format)
// Used for adding barrier function Hessian contributions
RZSIM_CUDA_API
void add_to_hessian_diagonal_gpu(
    const NeoHookeanCSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle diagonal_additions,  // [num_dof] values to add
    int num_dof,
    cuda::CUDALinearBufferHandle values);  // Modified in-place

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
