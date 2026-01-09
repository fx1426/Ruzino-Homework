#pragma once

#include <RHI/cuda.hpp>

#include "RHI/internal/cuda_extension.hpp"
#include "adjacency_map.cuh"
#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Pre-built CSR structure with position mapping for fast value updates
struct CSRStructure {
    cuda::CUDALinearBufferHandle row_offsets;
    cuda::CUDALinearBufferHandle col_indices;
    // For each spring, stores where its 36 Hessian entries go in the values array
    cuda::CUDALinearBufferHandle spring_value_positions;  // [num_springs * 36]
    // For each particle DOF, stores where its mass diagonal entry goes
    cuda::CUDALinearBufferHandle mass_value_positions;    // [num_particles * 3]
    int num_rows;
    int num_cols;
    int nnz;
};

RZSIM_CUDA_API
void explicit_step_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde);

RZSIM_CUDA_API
void setup_external_forces_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext);

RZSIM_CUDA_API
cuda::CUDALinearBufferHandle compute_rest_lengths_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle springs);

RZSIM_CUDA_API
void compute_gradient_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle grad);

// Build CSR structure once during initialization (sparsity pattern only)
RZSIM_CUDA_API
CSRStructure build_hessian_structure_gpu(
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    int num_particles);

// Fast update: directly fill values into pre-built CSR structure (NO SORTING)
RZSIM_CUDA_API
void update_hessian_values_gpu(
    const CSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle values);  // Output values array

RZSIM_CUDA_API
float compute_energy_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle adjacent_vertices,
    cuda::CUDALinearBufferHandle vertex_offsets,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle inertial_terms,
    cuda::CUDALinearBufferHandle spring_energies,
    cuda::CUDALinearBufferHandle potential_terms);

// GPU vector operations to avoid CPU-GPU transfers
RZSIM_CUDA_API
float compute_vector_norm_gpu(
    cuda::CUDALinearBufferHandle vec,
    int size);

RZSIM_CUDA_API
float compute_dot_product_gpu(
    cuda::CUDALinearBufferHandle vec1,
    cuda::CUDALinearBufferHandle vec2,
    int size);

RZSIM_CUDA_API
void axpy_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size);

RZSIM_CUDA_API
void negate_gpu(
    cuda::CUDALinearBufferHandle in,
    cuda::CUDALinearBufferHandle out,
    int size);

RZSIM_CUDA_API
void project_to_ground_gpu(
    cuda::CUDALinearBufferHandle positions,
    int num_particles,
    float ground_height);

RZSIM_CUDA_API
void compute_normals_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle face_vertex_indices,
    cuda::CUDALinearBufferHandle face_counts,
    bool flip_normal,
    cuda::CUDALinearBufferHandle normals);

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
