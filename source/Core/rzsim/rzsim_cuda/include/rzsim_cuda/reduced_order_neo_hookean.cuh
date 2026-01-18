#pragma once

#include <RHI/internal/cuda_extension.hpp>

#include "api.h"
#include "neo_hookean.cuh"

// Note: We cannot use std::vector or Eigen types directly in CUDA headers
// due to device code compilation issues. Instead we pass data through handles.

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Forward declaration
class VolumeAdjacencyMap;

// Reduced Order Model structures
// Each basis has 12 parameters: 9 for rotation matrix + 3 for translation
// Each vertex has a weight (from eigenvector) that controls how much this basis
// affects it Final displacement: x = rest + Σ_i weight[v,i] * (R_i * rest + t_i
// - rest)

struct ReducedOrderData {
    cuda::CUDALinearBufferHandle
        basis_weights;  // [num_particles * num_basis] - eigenvector weights
    cuda::CUDALinearBufferHandle
        rest_positions;  // [num_particles * 3] - rest positions
    int num_basis;
    int num_particles;
};

// Build reduced order data from Eigen basis vectors
// This function is implemented in .cu file and uses std::vector internally
// But the header avoids std:: types to prevent CUDA compilation issues
RZSIM_CUDA_API
ReducedOrderData build_reduced_order_data_gpu(
    const void* basis_data,  // Pointer to std::vector<Eigen::VectorXf>
    const void* rest_positions_data);  // Pointer to std::vector<glm::vec3>

// Map from reduced coordinates q [num_basis * 12] to full space positions x
// [num_particles * 3] Each basis i contributes: weight[v,i] * (R_i * rest[v] +
// t_i - rest[v]) where q[i*12..i*12+11] stores [R_i (9 params), t_i (3 params)]
RZSIM_CUDA_API
void map_reduced_to_full_gpu(
    cuda::CUDALinearBufferHandle q_reduced,  // [num_basis * 12]
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle x_full);  // [num_particles * 3]

// Compute Jacobian matrix J = dx/dq
// J[3v+d, 12i+p] = weight[v,i] * ∂(R_i*rest[v] + t_i)[d] / ∂q[12i+p]
RZSIM_CUDA_API
void compute_jacobian_gpu(
    cuda::CUDALinearBufferHandle q_reduced,  // [num_basis * 12]
    const ReducedOrderData& ro_data,
    cuda::CUDALinearBufferHandle
        jacobian);  // [num_particles * 3, num_basis * 12]

// Compute reduced gradient: grad_q = J^T * grad_x
// Where grad_x is the full-space gradient [num_particles * 3]
RZSIM_CUDA_API
void compute_reduced_gradient_gpu(
    cuda::CUDALinearBufferHandle
        jacobian,                         // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle grad_x,  // [num_particles * 3]
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle grad_q);  // [num_basis * 12]

// Compute reduced negative gradient: neg_grad_q = -J^T * grad_x
// Same as compute_reduced_gradient_gpu but directly computes negated result
// This avoids an extra negate kernel launch
RZSIM_CUDA_API
void compute_reduced_neg_gradient_gpu(
    cuda::CUDALinearBufferHandle
        jacobian,                         // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle grad_x,  // [num_particles * 3]
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle neg_grad_q);  // [num_basis * 12]

// Map reduced velocities to full space: v_full = J * q_dot
// This is the forward mapping for velocities (inverse of
// compute_reduced_gradient for velocities)
RZSIM_CUDA_API
void map_reduced_velocities_to_full_gpu(
    cuda::CUDALinearBufferHandle
        jacobian,                        // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle q_dot,  // [num_basis * 12]
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle v_full);  // [num_particles * 3]

// Compute reduced Hessian: H_q = J^T * H_x * J
// H_x is sparse CSR matrix [num_particles * 3, num_particles * 3]
// Result is dense matrix [num_basis * 12, num_basis * 12]
// We use: H_q = J^T * (H_x * J)
// First compute temp = H_x * J [num_particles * 3, num_basis * 12]
// Then compute H_q = J^T * temp [num_basis * 12, num_basis * 12]
RZSIM_CUDA_API
void compute_reduced_hessian_gpu(
    const NeoHookeanCSRStructure& hessian_structure,
    cuda::CUDALinearBufferHandle hessian_values,  // CSR values [nnz]
    cuda::CUDALinearBufferHandle
        jacobian,  // [num_particles * 3, num_basis * 12]
    int num_particles,
    int num_basis,
    cuda::CUDALinearBufferHandle
        temp_buffer,                    // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle H_q);  // [num_basis * 12, num_basis * 12]

// Utilities for reduced coordinates

// Initialize reduced coordinates to identity transforms (rest pose)
// For each basis: R = I (identity matrix), t = 0
RZSIM_CUDA_API
void initialize_reduced_coords_identity_gpu(
    int num_basis,
    cuda::CUDALinearBufferHandle q);  // [num_basis * 12]

// Explicit step in reduced space: q_tilde = q + dt * q_dot
RZSIM_CUDA_API
void explicit_step_reduced_gpu(
    cuda::CUDALinearBufferHandle q,      // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_dot,  // [num_basis * 12]
    float dt,
    int num_basis,
    cuda::CUDALinearBufferHandle q_tilde);  // [num_basis * 12]

// Update reduced velocities: q_dot = (q_new - q_old) / dt * damping
RZSIM_CUDA_API
void update_reduced_velocities_gpu(
    cuda::CUDALinearBufferHandle q_new,  // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_old,  // [num_basis * 12]
    float dt,
    float damping,
    int num_basis,
    cuda::CUDALinearBufferHandle q_dot);  // [num_basis * 12]

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
