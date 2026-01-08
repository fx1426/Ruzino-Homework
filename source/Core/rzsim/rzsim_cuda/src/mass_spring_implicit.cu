#include <cusparse.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <RHI/cuda.hpp>
#include <cstddef>
#include <map>
#include <set>

#include "RZSolver/Solver.hpp"
#include "rzsim_cuda/mass_spring_implicit.cuh"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Kernel to extract edges from triangles
__global__ void
extract_edges_kernel(const int* triangles, int num_triangles, int* edge_pairs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triangles)
        return;

    int base_idx = tid * 3;
    int v0 = triangles[base_idx];
    int v1 = triangles[base_idx + 1];
    int v2 = triangles[base_idx + 2];

    // Each triangle produces 3 edges
    int output_base = tid * 6;

    // Edge 0-1
    edge_pairs[output_base + 0] = min(v0, v1);
    edge_pairs[output_base + 1] = max(v0, v1);

    // Edge 1-2
    edge_pairs[output_base + 2] = min(v1, v2);
    edge_pairs[output_base + 3] = max(v1, v2);

    // Edge 2-0
    edge_pairs[output_base + 4] = min(v2, v0);
    edge_pairs[output_base + 5] = max(v2, v0);
}

// Functor for comparing edge pairs
struct EdgePairEqual {
    __host__ __device__ bool operator()(
        const thrust::tuple<int, int>& a,
        const thrust::tuple<int, int>& b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    }
};

// Kernel to separate interleaved edges
__global__ void separate_edges_kernel(
    const int* interleaved,
    int* first,
    int* second,
    int num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges)
        return;

    first[tid] = interleaved[tid * 2];
    second[tid] = interleaved[tid * 2 + 1];
}

// Kernel to copy separated edges back to interleaved format
__global__ void interleave_edges_kernel(
    const int* edge_first,
    const int* edge_second,
    int* output,
    int num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges)
        return;

    output[tid * 2] = edge_first[tid];
    output[tid * 2 + 1] = edge_second[tid];
}

cuda::CUDALinearBufferHandle build_edge_set_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle edges)
{
    // Get triangle count
    size_t num_triangles = edges->getDesc().element_count / 3;

    printf("[GPU] Building edge set from %zu triangles\n", num_triangles);

    // Allocate temporary buffer for all edges (3 edges per triangle)
    thrust::device_vector<int> all_edges(num_triangles * 6);

    // Launch kernel to extract edges
    int block_size = 256;
    int num_blocks = (num_triangles + block_size - 1) / block_size;

    extract_edges_kernel<<<num_blocks, block_size>>>(
        edges->get_device_ptr<int>(),
        num_triangles,
        thrust::raw_pointer_cast(all_edges.data()));

    cudaDeviceSynchronize();

    // Create vectors for edge pairs
    thrust::device_vector<int> edge_first(num_triangles * 3);
    thrust::device_vector<int> edge_second(num_triangles * 3);

    // Separate the interleaved edge data using kernel
    int sep_blocks = (num_triangles * 3 + block_size - 1) / block_size;
    separate_edges_kernel<<<sep_blocks, block_size>>>(
        thrust::raw_pointer_cast(all_edges.data()),
        thrust::raw_pointer_cast(edge_first.data()),
        thrust::raw_pointer_cast(edge_second.data()),
        num_triangles * 3);

    cudaDeviceSynchronize();

    // Create zip iterator
    auto edge_begin = thrust::make_zip_iterator(
        thrust::make_tuple(edge_first.begin(), edge_second.begin()));
    auto edge_end = thrust::make_zip_iterator(
        thrust::make_tuple(edge_first.end(), edge_second.end()));

    // Sort edges
    thrust::sort(
        edge_begin,
        edge_end,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            return thrust::get<1>(a) < thrust::get<1>(b);
        });

    // Remove duplicates
    auto new_end = thrust::unique(edge_begin, edge_end, EdgePairEqual());

    // Calculate unique edge count
    size_t num_unique_edges = new_end - edge_begin;

    printf("[GPU] Found %zu unique edges\n", num_unique_edges);

    // Copy unique edges to output buffer (interleaved format)
    auto output_buffer =
        cuda::create_cuda_linear_buffer<int>(size_t(num_unique_edges * 2));

    int* output_ptr = output_buffer->get_device_ptr<int>();

    // Use kernel to interleave the data
    int copy_blocks = (num_unique_edges + block_size - 1) / block_size;
    interleave_edges_kernel<<<copy_blocks, block_size>>>(
        thrust::raw_pointer_cast(edge_first.data()),
        thrust::raw_pointer_cast(edge_second.data()),
        output_ptr,
        num_unique_edges);

    cudaDeviceSynchronize();

    return output_buffer;
}

// Kernel to compute explicit step: x_tilde = x + dt * v
__global__ void explicit_step_kernel(
    const float* x,
    const float* v,
    float dt,
    int num_particles,
    float* x_tilde)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles)
        return;

    x_tilde[tid * 3 + 0] = x[tid * 3 + 0] + dt * v[tid * 3 + 0];
    x_tilde[tid * 3 + 1] = x[tid * 3 + 1] + dt * v[tid * 3 + 1];
    x_tilde[tid * 3 + 2] = x[tid * 3 + 2] + dt * v[tid * 3 + 2];
}

// Kernel to setup external forces (gravity)
__global__ void setup_external_forces_kernel(
    float mass,
    float gravity,
    int num_particles,
    float* f_ext)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles)
        return;

    f_ext[tid * 3 + 0] = 0.0f;
    f_ext[tid * 3 + 1] = 0.0f;
    f_ext[tid * 3 + 2] = mass * gravity;  // gravity in z direction
}

// Kernel to compute rest lengths from initial positions
__global__ void compute_rest_lengths_kernel(
    const float* positions,
    const int* springs,
    float* rest_lengths,
    int num_springs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_springs)
        return;

    int i = springs[tid * 2];
    int j = springs[tid * 2 + 1];

    float dx = positions[i * 3 + 0] - positions[j * 3 + 0];
    float dy = positions[i * 3 + 1] - positions[j * 3 + 1];
    float dz = positions[i * 3 + 2] - positions[j * 3 + 2];

    rest_lengths[tid] = sqrtf(dx * dx + dy * dy + dz * dz);
}

void explicit_step_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde)
{
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    explicit_step_kernel<<<num_blocks, block_size>>>(
        x->get_device_ptr<float>(),
        v->get_device_ptr<float>(),
        dt,
        num_particles,
        x_tilde->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

void setup_external_forces_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext)
{
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    setup_external_forces_kernel<<<num_blocks, block_size>>>(
        mass, gravity, num_particles, f_ext->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

cuda::CUDALinearBufferHandle compute_rest_lengths_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle springs)
{
    size_t num_springs = springs->getDesc().element_count / 2;

    auto rest_lengths_buffer =
        cuda::create_cuda_linear_buffer<float>(num_springs);

    int block_size = 256;
    int num_blocks = (num_springs + block_size - 1) / block_size;

    compute_rest_lengths_kernel<<<num_blocks, block_size>>>(
        positions->get_device_ptr<float>(),
        springs->get_device_ptr<int>(),
        rest_lengths_buffer->get_device_ptr<float>(),
        num_springs);

    cudaDeviceSynchronize();

    return rest_lengths_buffer;
}

// Kernel to compute gradient
__global__ void compute_gradient_kernel(
    const float* x_curr,
    const float* x_tilde,
    const float* M_diag,
    const float* f_ext,
    const int* springs,
    const float* rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    int num_springs,
    float* grad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles)
        return;

    // Initialize with inertial term: M * (x - x_tilde)
    grad[tid * 3 + 0] =
        M_diag[tid * 3 + 0] * (x_curr[tid * 3 + 0] - x_tilde[tid * 3 + 0]);
    grad[tid * 3 + 1] =
        M_diag[tid * 3 + 1] * (x_curr[tid * 3 + 1] - x_tilde[tid * 3 + 1]);
    grad[tid * 3 + 2] =
        M_diag[tid * 3 + 2] * (x_curr[tid * 3 + 2] - x_tilde[tid * 3 + 2]);

    // Add spring forces
    for (int s = 0; s < num_springs; ++s) {
        int i = springs[s * 2];
        int j = springs[s * 2 + 1];

        if (i != tid && j != tid)
            continue;

        float l0 = rest_lengths[s];
        float l0_sq = l0 * l0;

        float dx = x_curr[i * 3 + 0] - x_curr[j * 3 + 0];
        float dy = x_curr[i * 3 + 1] - x_curr[j * 3 + 1];
        float dz = x_curr[i * 3 + 2] - x_curr[j * 3 + 2];
        float diff_sq = dx * dx + dy * dy + dz * dz;

        float factor = 2.0f * stiffness * (diff_sq / l0_sq - 1.0f) * dt * dt;

        if (i == tid) {
            grad[tid * 3 + 0] += factor * dx;
            grad[tid * 3 + 1] += factor * dy;
            grad[tid * 3 + 2] += factor * dz;
        }
        else {  // j == tid
            grad[tid * 3 + 0] -= factor * dx;
            grad[tid * 3 + 1] -= factor * dy;
            grad[tid * 3 + 2] -= factor * dz;
        }
    }

    // Subtract external forces
    grad[tid * 3 + 0] -= dt * dt * f_ext[tid * 3 + 0];
    grad[tid * 3 + 1] -= dt * dt * f_ext[tid * 3 + 1];
    grad[tid * 3 + 2] -= dt * dt * f_ext[tid * 3 + 2];
}

void compute_gradient_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle grad)
{
    int num_springs = springs->getDesc().element_count / 2;

    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    compute_gradient_kernel<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        x_tilde->get_device_ptr<float>(),
        M_diag->get_device_ptr<float>(),
        f_ext->get_device_ptr<float>(),
        springs->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        stiffness,
        dt,
        num_particles,
        num_springs,
        grad->get_device_ptr<float>());

    cudaDeviceSynchronize();

    // Debug: print first few gradient values
    std::vector<float> grad_host = grad->get_host_vector<float>();
    for (int i = 0; i < std::min(10, (int)grad_host.size()); i++) {
        printf("  grad[%d] = %.6e\n", i, grad_host[i]);
    }
}

// Custom 3x3 symmetric eigenvalue decomposition using Jacobi rotation
// This is needed because Eigen's SelfAdjointEigenSolver doesn't work on CUDA device
__device__ void eigen_decomposition_3x3(
    const Eigen::Matrix3f& A,
    Eigen::Vector3f& eigenvalues,
    Eigen::Matrix3f& eigenvectors)
{
    const int MAX_ITER = 50;
    const float EPSILON = 1e-10f;
    
    // Initialize eigenvectors as identity
    eigenvectors.setIdentity();
    
    // Copy A to working matrix
    Eigen::Matrix3f a = A;
    
    // Jacobi rotation
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest off-diagonal element
        float max_offdiag = 0.0f;
        int p = 0, q = 1;
        
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float abs_aij = fabsf(a(i, j));
                if (abs_aij > max_offdiag) {
                    max_offdiag = abs_aij;
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check convergence
        if (max_offdiag < EPSILON) {
            break;
        }
        
        // Compute rotation angle
        float diff = a(q, q) - a(p, p);
        float theta = 0.5f * atan2f(2.0f * a(p, q), diff);
        float c = cosf(theta);
        float s = sinf(theta);
        
        // Apply rotation to a: a = J^T * a * J
        Eigen::Matrix3f J;
        J.setIdentity();
        J(p, p) = c;
        J(q, q) = c;
        J(p, q) = s;
        J(q, p) = -s;
        
        a = J.transpose() * a * J;
        
        // Accumulate eigenvectors
        eigenvectors = eigenvectors * J;
    }
    
    // Extract eigenvalues from diagonal
    eigenvalues(0) = a(0, 0);
    eigenvalues(1) = a(1, 1);
    eigenvalues(2) = a(2, 2);
}

// Custom PSD projection for 3x3 symmetric matrix
__device__ Eigen::Matrix3f project_psd_custom(const Eigen::Matrix3f& H, int debug_sid = -1)
{
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    
    // Compute eigendecomposition
    eigen_decomposition_3x3(H, eigenvalues, eigenvectors);
    
    if (debug_sid == 0) {
        printf("[GPU PSD Custom] Input matrix H:\n");
        printf("  [%.6e %.6e %.6e]\n", H(0,0), H(0,1), H(0,2));
        printf("  [%.6e %.6e %.6e]\n", H(1,0), H(1,1), H(1,2));
        printf("  [%.6e %.6e %.6e]\n", H(2,0), H(2,1), H(2,2));
        printf("[GPU PSD Custom] Eigenvalues: [%.6e, %.6e, %.6e]\n", 
               eigenvalues(0), eigenvalues(1), eigenvalues(2));
    }
    
    // Clamp negative eigenvalues to zero
    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < 0.0f) {
            eigenvalues(i) = 0.0f;
        }
    }
    
    if (debug_sid == 0) {
        printf("[GPU PSD Custom] After clamping: [%.6e, %.6e, %.6e]\n", 
               eigenvalues(0), eigenvalues(1), eigenvalues(2));
    }
    
    // Reconstruct: H_psd = V * Lambda * V^T
    Eigen::Matrix3f result = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    
    if (debug_sid == 0) {
        printf("[GPU PSD Custom] Output matrix:\n");
        printf("  [%.6e %.6e %.6e]\n", result(0,0), result(0,1), result(0,2));
        printf("  [%.6e %.6e %.6e]\n", result(1,0), result(1,1), result(1,2));
        printf("  [%.6e %.6e %.6e]\n", result(2,0), result(2,1), result(2,2));
    }
    
    return result;
}

// Test kernel to verify custom eigensolver works on device
__global__ void test_eigen_solver_kernel()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n========== TESTING CUSTOM 3x3 EIGENSOLVER ==========\n");
        
        // Test 1: Diagonal matrix with known eigenvalues
        Eigen::Matrix3f test1;
        test1 << 12000.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f;
        
        printf("[TEST 1] Input diagonal matrix: diag(12000, 0, 0)\n");
        
        Eigen::Vector3f evals1;
        Eigen::Matrix3f evecs1;
        eigen_decomposition_3x3(test1, evals1, evecs1);
        printf("  Computed eigenvalues: [%.6e, %.6e, %.6e]\n", evals1(0), evals1(1), evals1(2));
        printf("  Expected: [0, 0, 12000] (in some order)\n");
        
        // Reconstruct and verify
        Eigen::Matrix3f reconstructed1 = evecs1 * evals1.asDiagonal() * evecs1.transpose();
        printf("  Reconstruction error: %.6e\n", (reconstructed1 - test1).norm());
        
        // Test 2: Symmetric matrix
        Eigen::Matrix3f test2;
        test2 << 4.0f, 1.0f, 0.0f,
                 1.0f, 3.0f, 0.0f,
                 0.0f, 0.0f, 2.0f;
        
        printf("[TEST 2] Input symmetric matrix:\n");
        printf("  [4 1 0; 1 3 0; 0 0 2]\n");
        
        Eigen::Vector3f evals2;
        Eigen::Matrix3f evecs2;
        eigen_decomposition_3x3(test2, evals2, evecs2);
        printf("  Computed eigenvalues: [%.6e, %.6e, %.6e]\n", evals2(0), evals2(1), evals2(2));
        
        Eigen::Matrix3f reconstructed2 = evecs2 * evals2.asDiagonal() * evecs2.transpose();
        printf("  Reconstruction error: %.6e\n", (reconstructed2 - test2).norm());
        
        // Test 3: PSD projection with negative eigenvalue
        Eigen::Matrix3f test3;
        test3 << 2.0f, 1.0f, 0.0f,
                 1.0f, -1.0f, 0.0f,
                 0.0f, 0.0f, 3.0f;
        
        printf("[TEST 3] Matrix with negative eigenvalue:\n");
        Eigen::Matrix3f projected3 = project_psd_custom(test3);
        printf("  Output should be PSD\n");
        
        // Verify it's PSD by checking eigenvalues
        Eigen::Vector3f evals3_check;
        Eigen::Matrix3f evecs3_check;
        eigen_decomposition_3x3(projected3, evals3_check, evecs3_check);
        printf("  Final eigenvalues: [%.6e, %.6e, %.6e]\n", 
               evals3_check(0), evals3_check(1), evals3_check(2));
        printf("  All should be >= 0\n");
        
        printf("========== CUSTOM EIGENSOLVER TEST COMPLETE ==========\n\n");
    }
}

// Kernel to compute 3x3 eigenvalues and eigenvectors using analytical solution
__device__ Eigen::Matrix3f project_psd(const Eigen::Matrix3f& H, int debug_sid = -1)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(H);
    Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors();

    if (debug_sid == 0) {
        printf("[GPU PSD] Input matrix H:\n");
        printf("  [%.6e %.6e %.6e]\n", H(0,0), H(0,1), H(0,2));
        printf("  [%.6e %.6e %.6e]\n", H(1,0), H(1,1), H(1,2));
        printf("  [%.6e %.6e %.6e]\n", H(2,0), H(2,1), H(2,2));
        printf("[GPU PSD] Eigenvalues: [%.6e, %.6e, %.6e]\n", 
               eigenvalues(0), eigenvalues(1), eigenvalues(2));
        printf("[GPU PSD] Solver info: %d\n", (int)eigensolver.info());
    }

    // Clamp negative eigenvalues to zero
    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < 0.0f)
            eigenvalues(i) = 0.0f;
    }

    if (debug_sid == 0) {
        printf("[GPU PSD] After clamping: [%.6e, %.6e, %.6e]\n", 
               eigenvalues(0), eigenvalues(1), eigenvalues(2));
    }

    // Reconstruct: H_psd = V * Lambda * V^T
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

// Kernel to compute triplets for spring Hessian blocks
__global__ void compute_spring_hessian_kernel(
    const float* x_curr,
    const int* springs,
    const float* rest_lengths,
    float stiffness,
    float dt,
    int num_springs,
    int* triplet_rows,
    int* triplet_cols,
    float* triplet_vals,
    int* num_triplets_per_spring)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_springs)
        return;

    int vi = springs[sid * 2];
    int vj = springs[sid * 2 + 1];
    float l0 = rest_lengths[sid];

    if (l0 < 1e-10f) {
        num_triplets_per_spring[sid] = 0;
        return;
    }

    float k = stiffness;
    float l0_sq = l0 * l0;

    // Get positions
    Eigen::Vector3f xi(x_curr[vi * 3], x_curr[vi * 3 + 1], x_curr[vi * 3 + 2]);
    Eigen::Vector3f xj(x_curr[vj * 3], x_curr[vj * 3 + 1], x_curr[vj * 3 + 2]);
    Eigen::Vector3f diff = xi - xj;
    float diff_sq = diff.squaredNorm();

    // H_diff = 2*k/l0^2 * (2*outer(diff,diff) + (diff_sq - l0^2)*I)
    Eigen::Matrix3f outer = diff * diff.transpose();
    Eigen::Matrix3f H_diff =
        2.0f * k / l0_sq *
        (2.0f * outer + (diff_sq - l0_sq) * Eigen::Matrix3f::Identity());

    // Use custom PSD projection (Eigen's solver doesn't work on CUDA device)
    H_diff = project_psd_custom(H_diff, sid);

    // Scale by dt^2
    float scale = dt * dt;
    H_diff *= scale;

    // Write triplets for 6x6 block (4 3x3 blocks)
    int base_idx = sid * 36;  // Each spring contributes 36 triplets (4 blocks * 9 entries)
    int count = 0;

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float val = H_diff(r, c);
            
            // Block (vi, vi)
            triplet_rows[base_idx + count] = vi * 3 + r;
            triplet_cols[base_idx + count] = vi * 3 + c;
            triplet_vals[base_idx + count] = val;
            count++;

            // Block (vi, vj)
            triplet_rows[base_idx + count] = vi * 3 + r;
            triplet_cols[base_idx + count] = vj * 3 + c;
            triplet_vals[base_idx + count] = -val;
            count++;

            // Block (vj, vi)
            triplet_rows[base_idx + count] = vj * 3 + r;
            triplet_cols[base_idx + count] = vi * 3 + c;
            triplet_vals[base_idx + count] = -val;
            count++;

            // Block (vj, vj)
            triplet_rows[base_idx + count] = vj * 3 + r;
            triplet_cols[base_idx + count] = vj * 3 + c;
            triplet_vals[base_idx + count] = val;
            count++;
        }
    }

    num_triplets_per_spring[sid] = count;
}

// Kernel to add mass matrix diagonal
__global__ void add_mass_diagonal_kernel(
    const float* M_diag,
    int num_particles,
    int* triplet_rows,
    int* triplet_cols,
    float* triplet_vals,
    int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles * 3)
        return;

    int particle_id = tid / 3;  // M_diag is per-particle, not per-DOF
    float regularization = 1e-6f;
    triplet_rows[offset + tid] = tid;
    triplet_cols[offset + tid] = tid;
    triplet_vals[offset + tid] = M_diag[particle_id] + regularization;
}

// Kernel to convert COO to CSR format
__global__ void coo_to_csr_kernel(
    const int* row_indices,
    int nnz,
    int num_rows,
    int* row_offsets)
{
    // Initialize row_offsets
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= num_rows) {
        row_offsets[tid] = 0;
    }
    __syncthreads();

    if (tid < nnz) {
        int row = row_indices[tid];
        atomicAdd(&row_offsets[row + 1], 1);
    }
}

// Kernel to compute prefix sum for row offsets
__global__ void prefix_sum_kernel(int* row_offsets, int num_rows)
{
    // Simple sequential prefix sum (for small matrices)
    // For large matrices, use thrust::exclusive_scan
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i <= num_rows; ++i) {
            row_offsets[i] += row_offsets[i - 1];
        }
    }
}

// Functor for sorting triplets by (row, col)
struct TripletComparator {
    __host__ __device__ bool operator()(
        const thrust::tuple<int, int, float>& a,
        const thrust::tuple<int, int, float>& b) const
    {
        int row_a = thrust::get<0>(a);
        int row_b = thrust::get<0>(b);
        if (row_a != row_b)
            return row_a < row_b;
        return thrust::get<1>(a) < thrust::get<1>(b);
    }
};

// Functor for comparing keys in reduce_by_key
struct KeyEqual {
    __host__ __device__ bool operator()(
        const thrust::tuple<int, int>& a,
        const thrust::tuple<int, int>& b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b);
    }
};

// Functor to atomically increment row counts
struct IncrementRowCount {
    int* row_offsets;
    
    IncrementRowCount(int* offsets) : row_offsets(offsets) {}
    
    __device__ void operator()(int row) {
        atomicAdd(&row_offsets[row + 1], 1);
    }
};

CSRMatrix assemble_hessian_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles)
{
    CSRMatrix result;
    int num_springs = springs->getDesc().element_count / 2;
    int n = num_particles * 3;

    printf("[GPU] Assembling Hessian: %d particles, %d springs\n", num_particles, num_springs);
    printf("[GPU] dt = %.6f, stiffness = %.6f\n", dt, stiffness);

    // Run Eigen solver test ONCE
    static bool eigen_test_done = false;
    if (!eigen_test_done) {
        printf("\n========== TESTING EIGEN SOLVER ON DEVICE ==========\n");
        test_eigen_solver_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        printf("========== EIGEN TEST COMPLETE ==========\n\n");
        eigen_test_done = true;
    }

    // Calculate total number of triplets
    // Mass matrix: num_particles * 3 diagonal entries  
    // Spring contributions: num_springs * 36 entries (4 blocks * 9)
    int num_mass_triplets = num_particles * 3;
    size_t max_spring_triplets = (size_t)num_springs * 36;
    size_t max_total_triplets = num_mass_triplets + max_spring_triplets;

    printf("[GPU] Allocating %zu triplets (%d mass + %zu spring)\n", 
           max_total_triplets, num_mass_triplets, max_spring_triplets);

    // Allocate temporary buffers for triplets
    auto d_triplet_rows = cuda::create_cuda_linear_buffer<int>(max_total_triplets);
    auto d_triplet_cols = cuda::create_cuda_linear_buffer<int>(max_total_triplets);
    auto d_triplet_vals = cuda::create_cuda_linear_buffer<float>(max_total_triplets);
    auto d_num_triplets_per_spring = cuda::create_cuda_linear_buffer<int>(num_springs);

    // Compute spring Hessian blocks
    int block_size = 256;
    int num_blocks = (num_springs + block_size - 1) / block_size;

    compute_spring_hessian_kernel<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        springs->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        stiffness,
        dt,
        num_springs,
        d_triplet_rows->get_device_ptr<int>(),
        d_triplet_cols->get_device_ptr<int>(),
        d_triplet_vals->get_device_ptr<float>(),
        d_num_triplets_per_spring->get_device_ptr<int>());

    cudaDeviceSynchronize();

    // Add mass matrix diagonal
    int mass_blocks = (num_mass_triplets + block_size - 1) / block_size;
    add_mass_diagonal_kernel<<<mass_blocks, block_size>>>(
        M_diag->get_device_ptr<float>(),
        num_particles,
        d_triplet_rows->get_device_ptr<int>(),
        d_triplet_cols->get_device_ptr<int>(),
        d_triplet_vals->get_device_ptr<float>(),
        max_spring_triplets);

    cudaDeviceSynchronize();

    // Total number of triplets (all valid ones)
    size_t total_triplets = num_mass_triplets + max_spring_triplets;

    printf("[GPU] Total allocated triplets: %zu\n", total_triplets);

    // Debug: Check mass matrix contribution at offset
    int mass_offset = max_spring_triplets;
    std::vector<int> mass_rows(10);
    std::vector<int> mass_cols(10);
    std::vector<float> mass_vals(10);
    thrust::copy(thrust::device_ptr<int>(d_triplet_rows->get_device_ptr<int>()) + mass_offset,
                 thrust::device_ptr<int>(d_triplet_rows->get_device_ptr<int>()) + mass_offset + 10,
                 mass_rows.begin());
    thrust::copy(thrust::device_ptr<int>(d_triplet_cols->get_device_ptr<int>()) + mass_offset,
                 thrust::device_ptr<int>(d_triplet_cols->get_device_ptr<int>()) + mass_offset + 10,
                 mass_cols.begin());
    thrust::copy(thrust::device_ptr<float>(d_triplet_vals->get_device_ptr<float>()) + mass_offset,
                 thrust::device_ptr<float>(d_triplet_vals->get_device_ptr<float>()) + mass_offset + 10,
                 mass_vals.begin());
    printf("[GPU] Mass matrix triplets (at offset %d):\n", mass_offset);
    for (int i = 0; i < 10; i++) {
        printf("  [%d] (%d, %d) = %.6e\n", mass_offset + i, mass_rows[i], mass_cols[i], mass_vals[i]);
    }

    // Debug: Search for all (0,0) entries in spring triplets
    std::vector<int> spring_rows(std::min(100, (int)max_spring_triplets));
    std::vector<int> spring_cols(std::min(100, (int)max_spring_triplets));
    std::vector<float> spring_vals(std::min(100, (int)max_spring_triplets));
    thrust::copy(thrust::device_ptr<int>(d_triplet_rows->get_device_ptr<int>()),
                 thrust::device_ptr<int>(d_triplet_rows->get_device_ptr<int>()) + std::min(100, (int)max_spring_triplets),
                 spring_rows.begin());
    thrust::copy(thrust::device_ptr<int>(d_triplet_cols->get_device_ptr<int>()),
                 thrust::device_ptr<int>(d_triplet_cols->get_device_ptr<int>()) + std::min(100, (int)max_spring_triplets),
                 spring_cols.begin());
    thrust::copy(thrust::device_ptr<float>(d_triplet_vals->get_device_ptr<float>()),
                 thrust::device_ptr<float>(d_triplet_vals->get_device_ptr<float>()) + std::min(100, (int)max_spring_triplets),
                 spring_vals.begin());
    printf("[GPU] Searching first 100 spring triplets for (0,0) contributions:\n");
    int count_00 = 0;
    float sum_00 = 0.0f;
    for (int i = 0; i < std::min(100, (int)max_spring_triplets); i++) {
        if (spring_rows[i] == 0 && spring_cols[i] == 0) {
            printf("  [%d] (0, 0) = %.6e\n", i, spring_vals[i]);
            count_00++;
            sum_00 += spring_vals[i];
        }
    }
    printf("[GPU] Found %d (0,0) entries in first 100, sum = %.6e\n", count_00, sum_00);

    // Check for CUDA errors before sorting
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[GPU ERROR] Before sort: %s\n", cudaGetErrorString(err));
        result.nnz = 0;
        return result;
    }

    // Sort triplets by (row, col) using Thrust
    thrust::device_ptr<int> rows_ptr(d_triplet_rows->get_device_ptr<int>());
    thrust::device_ptr<int> cols_ptr(d_triplet_cols->get_device_ptr<int>());
    thrust::device_ptr<float> vals_ptr(d_triplet_vals->get_device_ptr<float>());

    // Create zip iterator for sorting
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(rows_ptr, cols_ptr, vals_ptr));
    auto zip_end = zip_begin + total_triplets;

    thrust::sort(thrust::device, zip_begin, zip_end, TripletComparator());

    cudaDeviceSynchronize();

    // Sum duplicate entries using thrust::reduce_by_key
    thrust::device_vector<int> unique_rows(total_triplets);
    thrust::device_vector<int> unique_cols(total_triplets);
    thrust::device_vector<float> unique_vals(total_triplets);

    auto key_begin = thrust::make_zip_iterator(
        thrust::make_tuple(rows_ptr, cols_ptr));
    auto key_end = key_begin + total_triplets;

    auto new_end = thrust::reduce_by_key(
        thrust::device,
        key_begin,
        key_end,
        vals_ptr,
        thrust::make_zip_iterator(
            thrust::make_tuple(unique_rows.begin(), unique_cols.begin())),
        unique_vals.begin(),
        KeyEqual());

    int nnz = new_end.second - unique_vals.begin();

    printf("[GPU] Non-zeros after reduction: %d\n", nnz);
    printf("[GPU] Matrix size: %d x %d\n", n, n);

    // Debug: Print first few unique triplets
    std::vector<int> debug_rows(std::min(20, nnz));
    std::vector<int> debug_cols(std::min(20, nnz));
    std::vector<float> debug_vals(std::min(20, nnz));
    thrust::copy(unique_rows.begin(), unique_rows.begin() + std::min(20, nnz),
                 debug_rows.begin());
    thrust::copy(unique_cols.begin(), unique_cols.begin() + std::min(20, nnz),
                 debug_cols.begin());
    thrust::copy(unique_vals.begin(), unique_vals.begin() + std::min(20, nnz),
                 debug_vals.begin());
    printf("[GPU] First 20 triplets after reduction (sorted by row,col):\n");
    int row0_count = 0;
    for (int i = 0; i < std::min(20, nnz); i++) {
        printf("  (%d, %d) = %.6e\n", debug_rows[i], debug_cols[i], debug_vals[i]);
        if (debug_rows[i] == 0) row0_count++;
    }
    printf("[GPU] Row 0 has %d entries in first 20 triplets\n", row0_count);

    // Convert to CSR format
    result.num_rows = n;
    result.num_cols = n;
    result.nnz = nnz;

    result.col_indices = cuda::create_cuda_linear_buffer<int>(nnz);
    result.values = cuda::create_cuda_linear_buffer<float>(nnz);
    result.row_offsets = cuda::create_cuda_linear_buffer<int>(n + 1);

    // Copy unique cols and vals
    thrust::copy(unique_cols.begin(), unique_cols.begin() + nnz,
                 thrust::device_ptr<int>(result.col_indices->get_device_ptr<int>()));
    thrust::copy(unique_vals.begin(), unique_vals.begin() + nnz,
                 thrust::device_ptr<float>(result.values->get_device_ptr<float>()));

    // Build row_offsets: histogram approach
    thrust::device_ptr<int> row_offsets_ptr(result.row_offsets->get_device_ptr<int>());
    int* row_offsets_raw = thrust::raw_pointer_cast(row_offsets_ptr);
    
    // Initialize all row_offsets to 0
    thrust::fill(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, 0);
    
    // Count entries per row using histogram
    thrust::device_ptr<int> unique_rows_ptr(unique_rows.data());
    int* unique_rows_raw = thrust::raw_pointer_cast(unique_rows_ptr);
    
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nnz),
        [row_offsets_raw, unique_rows_raw] __device__ (int idx) {
            int row = unique_rows_raw[idx];
            atomicAdd(&row_offsets_raw[row], 1);
        });
    
    cudaDeviceSynchronize();

    // Compute prefix sum to get row offsets
    thrust::exclusive_scan(
        thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, row_offsets_ptr);

    cudaDeviceSynchronize();

    printf("[GPU] CSR assembly complete: %dx%d matrix with %d non-zeros\n", n, n, nnz);

    return result;
}

// Kernel to compute spring energy on GPU
__global__ void compute_spring_energy_kernel(
    const float* x_curr,
    const int* springs,
    const float* rest_lengths,
    float stiffness,
    int num_springs,
    float* spring_energies)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_springs)
        return;
    
    int vi = springs[sid * 2];
    int vj = springs[sid * 2 + 1];
    float l0 = rest_lengths[sid];
    
    // Get positions
    float xi[3] = {x_curr[vi * 3], x_curr[vi * 3 + 1], x_curr[vi * 3 + 2]};
    float xj[3] = {x_curr[vj * 3], x_curr[vj * 3 + 1], x_curr[vj * 3 + 2]};
    
    // Compute current length
    float diff[3] = {xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]};
    float l = sqrtf(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
    
    // Spring energy: 0.5 * k * (l - l0)^2
    float stretch = l - l0;
    spring_energies[sid] = 0.5f * stiffness * stretch * stretch;
}

// Compute total energy: E = 0.5 * M * ||x - x_tilde||^2 + spring_energy - f_ext^T * x
float compute_energy_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles)
{
    int n = num_particles * 3;
    int num_springs = springs->getDesc().element_count / 2;
    
    float* x_ptr = reinterpret_cast<float*>(x_curr->get_device_ptr());
    float* x_tilde_ptr = reinterpret_cast<float*>(x_tilde->get_device_ptr());
    float* M_ptr = reinterpret_cast<float*>(M_diag->get_device_ptr());
    float* f_ptr = reinterpret_cast<float*>(f_ext->get_device_ptr());
    
    // Compute inertial energy: 0.5 * M * ||x - x_tilde||^2
    auto d_inertial_terms = cuda::create_cuda_linear_buffer<float>(n);
    float* inertial_ptr = reinterpret_cast<float*>(d_inertial_terms->get_device_ptr());
    
    cuda::GPUParallelFor("compute_inertial_energy", n, GPU_LAMBDA_Ex(int i) {
        int particle_id = i / 3;
        float diff = x_ptr[i] - x_tilde_ptr[i];
        inertial_ptr[i] = 0.5f * M_ptr[particle_id] * diff * diff;
    });
    
    // Sum inertial energy
    thrust::device_ptr<float> d_inertial_thrust(inertial_ptr);
    float E_inertial = thrust::reduce(d_inertial_thrust, d_inertial_thrust + n, 0.0f);
    
    // Compute spring energy
    auto d_spring_energies = cuda::create_cuda_linear_buffer<float>(num_springs);
    float* spring_energy_ptr = reinterpret_cast<float*>(d_spring_energies->get_device_ptr());
    
    int block_size = 256;
    int num_blocks = (num_springs + block_size - 1) / block_size;
    compute_spring_energy_kernel<<<num_blocks, block_size>>>(
        x_ptr,
        reinterpret_cast<const int*>(springs->get_device_ptr()),
        reinterpret_cast<const float*>(rest_lengths->get_device_ptr()),
        stiffness,
        num_springs,
        spring_energy_ptr);
    
    cudaDeviceSynchronize();
    
    // Sum spring energy
    thrust::device_ptr<float> d_spring_thrust(spring_energy_ptr);
    float E_spring = thrust::reduce(d_spring_thrust, d_spring_thrust + num_springs, 0.0f);
    
    // Compute potential energy: -f_ext^T * x
    auto d_potential_terms = cuda::create_cuda_linear_buffer<float>(n);
    float* potential_ptr = reinterpret_cast<float*>(d_potential_terms->get_device_ptr());
    
    cuda::GPUParallelFor("compute_potential_energy", n, GPU_LAMBDA_Ex(int i) {
        potential_ptr[i] = -f_ptr[i] * x_ptr[i] * dt * dt;
    });
    
    thrust::device_ptr<float> d_potential_thrust(potential_ptr);
    float E_potential = thrust::reduce(d_potential_thrust, d_potential_thrust + n, 0.0f);
    
    float total_energy = E_inertial + E_spring + E_potential;
    
    return total_energy;
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
