#include <cusparse.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
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

// Build per-vertex spring adjacency: for each vertex, store list of spring
// indices it belongs to
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
build_vertex_spring_adjacency_gpu(
    cuda::CUDALinearBufferHandle springs,
    int num_particles)
{
    int num_springs = springs->getDesc().element_count / 2;
    const int* springs_ptr = springs->get_device_ptr<int>();

    // Count springs per vertex
    auto d_spring_count = cuda::create_cuda_linear_buffer<int>(num_particles);
    int* count_ptr = d_spring_count->get_device_ptr<int>();

    // Initialize counts to zero
    cudaMemset(count_ptr, 0, num_particles * sizeof(int));

    // Count: each spring contributes to 2 vertices
    cuda::GPUParallelFor(
        "count_springs_per_vertex", num_springs, GPU_LAMBDA_Ex(int s) {
            int i = springs_ptr[s * 2];
            int j = springs_ptr[s * 2 + 1];
            atomicAdd(&count_ptr[i], 1);
            atomicAdd(&count_ptr[j], 1);
        });

    // Build offset buffer (prefix sum)
    auto d_offsets = cuda::create_cuda_linear_buffer<int>(num_particles + 1);
    int* offsets_ptr = d_offsets->get_device_ptr<int>();

    thrust::device_ptr<int> count_thrust(count_ptr);
    thrust::device_ptr<int> offsets_thrust(offsets_ptr);
    thrust::exclusive_scan(
        thrust::device,
        count_thrust,
        count_thrust + num_particles,
        offsets_thrust);

    // Get total count for last offset
    int total_entries;
    cudaMemcpy(
        &total_entries,
        count_ptr + num_particles - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    int last_offset;
    cudaMemcpy(
        &last_offset,
        offsets_ptr + num_particles - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    total_entries += last_offset;
    cudaMemcpy(
        offsets_ptr + num_particles,
        &total_entries,
        sizeof(int),
        cudaMemcpyHostToDevice);

    // Allocate spring indices array
    auto d_spring_indices = cuda::create_cuda_linear_buffer<int>(total_entries);
    int* indices_ptr = d_spring_indices->get_device_ptr<int>();

    // Reset counts for filling
    cudaMemset(count_ptr, 0, num_particles * sizeof(int));

    // Fill spring indices: each spring adds itself to both vertices
    cuda::GPUParallelFor(
        "fill_spring_indices", num_springs, GPU_LAMBDA_Ex(int s) {
            int i = springs_ptr[s * 2];
            int j = springs_ptr[s * 2 + 1];

            // Add spring index to vertex i's list
            int pos_i = offsets_ptr[i] + atomicAdd(&count_ptr[i], 1);
            indices_ptr[pos_i] = s;

            // Add spring index to vertex j's list
            int pos_j = offsets_ptr[j] + atomicAdd(&count_ptr[j], 1);
            indices_ptr[pos_j] = s;
        });

    cudaDeviceSynchronize();

    return { d_spring_indices, d_offsets };
}

// Combined function: extract edges AND build adjacency in one optimized pass
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
build_springs_with_adjacency_gpu(
    cuda::CUDALinearBufferHandle triangles,
    int num_particles)
{
    // Step 1: Extract and deduplicate edges (reuse existing function)
    auto springs = build_edge_set_gpu(cuda::CUDALinearBufferHandle(), triangles);
    
    // Step 2: Build vertex-spring adjacency (reuse existing function)
    auto [spring_indices, vertex_offsets] = 
        build_vertex_spring_adjacency_gpu(springs, num_particles);
    
    return { springs, spring_indices, vertex_offsets };
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

// Optimized gradient kernel using per-vertex spring adjacency
__global__ void compute_gradient_kernel_optimized(
    const float* x_curr,
    const float* x_tilde,
    const float* M_diag,
    const float* f_ext,
    const int* springs,
    const float* rest_lengths,
    const int* spring_indices_per_vertex,
    const int* vertex_spring_offsets,
    float stiffness,
    float dt,
    int num_particles,
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

    // Add spring forces - only iterate over springs connected to this vertex
    int start = vertex_spring_offsets[tid];
    int end = vertex_spring_offsets[tid + 1];

    for (int idx = start; idx < end; ++idx) {
        int s = spring_indices_per_vertex[idx];

        int i = springs[s * 2];
        int j = springs[s * 2 + 1];
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
    cuda::CUDALinearBufferHandle spring_indices_per_vertex,
    cuda::CUDALinearBufferHandle vertex_spring_offsets,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle grad)
{
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    compute_gradient_kernel_optimized<<<num_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        x_tilde->get_device_ptr<float>(),
        M_diag->get_device_ptr<float>(),
        f_ext->get_device_ptr<float>(),
        springs->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        spring_indices_per_vertex->get_device_ptr<int>(),
        vertex_spring_offsets->get_device_ptr<int>(),
        stiffness,
        dt,
        num_particles,
        grad->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

// Custom 3x3 symmetric eigenvalue decomposition using Jacobi rotation
// This is needed because Eigen's SelfAdjointEigenSolver doesn't work on CUDA
// device
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
__device__ Eigen::Matrix3f project_psd_custom(const Eigen::Matrix3f& H)
{
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;

    // Compute eigendecomposition
    eigen_decomposition_3x3(H, eigenvalues, eigenvectors);

    // Clamp negative eigenvalues to zero
    for (int i = 0; i < 3; i++) {
        if (eigenvalues(i) < 0.0f) {
            eigenvalues(i) = 0.0f;
        }
    }

    // Reconstruct: H_psd = V * Lambda * V^T
    Eigen::Matrix3f result =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    return result;
}

// ============================================================================
// NEW: Zero-sort direct-fill implementation
// ============================================================================

// Kernel to generate all matrix entry (row, col) pairs in parallel
__global__ void generate_matrix_entries_kernel(
    const int* springs,
    int num_springs,
    int num_particles,
    int* rows,
    int* cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int n = num_particles * 3;
    int num_mass_entries = n;
    int num_spring_entries = num_springs * 36;
    int total_entries = num_mass_entries + num_spring_entries;

    if (tid >= total_entries)
        return;

    if (tid < num_mass_entries) {
        // Mass diagonal entry
        rows[tid] = tid;
        cols[tid] = tid;
    }
    else {
        // Spring entry
        int spring_idx = tid - num_mass_entries;
        int sid = spring_idx / 36;
        int entry_in_spring = spring_idx % 36;

        int vi = springs[sid * 2];
        int vj = springs[sid * 2 + 1];

        // Decode entry_in_spring into block and position
        int block_idx = entry_in_spring / 9;  // 0-3 for the 4 blocks
        int pos_in_block = entry_in_spring % 9;

        int block_r = block_idx / 2;
        int block_c = block_idx % 2;
        int r = pos_in_block / 3;
        int c = pos_in_block % 3;

        int v_row = (block_r == 0) ? vi : vj;
        int v_col = (block_c == 0) ? vi : vj;

        rows[tid] = v_row * 3 + r;
        cols[tid] = v_col * 3 + c;
    }
}

// Binary search helper for finding (row, col) in sorted unique arrays
__device__ int find_entry_position(
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
        int mid_row = unique_rows[mid];
        int mid_col = unique_cols[mid];

        if (mid_row == target_row && mid_col == target_col) {
            return mid;
        }

        // Compare as (row, col) pairs
        if (mid_row < target_row ||
            (mid_row == target_row && mid_col < target_col)) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }

    return -1;  // Not found (should never happen for valid structure)
}

// Kernel to build mass diagonal positions using binary search
__global__ void build_mass_positions_kernel(
    const int* unique_rows,
    const int* unique_cols,
    int nnz,
    int num_dofs,
    int* mass_positions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dofs)
        return;

    // Mass diagonal: find position of (tid, tid)
    mass_positions[tid] =
        find_entry_position(unique_rows, unique_cols, nnz, tid, tid);
}

// Kernel to build spring positions using binary search
__global__ void build_spring_positions_kernel(
    const int* unique_rows,
    const int* unique_cols,
    const int* springs,
    int nnz,
    int num_springs,
    int* spring_positions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_springs)
        return;

    int vi = springs[tid * 2];
    int vj = springs[tid * 2 + 1];
    int base_idx = tid * 36;
    int count = 0;

    // Same order as kernel: (vi,vi), (vi,vj), (vj,vi), (vj,vj)
    for (int block_r = 0; block_r < 2; ++block_r) {
        for (int block_c = 0; block_c < 2; ++block_c) {
            int v_row = (block_r == 0) ? vi : vj;
            int v_col = (block_c == 0) ? vi : vj;

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    int global_row = v_row * 3 + r;
                    int global_col = v_col * 3 + c;

                    spring_positions[base_idx + count] = find_entry_position(
                        unique_rows, unique_cols, nnz, global_row, global_col);
                    count++;
                }
            }
        }
    }
}

// Build CSR sparsity pattern once during initialization (GPU version)
CSRStructure build_hessian_structure_gpu(
    cuda::CUDALinearBufferHandle springs,
    int num_particles)
{
    CSRStructure structure;
    int num_springs = springs->getDesc().element_count / 2;
    int n = num_particles * 3;

    int num_mass_entries = n;
    int num_spring_entries = num_springs * 36;
    int total_entries = num_mass_entries + num_spring_entries;

    // Allocate temporary buffers for all entries
    auto d_all_rows = cuda::create_cuda_linear_buffer<int>(total_entries);
    auto d_all_cols = cuda::create_cuda_linear_buffer<int>(total_entries);

    // Generate all (row, col) pairs on GPU
    int block_size = 256;
    int num_blocks = (total_entries + block_size - 1) / block_size;

    generate_matrix_entries_kernel<<<num_blocks, block_size>>>(
        springs->get_device_ptr<int>(),
        num_springs,
        num_particles,
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>());

    cudaDeviceSynchronize();

    // Sort by (row, col) using thrust
    thrust::device_ptr<int> rows_ptr(d_all_rows->get_device_ptr<int>());
    thrust::device_ptr<int> cols_ptr(d_all_cols->get_device_ptr<int>());

    auto zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(rows_ptr, cols_ptr));

    thrust::sort(
        thrust::device,
        zip_begin,
        zip_begin + total_entries,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            return thrust::get<1>(a) < thrust::get<1>(b);
        });

    cudaDeviceSynchronize();

    // Remove duplicates
    auto new_end = thrust::unique(
        thrust::device,
        zip_begin,
        zip_begin + total_entries,
        [] __device__(
            const thrust::tuple<int, int>& a,
            const thrust::tuple<int, int>& b) {
            return thrust::get<0>(a) == thrust::get<0>(b) &&
                   thrust::get<1>(a) == thrust::get<1>(b);
        });

    int nnz = new_end - zip_begin;

    cudaDeviceSynchronize();

    // Allocate CSR arrays
    structure.col_indices = cuda::create_cuda_linear_buffer<int>(nnz);
    structure.row_offsets = cuda::create_cuda_linear_buffer<int>(n + 1);
    structure.mass_value_positions = cuda::create_cuda_linear_buffer<int>(n);
    structure.spring_value_positions =
        cuda::create_cuda_linear_buffer<int>(num_springs * 36);

    // Copy unique columns
    cudaMemcpy(
        structure.col_indices->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        nnz * sizeof(int),
        cudaMemcpyDeviceToDevice);

    // Build row_offsets using histogram
    thrust::device_ptr<int> row_offsets_ptr(
        structure.row_offsets->get_device_ptr<int>());
    thrust::fill(thrust::device, row_offsets_ptr, row_offsets_ptr + n + 1, 0);

    int* row_offsets_raw = structure.row_offsets->get_device_ptr<int>();
    int* unique_rows_raw = d_all_rows->get_device_ptr<int>();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nnz),
        [row_offsets_raw, unique_rows_raw] __device__(int idx) {
            int row = unique_rows_raw[idx];
            atomicAdd(&row_offsets_raw[row], 1);
        });

    cudaDeviceSynchronize();

    // Compute prefix sum for row offsets
    thrust::exclusive_scan(
        thrust::device,
        row_offsets_ptr,
        row_offsets_ptr + n + 1,
        row_offsets_ptr);

    cudaDeviceSynchronize();

    // Build position mappings on GPU using binary search
    int mass_blocks = (n + block_size - 1) / block_size;
    build_mass_positions_kernel<<<mass_blocks, block_size>>>(
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        nnz,
        n,
        structure.mass_value_positions->get_device_ptr<int>());

    int spring_blocks = (num_springs + block_size - 1) / block_size;
    build_spring_positions_kernel<<<spring_blocks, block_size>>>(
        d_all_rows->get_device_ptr<int>(),
        d_all_cols->get_device_ptr<int>(),
        springs->get_device_ptr<int>(),
        nnz,
        num_springs,
        structure.spring_value_positions->get_device_ptr<int>());

    cudaDeviceSynchronize();

    structure.num_rows = n;
    structure.num_cols = n;
    structure.nnz = nnz;

    return structure;
}

// Kernel: Directly fill spring Hessian values into CSR (no sorting needed!)
__global__ void fill_spring_hessian_values_kernel(
    const float* x_curr,
    const int* springs,
    const float* rest_lengths,
    const int* value_positions,  // Pre-computed positions in values array
    float stiffness,
    float dt,
    int num_springs,
    float* values)  // Output CSR values array
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_springs)
        return;

    int vi = springs[sid * 2];
    int vj = springs[sid * 2 + 1];
    float l0 = rest_lengths[sid];

    if (l0 < 1e-10f) {
        return;  // Skip degenerate springs
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

    // PSD projection
    H_diff = project_psd_custom(H_diff);

    // Scale by dt^2
    float scale = dt * dt;
    H_diff *= scale;

    // Directly write to pre-computed positions (using atomic add for safety)
    int base_idx = sid * 36;
    int count = 0;

    // 4 blocks: (vi,vi), (vi,vj), (vj,vi), (vj,vj)
    for (int block_r = 0; block_r < 2; ++block_r) {
        for (int block_c = 0; block_c < 2; ++block_c) {
            float sign_row = (block_r == 0) ? 1.0f : -1.0f;
            float sign_col = (block_c == 0) ? 1.0f : -1.0f;

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float val = H_diff(r, c) * sign_row * sign_col;
                    int pos = value_positions[base_idx + count++];
                    atomicAdd(&values[pos], val);
                }
            }
        }
    }
}

// Kernel: Directly fill mass diagonal into CSR
__global__ void fill_mass_diagonal_kernel(
    const float* M_diag,
    const int* mass_positions,
    int num_dofs,
    float* values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_dofs)
        return;

    float regularization = 1e-6f;
    int pos = mass_positions[tid];
    values[pos] = M_diag[tid] + regularization;
}

// Fast update: directly fill values (NO SORTING!)
void update_hessian_values_gpu(
    const CSRStructure& csr_structure,
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle values)
{
    int num_springs = springs->getDesc().element_count / 2;
    int num_dofs = num_particles * 3;
    int block_size = 256;

    // Zero out values array
    cudaMemset(
        values->get_device_ptr<float>(), 0, csr_structure.nnz * sizeof(float));

    // Fill mass diagonal
    int mass_blocks = (num_dofs + block_size - 1) / block_size;
    fill_mass_diagonal_kernel<<<mass_blocks, block_size>>>(
        M_diag->get_device_ptr<float>(),
        csr_structure.mass_value_positions->get_device_ptr<int>(),
        num_dofs,
        values->get_device_ptr<float>());

    // Fill spring contributions
    int spring_blocks = (num_springs + block_size - 1) / block_size;
    fill_spring_hessian_values_kernel<<<spring_blocks, block_size>>>(
        x_curr->get_device_ptr<float>(),
        springs->get_device_ptr<int>(),
        rest_lengths->get_device_ptr<float>(),
        csr_structure.spring_value_positions->get_device_ptr<int>(),
        stiffness,
        dt,
        num_springs,
        values->get_device_ptr<float>());

    cudaDeviceSynchronize();
}

// ============================================================================
// End of zero-sort implementation
// ============================================================================

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
    float l0_sq = l0 * l0;

    // Get positions
    float xi[3] = { x_curr[vi * 3], x_curr[vi * 3 + 1], x_curr[vi * 3 + 2] };
    float xj[3] = { x_curr[vj * 3], x_curr[vj * 3 + 1], x_curr[vj * 3 + 2] };

    // Compute squared distance
    float diff[3] = { xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2] };
    float diff_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

    // Spring energy matching gradient: 0.5 * k * l0^2 * ((diff_sq / l0^2) -
    // 1)^2
    float ratio = diff_sq / l0_sq - 1.0f;
    spring_energies[sid] = 0.5f * stiffness * l0_sq * ratio * ratio;
}

// Compute total energy: E = 0.5 * M * ||x - x_tilde||^2 + spring_energy -
// f_ext^T * x
float compute_energy_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle d_inertial_terms,
    cuda::CUDALinearBufferHandle d_spring_energies,
    cuda::CUDALinearBufferHandle d_potential_terms)
{
    int n = num_particles * 3;
    int num_springs = springs->getDesc().element_count / 2;

    float* x_ptr = reinterpret_cast<float*>(x_curr->get_device_ptr());
    float* x_tilde_ptr = reinterpret_cast<float*>(x_tilde->get_device_ptr());
    float* M_ptr = reinterpret_cast<float*>(M_diag->get_device_ptr());
    float* f_ptr = reinterpret_cast<float*>(f_ext->get_device_ptr());

    // Use pre-allocated buffers instead of creating new ones
    float* inertial_ptr =
        reinterpret_cast<float*>(d_inertial_terms->get_device_ptr());

    cuda::GPUParallelFor(
        "compute_inertial_energy", n, GPU_LAMBDA_Ex(int i) {
            float diff = x_ptr[i] - x_tilde_ptr[i];
            inertial_ptr[i] = 0.5f * M_ptr[i] * diff * diff;
        });

    // Sum inertial energy
    thrust::device_ptr<float> d_inertial_thrust(inertial_ptr);
    float E_inertial =
        thrust::reduce(d_inertial_thrust, d_inertial_thrust + n, 0.0f);

    // Use pre-allocated buffer for spring energies
    float* spring_energy_ptr =
        reinterpret_cast<float*>(d_spring_energies->get_device_ptr());

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
    float E_spring =
        thrust::reduce(d_spring_thrust, d_spring_thrust + num_springs, 0.0f);

    // Use pre-allocated buffer for potential energy
    float* potential_ptr =
        reinterpret_cast<float*>(d_potential_terms->get_device_ptr());

    cuda::GPUParallelFor(
        "compute_potential_energy", n, GPU_LAMBDA_Ex(int i) {
            potential_ptr[i] = -f_ptr[i] * x_ptr[i] * dt * dt;
        });

    thrust::device_ptr<float> d_potential_thrust(potential_ptr);
    float E_potential =
        thrust::reduce(d_potential_thrust, d_potential_thrust + n, 0.0f);

    float total_energy = E_inertial + dt * dt * E_spring + E_potential;

    return total_energy;
}

// Functors for thrust operations (must be defined outside functions for CUDA
// compatibility)
struct square_op {
    __device__ float operator()(float x) const
    {
        return x * x;
    }
};

// Kernel for axpy: result = y + alpha * x
__global__ void axpy_kernel(
    float alpha,
    const float* x,
    const float* y,
    float* result,
    int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        result[tid] = y[tid] + alpha * x[tid];
    }
}

// Kernel for negation: out = -in
__global__ void negate_kernel(const float* in, float* out, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = -in[tid];
    }
}

// Kernel to project positions to ground (enforce z >= ground_height)
__global__ void project_to_ground_kernel(
    float* positions,
    int num_particles,
    float ground_height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_particles) {
        if (positions[tid * 3 + 2] < ground_height) {
            positions[tid * 3 + 2] = ground_height;
        }
    }
}

// GPU vector operations to avoid CPU-GPU transfers
float compute_vector_norm_gpu(cuda::CUDALinearBufferHandle vec, int size)
{
    float* vec_ptr = reinterpret_cast<float*>(vec->get_device_ptr());
    thrust::device_ptr<float> d_vec(vec_ptr);

    // Compute sum of squares using functor
    float sum_sq = thrust::transform_reduce(
        thrust::device,
        d_vec,
        d_vec + size,
        square_op(),
        0.0f,
        thrust::plus<float>());

    return sqrtf(sum_sq);
}

float compute_dot_product_gpu(
    cuda::CUDALinearBufferHandle vec1,
    cuda::CUDALinearBufferHandle vec2,
    int size)
{
    float* vec1_ptr = reinterpret_cast<float*>(vec1->get_device_ptr());
    float* vec2_ptr = reinterpret_cast<float*>(vec2->get_device_ptr());

    thrust::device_ptr<float> d_vec1(vec1_ptr);
    thrust::device_ptr<float> d_vec2(vec2_ptr);

    return thrust::inner_product(
        thrust::device, d_vec1, d_vec1 + size, d_vec2, 0.0f);
}

void axpy_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size)
{
    float* x_ptr = reinterpret_cast<float*>(x->get_device_ptr());
    float* y_ptr = reinterpret_cast<float*>(y->get_device_ptr());
    float* result_ptr = reinterpret_cast<float*>(result->get_device_ptr());

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    axpy_kernel<<<num_blocks, block_size>>>(
        alpha, x_ptr, y_ptr, result_ptr, size);
    cudaDeviceSynchronize();
}

void negate_gpu(
    cuda::CUDALinearBufferHandle in,
    cuda::CUDALinearBufferHandle out,
    int size)
{
    float* in_ptr = reinterpret_cast<float*>(in->get_device_ptr());
    float* out_ptr = reinterpret_cast<float*>(out->get_device_ptr());

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    negate_kernel<<<num_blocks, block_size>>>(in_ptr, out_ptr, size);
    cudaDeviceSynchronize();
}

void project_to_ground_gpu(
    cuda::CUDALinearBufferHandle positions,
    int num_particles,
    float ground_height)
{
    float* pos_ptr = reinterpret_cast<float*>(positions->get_device_ptr());

    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;
    project_to_ground_kernel<<<num_blocks, block_size>>>(
        pos_ptr, num_particles, ground_height);
    cudaDeviceSynchronize();
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
