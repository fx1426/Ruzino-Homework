#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <Eigen/Eigen>
#include <RHI/cuda.hpp>
#include <RHI/rhi.hpp>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include "RHI/internal/cuda_extension.hpp"
#include "rzsim_cuda/adjacency_map.cuh"


RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Kernel to count neighbors for each vertex
__global__ void count_neighbors_kernel(
    const int* face_vertex_counts,
    const int* face_vertex_indices,
    unsigned* neighbor_counts,
    unsigned num_vertices,
    unsigned num_faces)
{
    unsigned face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces)
        return;

    int vertex_count = face_vertex_counts[face_idx];

    // Calculate starting index for this face's vertices
    unsigned face_start_idx = 0;
    for (unsigned i = 0; i < face_idx; i++) {
        face_start_idx += face_vertex_counts[i];
    }

    // For each edge in the face
    for (int i = 0; i < vertex_count; i++) {
        int v0_idx = face_vertex_indices[face_start_idx + i];
        int v1_idx =
            face_vertex_indices[face_start_idx + (i + 1) % vertex_count];

        // Both directions: v0->v1 and v1->v0
        atomicAdd(&neighbor_counts[v0_idx], 1);
        atomicAdd(&neighbor_counts[v1_idx], 1);
    }
}

// Kernel to fill the adjacency list
__global__ void fill_adjacency_kernel(
    const int* face_vertex_counts,
    const int* face_vertex_indices,
    unsigned*
        neighbor_write_pos,    // Current write positions (will be incremented)
    const unsigned* offsets,   // Offsets for each vertex
    unsigned* adjacency_list,  // Output adjacency data
    unsigned num_vertices,
    unsigned num_faces)
{
    unsigned face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces)
        return;

    int vertex_count = face_vertex_counts[face_idx];
    unsigned face_start_idx = 0;

    // Calculate starting index for this face's vertices
    for (unsigned i = 0; i < face_idx; i++) {
        face_start_idx += face_vertex_counts[i];
    }

    // For each edge in the face
    for (int i = 0; i < vertex_count; i++) {
        int v0_idx = face_vertex_indices[face_start_idx + i];
        int v1_idx =
            face_vertex_indices[face_start_idx + (i + 1) % vertex_count];

        // Add v1 as neighbor of v0
        unsigned offset = offsets[v0_idx];
        unsigned pos = atomicAdd(&neighbor_write_pos[v0_idx], 1);
        adjacency_list[offset + 1 + pos] = v1_idx;

        // Add v0 as neighbor of v1
        offset = offsets[v1_idx];
        pos = atomicAdd(&neighbor_write_pos[v1_idx], 1);
        adjacency_list[offset + 1 + pos] = v0_idx;
    }
}

cuda::CUDALinearBufferHandle compute_adjacency_map_gpu(
    cuda::CUDALinearBufferHandle vertices,
    cuda::CUDALinearBufferHandle faceVertexCounts,
    cuda::CUDALinearBufferHandle faceVertexIndices)
{
    auto vertex_buffer_addr = vertices->get_device_ptr();
    auto face_vertex_counts_addr = faceVertexCounts->get_device_ptr();
    auto face_vertex_indices_addr = faceVertexIndices->get_device_ptr();

    auto vertex_count = vertices->getDesc().element_count;
    auto face_count = faceVertexCounts->getDesc().element_count;

    // Step 1: Count neighbors for each vertex using atomic operations
    thrust::device_vector<unsigned> neighbor_counts(vertex_count, 0);

    int threads_per_block = 256;
    int blocks = (face_count + threads_per_block - 1) / threads_per_block;

    count_neighbors_kernel<<<blocks, threads_per_block>>>(
        (const int*)face_vertex_counts_addr,
        (const int*)face_vertex_indices_addr,
        thrust::raw_pointer_cast(neighbor_counts.data()),
        vertex_count,
        face_count);
    cudaDeviceSynchronize();

    // Step 2: Compute sizes array (each vertex needs size+1 space for count +
    // neighbors)
    thrust::device_vector<unsigned> sizes(vertex_count);

    thrust::transform(
        neighbor_counts.begin(),
        neighbor_counts.end(),
        sizes.begin(),
        [] __device__(unsigned count) { return count + 1; });

    // Step 3: Use CUB for high-performance exclusive scan to get offsets
    thrust::device_vector<unsigned> offsets(vertex_count);

    // Allocate temporary storage for CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Query temporary storage size
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(sizes.data()),
        thrust::raw_pointer_cast(offsets.data()),
        vertex_count);

    // Allocate temporary storage
    thrust::device_vector<char> temp_storage(temp_storage_bytes);

    // Run exclusive scan
    cub::DeviceScan::ExclusiveSum(
        thrust::raw_pointer_cast(temp_storage.data()),
        temp_storage_bytes,
        thrust::raw_pointer_cast(sizes.data()),
        thrust::raw_pointer_cast(offsets.data()),
        vertex_count);
    cudaDeviceSynchronize();

    // Step 4: Calculate total size and allocate output buffer
    unsigned total_size = thrust::reduce(sizes.begin(), sizes.end(), 0u);

    cuda::CUDALinearBufferDesc desc;
    desc.element_count = total_size;
    desc.element_size = sizeof(unsigned);

    auto result_buffer = cuda::create_cuda_linear_buffer(desc);
    auto result_ptr = (unsigned*)result_buffer->get_device_ptr();

    // Step 5: Fill count field for each vertex at its offset position
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(
            thrust::make_tuple(offsets.begin(), neighbor_counts.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(offsets.end(), neighbor_counts.end())),
        [result_ptr] __device__(const auto& tuple) {
            unsigned offset = thrust::get<0>(tuple);
            unsigned count = thrust::get<1>(tuple);
            result_ptr[offset] = count;
        });

    // Step 6: Fill adjacency list - reset write positions and call fill kernel
    thrust::fill(neighbor_counts.begin(), neighbor_counts.end(), 0);

    fill_adjacency_kernel<<<blocks, threads_per_block>>>(
        (const int*)face_vertex_counts_addr,
        (const int*)face_vertex_indices_addr,
        thrust::raw_pointer_cast(neighbor_counts.data()),
        thrust::raw_pointer_cast(offsets.data()),
        result_ptr,
        vertex_count,
        face_count);
    cudaDeviceSynchronize();

    return result_buffer;
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
