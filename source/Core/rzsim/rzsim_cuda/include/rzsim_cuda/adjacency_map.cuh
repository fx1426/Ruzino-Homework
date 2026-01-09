#pragma once

#include <vector>

#include "RHI/internal/cuda_extension.hpp"
#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Surface mesh (triangles): For each vertex, stores pairs of opposite edge
// vertices For vertex v in triangle (v, a, b), stores pair (a, b) Format:
// [count_v0, a1,b1, a2,b2, ... | count_v1, ... ]
// - adjacency_list: flattened pairs of opposite vertices
// - offset_buffer: starting position for each vertex
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_surface_adjacency_gpu(
    cuda::CUDALinearBufferHandle vertices,
    cuda::CUDALinearBufferHandle
        triangles);  // triangle indices [v0,v1,v2, ...]

// Volume mesh (tetrahedra): For each vertex, stores triplets of opposite face
// vertices For vertex v in tetrahedron (v, a, b, c), stores triplet (a, b, c)
// Input: triangle faces, CUDA will reconstruct tetrahedra topology
// Format: [count_v0, a1,b1,c1, a2,b2,c2, ... | count_v1, ... ]
// - adjacency_list: flattened triplets of opposite face vertices
// - offset_buffer: starting position for each vertex
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_volume_adjacency_gpu(
    cuda::CUDALinearBufferHandle vertices,
    cuda::CUDALinearBufferHandle triangles);  // triangle indices [v0,v1,v2, ...]

// Build edge set from triangles
// Returns: buffer of unique edges in format [v0, v1, v0, v1, ...]
RZSIM_CUDA_API
cuda::CUDALinearBufferHandle build_edge_set_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle edges);

// Compute rest lengths for springs/edges
// Returns: buffer of rest lengths, one per edge
RZSIM_CUDA_API
cuda::CUDALinearBufferHandle compute_rest_lengths_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle springs);

// Build adjacency list from triangles for mass-spring simulation
// Returns: (adjacent_vertices, vertex_offsets, rest_lengths)
// Format: adjacent_vertices[vertex_offsets[v]..vertex_offsets[v+1]] = neighbors of vertex v
// rest_lengths has the same length as adjacent_vertices
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
build_adjacency_list_gpu(
    cuda::CUDALinearBufferHandle triangles,
    cuda::CUDALinearBufferHandle positions,
    int num_particles);

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
