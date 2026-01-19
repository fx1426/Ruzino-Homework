#include <RHI/cuda.hpp>
#include <algorithm>
#include <glm/glm.hpp>
#include <limits>
#include <numeric>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "RZSolver/Solver.hpp"
#include "glm/ext/vector_float3.hpp"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "rzsim/reduced_order_basis.h"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "rzsim_cuda/neo_hookean.cuh"
#include "rzsim_cuda/reduced_order_neo_hookean.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct ReducedNeoHookeanGPUStorageIPC {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;

    // Mesh topology buffers (cached)
    cuda::CUDALinearBufferHandle face_vertex_indices_buffer;
    cuda::CUDALinearBufferHandle face_counts_buffer;
    cuda::CUDALinearBufferHandle normals_buffer;

    // Volume adjacency map encapsulates all tetrahedral topology
    std::unique_ptr<rzsim_cuda::VolumeAdjacencyMap> volume_adjacency;

    // Neo-Hookean specific reference data
    cuda::CUDALinearBufferHandle Dm_inv_buffer;
    cuda::CUDALinearBufferHandle volumes_buffer;

    // Pre-built CSR structure (built once, reused forever)
    rzsim_cuda::NeoHookeanCSRStructure hessian_structure;
    cuda::CUDALinearBufferHandle hessian_values;

    // Temporary buffers for Newton iterations
    cuda::CUDALinearBufferHandle x_new_buffer;
    cuda::CUDALinearBufferHandle newton_direction_buffer;
    cuda::CUDALinearBufferHandle neg_gradient_buffer;
    cuda::CUDALinearBufferHandle x_candidate_buffer;

    // Temporary buffers for energy computation
    cuda::CUDALinearBufferHandle inertial_terms_buffer;
    cuda::CUDALinearBufferHandle element_energies_buffer;
    cuda::CUDALinearBufferHandle potential_terms_buffer;

    // Reduced order model data
    rzsim_cuda::ReducedOrderData ro_data;
    cuda::CUDALinearBufferHandle q_reduced;        // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_dot_reduced;    // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_tilde_reduced;  // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_new_reduced;    // [num_basis * 12]
    cuda::CUDALinearBufferHandle
        jacobian;  // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle grad_reduced;  // [num_basis * 12]
    cuda::CUDALinearBufferHandle
        hessian_reduced;  // [num_basis * 12, num_basis * 12]
    cuda::CUDALinearBufferHandle
        temp_hessian_buffer;  // [num_particles * 3, num_basis * 12]
    cuda::CUDALinearBufferHandle newton_direction_reduced;  // [num_basis * 12]
    cuda::CUDALinearBufferHandle neg_gradient_reduced;      // [num_basis * 12]
    cuda::CUDALinearBufferHandle q_candidate_reduced;       // [num_basis * 12]
    int num_basis = 0;

    // Reuse solver instance
    std::unique_ptr<Ruzino::Solver::LinearSolver> solver;

    // Dirichlet boundary conditions
    cuda::CUDALinearBufferHandle bc_dofs_buffer;  // DOF indices with BC
    int num_bc_dofs = 0;
    std::vector<int> bc_dofs;  // Host copy for reference

    bool initialized = false;
    int num_particles = 0;
    int num_elements = 0;

    constexpr static bool has_storage = false;

    // Initialize all GPU buffers and structures
    void initialize(
        const std::vector<glm::vec3>& positions,
        const std::vector<int>& face_vertex_indices,
        const std::vector<int>& face_counts,
        float density,
        std::shared_ptr<Ruzino::ReducedOrderedBasis> reduced_basis)
    {
        num_particles = positions.size();

        // Validate reduced basis
        if (!reduced_basis || reduced_basis->basis.empty()) {
            spdlog::error("[ReducedNeoHookean] Reduced basis is empty or null");
            return;
        }

        num_basis = reduced_basis->basis.size();
        spdlog::debug("[ReducedNeoHookean] Using {} basis modes", num_basis);

        // Validate basis dimensions
        for (int i = 0; i < num_basis; ++i) {
            if (reduced_basis->basis[i].size() != num_particles) {
                spdlog::error(
                    "[ReducedNeoHookean] Basis {} size mismatch: {} vs {}",
                    i,
                    reduced_basis->basis[i].size(),
                    num_particles);
                return;
            }
        }

        // Write positions to GPU buffer
        positions_buffer = cuda::create_cuda_linear_buffer(positions);
        // Cache face topology buffers
        face_vertex_indices_buffer =
            cuda::create_cuda_linear_buffer(face_vertex_indices);
        face_counts_buffer = cuda::create_cuda_linear_buffer(face_counts);
        normals_buffer = cuda::create_cuda_linear_buffer<glm::vec3>(
            face_vertex_indices.size());

        // Build volume adjacency map (encapsulates all tetrahedral topology)
        volume_adjacency = std::make_unique<rzsim_cuda::VolumeAdjacencyMap>(
            positions_buffer, face_vertex_indices);

        num_elements = volume_adjacency->num_elements();

        if (num_elements == 0) {
            spdlog::error(
                "No tetrahedral elements found! Neo-Hookean requires "
                "volumetric mesh.");
            return;
        }

        // Compute Neo-Hookean specific reference data (Dm_inv and volumes)
        auto [Dm_inv, volumes] = rzsim_cuda::compute_nh_reference_data_gpu(
            positions_buffer, *volume_adjacency, num_elements);

        Dm_inv_buffer = Dm_inv;
        volumes_buffer = volumes;

        // Diagnostic: Check volumes
        auto volumes_host = volumes->get_host_vector<float>();
        float min_volume =
            *std::min_element(volumes_host.begin(), volumes_host.end());
        float max_volume =
            *std::max_element(volumes_host.begin(), volumes_host.end());
        float total_volume =
            std::accumulate(volumes_host.begin(), volumes_host.end(), 0.0f);
        spdlog::info(
            "[ReducedNeoHookean] Volume statistics: min={:.6e}, max={:.6e}, "
            "total={:.6e}, avg={:.6e}",
            min_volume,
            max_volume,
            total_volume,
            total_volume / num_elements);

        // Check for problematic volumes
        int num_small = std::count_if(
            volumes_host.begin(), volumes_host.end(), [](float v) {
                return v < 1e-10f;
            });
        if (num_small > 0) {
            spdlog::warn(
                "[ReducedNeoHookean] Found {} degenerate tetrahedra (volume < "
                "1e-10)",
                num_small);
        }

        // Initialize velocities to zero
        std::vector<glm::vec3> initial_velocities(
            num_particles, glm::vec3(0.0f));
        velocities_buffer = cuda::create_cuda_linear_buffer(initial_velocities);

        auto dof = num_particles * 3;

        next_positions_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        gradients_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        mass_matrix_buffer = cuda::create_cuda_linear_buffer<float>(dof);

        // Compute lumped mass matrix from density and element volumes
        // For each element: m_elem = density * volume
        // Distribute equally to 4 vertices: m_vertex += m_elem / 4
        rzsim_cuda::compute_lumped_mass_matrix_gpu(
            *volume_adjacency,
            volumes_buffer,
            density,
            num_particles,
            num_elements,
            mass_matrix_buffer);

        // Build Hessian CSR structure
        hessian_structure = rzsim_cuda::build_hessian_structure_nh_gpu(
            *volume_adjacency, num_particles, num_elements);

        hessian_values =
            cuda::create_cuda_linear_buffer<float>(hessian_structure.nnz);

        // Allocate temporary buffers for Newton iterations
        x_new_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        newton_direction_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        neg_gradient_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        x_candidate_buffer = cuda::create_cuda_linear_buffer<float>(dof);

        // Allocate temporary buffers for energy computation
        inertial_terms_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        element_energies_buffer =
            cuda::create_cuda_linear_buffer<float>(num_elements);
        potential_terms_buffer = cuda::create_cuda_linear_buffer<float>(dof);
        // Build reduced order data
        ro_data = rzsim_cuda::build_reduced_order_data_gpu(
            &reduced_basis->basis, &positions);

        // Allocate reduced coordinate buffers (num_basis * 12 DOF for affine
        // transforms)
        int reduced_dof = num_basis * 12;
        q_reduced = cuda::create_cuda_linear_buffer<float>(reduced_dof);
        q_dot_reduced = cuda::create_cuda_linear_buffer<float>(reduced_dof);
        q_tilde_reduced = cuda::create_cuda_linear_buffer<float>(reduced_dof);
        q_new_reduced = cuda::create_cuda_linear_buffer<float>(reduced_dof);
        grad_reduced = cuda::create_cuda_linear_buffer<float>(reduced_dof);
        newton_direction_reduced =
            cuda::create_cuda_linear_buffer<float>(reduced_dof);
        neg_gradient_reduced =
            cuda::create_cuda_linear_buffer<float>(reduced_dof);
        q_candidate_reduced =
            cuda::create_cuda_linear_buffer<float>(reduced_dof);

        // Allocate Jacobian and Hessian buffers
        jacobian = cuda::create_cuda_linear_buffer<float>(dof * reduced_dof);
        hessian_reduced =
            cuda::create_cuda_linear_buffer<float>(reduced_dof * reduced_dof);
        temp_hessian_buffer =
            cuda::create_cuda_linear_buffer<float>(dof * reduced_dof);

        // Initialize reduced coordinates to identity (R=I, t=0 for each basis)
        rzsim_cuda::initialize_reduced_coords_identity_gpu(
            num_basis, q_reduced);

        // Initialize reduced velocities to zero
        cudaMemset(
            reinterpret_cast<void*>(q_dot_reduced->get_device_ptr()),
            0,
            reduced_dof * sizeof(float));

        // Create solver instance
        solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUSOLVER_CHOLESKY);

        spdlog::debug(
            "[ReducedNeoHookean] Initialized with {} particles, {} elements, "
            "{} basis modes, {} reduced DOF",
            num_particles,
            num_elements,
            num_basis,
            num_basis * 12);

        initialized = true;
    }
};

NODE_DECLARATION_FUNCTION(reduced_order_neo_hookean_ipc)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<Geometry>("Init Geometry");
    b.add_input<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
    b.add_input<std::shared_ptr<Ruzino::AffineTransform>>("Initial Transform");

    b.add_input<float>("Density").default_val(1000.0f).min(1.0f).max(10000.0f);
    b.add_input<float>("Young's Modulus").default_val(5e4f).min(1e3f).max(1e9f);
    b.add_input<float>("Poisson's Ratio")
        .default_val(0.35f)
        .min(0.0f)
        .max(0.49f);
    b.add_input<float>("Damping").default_val(0.99f).min(0.0f).max(1.0f);
    b.add_input<int>("Substeps").default_val(5).min(1).max(20);
    b.add_input<int>("Newton Iterations").default_val(50).min(1).max(100);
    b.add_input<float>("Newton Tolerance")
        .default_val(1e-2f)
        .min(1e-8f)
        .max(1e-1f);
    b.add_input<float>("Gravity").default_val(-9.81f).min(-20.0f).max(0.0f);
    b.add_input<float>("Ground Restitution")
        .default_val(0.3f)
        .min(0.0f)
        .max(1.0f);

    b.add_input<float>("Allowed Width").min(0.0001).max(0.1).default_val(0.01);
    b.add_input<bool>("Consider BC").default_val(false);
    b.add_input<bool>("Flip Normal").default_val(false);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(reduced_order_neo_hookean_ipc)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<ReducedNeoHookeanGPUStorageIPC&>();

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    auto init_geom = params.get_input<Geometry>("Init Geometry");
    init_geom.apply_transform();

    auto reduced_basis =
        params.get_input<std::shared_ptr<Ruzino::ReducedOrderedBasis>>(
            "Reduced Basis");

    auto initial_transform =
        params.get_input<std::shared_ptr<Ruzino::AffineTransform>>(
            "Initial Transform");

    float density = params.get_input<float>("Density");
    float youngs_modulus = params.get_input<float>("Young's Modulus");
    float poisson_ratio = params.get_input<float>("Poisson's Ratio");
    float damping = params.get_input<float>("Damping");
    int substeps = params.get_input<int>("Substeps");
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance =
        std::max(tolerance, 1e-10f);  // Use very tight tolerance for symmetry
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool consider_bc = params.get_input<bool>("Consider BC");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Convert Young's modulus and Poisson's ratio to Lamé parameters
    float mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    float lambda = youngs_modulus * poisson_ratio /
                   ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));

    // Get mesh component from init geometry for initialization
    auto init_mesh_component = init_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> init_positions;

    if (init_mesh_component) {
        init_positions = init_mesh_component->get_vertices();
    }
    else {
        auto init_points_component = init_geom.get_component<PointsComponent>();
        if (init_points_component) {
            init_positions = init_points_component->get_vertices();
        }
    }

    // Get mesh component for topology
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_counts;
    std::vector<float> dirichlet_face_values;  // Face quantity for BC

    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
        face_counts = mesh_component->get_face_vertex_counts();
        // Try to get dirichlet face quantity
        dirichlet_face_values =
            mesh_component->get_face_scalar_quantity("dirichlet");
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        if (!points_component) {
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }
        positions = points_component->get_vertices();
    }

    int num_particles = positions.size();
    if (num_particles == 0) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Initialize buffers only once or when particle count changes
    // ALWAYS use rest pose (input_geom positions) for reference configuration
    if (!storage.initialized || storage.num_particles != num_particles) {
        storage.initialize(
            positions,  // Use rest pose for Dm_inv, volumes calculation
            face_vertex_indices,
            face_counts,
            density,
            reduced_basis);
    }

    // If Init Geometry is provided, use it as the starting point for simulation
    if (!init_positions.empty()) {
        // Check topology consistency
        bool topology_matches = (init_positions.size() == positions.size());

        if (topology_matches) {
            // Write init positions to GPU buffer as simulation starting point
            storage.positions_buffer->assign_host_vector(init_positions);
            spdlog::info(
                "[ReducedNeoHookean] Using Init Geometry as simulation "
                "starting point (vertices={})",
                init_positions.size());
        }
        else {
            spdlog::warn(
                "[ReducedNeoHookean] Init Geometry topology mismatch! "
                "Init: {} vertices; Rest pose: {} vertices. Using rest pose as "
                "starting point.",
                init_positions.size(),
                positions.size());
        }
    }

    // Apply initial transform on the first frame
    if (global_payload.is_simulating == false && initial_transform) {
        // Update Dirichlet boundary conditions from face quantities
        // Find all vertices that belong to faces marked as dirichlet
        std::set<int> bc_vertices;
        if (!dirichlet_face_values.empty() &&
            dirichlet_face_values.size() == face_counts.size()) {
            int face_idx = 0;
            int vertex_offset = 0;
            for (int face = 0; face < face_counts.size(); ++face) {
                // If this face is marked as dirichlet (non-zero value)
                if (dirichlet_face_values[face] > 0.5f) {
                    int num_verts = face_counts[face];
                    for (int v = 0; v < num_verts; ++v) {
                        int vert_idx = face_vertex_indices[vertex_offset + v];
                        bc_vertices.insert(vert_idx);
                    }
                }
                vertex_offset += face_counts[face];
            }
        }

        // Convert vertex indices to DOF indices (each vertex has 3 DOFs: x, y,
        // z)
        storage.bc_dofs.clear();
        for (int v : bc_vertices) {
            storage.bc_dofs.push_back(v * 3 + 0);  // x DOF
            storage.bc_dofs.push_back(v * 3 + 1);  // y DOF
            storage.bc_dofs.push_back(v * 3 + 2);  // z DOF
        }
        storage.num_bc_dofs = storage.bc_dofs.size();

        // Upload BC DOFs to GPU
        if (storage.num_bc_dofs > 0) {
            storage.bc_dofs_buffer =
                cuda::create_cuda_linear_buffer(storage.bc_dofs);
            spdlog::info(
                "[ReducedNeoHookean] Dirichlet BC applied to {} vertices ({} "
                "DOFs)",
                bc_vertices.size(),
                storage.num_bc_dofs);
        }
        else {
            spdlog::info(
                "[ReducedNeoHookean] No Dirichlet boundary conditions");
        }

        int reduced_dof = storage.num_basis * 12;

        // Validate that transform has the right number of modes
        if (initial_transform->num_modes() != storage.num_basis) {
            spdlog::warn(
                "[ReducedNeoHookean] Initial transform has {} modes but "
                "reduced basis has {} modes",
                initial_transform->num_modes(),
                storage.num_basis);
        }
        else {
            auto q_host = storage.q_reduced->get_host_vector<float>();

            // For each basis, copy the transform from initial_transform
            for (int i = 0; i < storage.num_basis; ++i) {
                const auto& transform = initial_transform->get_transform(i);

                // Copy all 12 DOF (rotation matrix + translation)
                for (int j = 0; j < 12; ++j) {
                    q_host[i * 12 + j] = transform[j];
                }
            }

            storage.q_reduced->assign_host_vector(q_host);
            spdlog::info(
                "[ReducedNeoHookean] Applied initial transform to all {} bases",
                storage.num_basis);
        }
    }

    if (!storage.initialized || storage.num_elements == 0) {
        spdlog::warn(
            "[NeoHookean] Neo-Hookean simulation requires tetrahedral mesh. "
            "Skipping simulation.");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    auto d_positions = storage.positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;

    // Substep loop
    float dt_sub = dt / substeps;

    // Track statistics
    int max_newton_iterations = 0;
    int max_line_search_iterations = 0;

    spdlog::debug(
        "[ReducedNeoHookean] Starting simulation with {} reduced DOF",
        storage.num_basis);
    spdlog::debug(
        "[ReducedNeoHookean] dt={:.4f}, substeps={}, gravity={:.2f}",
        dt,
        substeps,
        gravity);

    for (int substep = 0; substep < substeps; ++substep) {
        if (substep == 0) {
            spdlog::debug(
                "[ReducedNeoHookean] Substep {}/{}, dt_sub={:.4f}",
                substep + 1,
                substeps,
                dt_sub);
        }

        // Compute q_tilde = q + dt_sub * q_dot in reduced space
        rzsim_cuda::explicit_step_reduced_gpu(
            storage.q_reduced,
            storage.q_dot_reduced,
            dt_sub,
            storage.num_basis,
            storage.q_tilde_reduced);

        // Newton's method iterations in reduced space
        storage.q_new_reduced->copy_from_device(storage.q_tilde_reduced.Get());

        bool converged = false;
        int newton_iter_count = 0;

        // Log initial energy before Newton iterations
        if (substep == 0) {
            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_new_reduced, storage.ro_data, storage.x_new_buffer);

            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_tilde_reduced,
                storage.ro_data,
                storage.next_positions_buffer);

            float initial_energy = rzsim_cuda::compute_energy_nh_gpu(
                storage.x_new_buffer,
                storage.next_positions_buffer,
                d_M_diag,
                *storage.volume_adjacency,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                density,
                gravity,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.inertial_terms_buffer,
                storage.element_energies_buffer);

            spdlog::debug(
                "[ReducedNeoHookean] Initial energy before Newton: {:.6f}",
                initial_energy);
        }

        for (int iter = 0; iter < max_iterations; iter++) {
            // Map q_new to full space positions
            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_new_reduced, storage.ro_data, storage.x_new_buffer);

            // Compute Jacobian J = dx/dq
            rzsim_cuda::compute_jacobian_gpu(
                storage.q_new_reduced, storage.ro_data, storage.jacobian);

            // Compute gradient in full space
            // For reduced order, we don't use velocities in the inertial term
            // Instead, we use q_tilde mapped to full space
            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_tilde_reduced,
                storage.ro_data,
                storage.next_positions_buffer);

            // Compute negative gradient in full space (for Newton's method)
            rzsim_cuda::compute_neg_gradient_nh_gpu(
                storage.x_new_buffer,
                storage.next_positions_buffer,
                d_M_diag,
                *storage.volume_adjacency,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                density,
                gravity,
                dt_sub,
                num_particles,
                storage.num_elements,
                d_gradients);

            // Apply Dirichlet boundary conditions to full-space gradient
            if (consider_bc && storage.num_bc_dofs > 0) {
                rzsim_cuda::apply_dirichlet_bc_to_gradient_gpu(
                    storage.bc_dofs_buffer, storage.num_bc_dofs, d_gradients);
            }

            // Project negative gradient to reduced space: -grad_q = J^T *
            // (-grad_x) Since we already have negative gradient, just use
            // regular projection
            rzsim_cuda::compute_reduced_gradient_gpu(
                storage.jacobian,
                d_gradients,
                num_particles,
                storage.num_basis,
                storage.neg_gradient_reduced);

            // Compute gradient norm in reduced space (norm of -g equals norm of
            // g)
            int reduced_dof = storage.num_basis * 12;
            float grad_norm = rzsim_cuda::compute_vector_norm_nh_gpu(
                storage.neg_gradient_reduced, reduced_dof);

            if (!std::isfinite(grad_norm)) {
                spdlog::error(
                    "[ReducedNeoHookean] Gradient norm is not finite! "
                    "Simulation "
                    "unstable.");
                break;
            }

            grad_norm = grad_norm / reduced_dof;

            newton_iter_count = iter;

            // Log first few iterations for debugging
            if (substep == 0 && iter < 3) {
                spdlog::debug(
                    "[ReducedNeoHookean]   Newton iter {}: grad_norm={:.6e}",
                    iter,
                    grad_norm);
            }

            // Run at least one iteration
            if (iter > 0 && grad_norm < tolerance) {
                converged = true;
                if (substep == 0) {
                    spdlog::debug(
                        "[ReducedNeoHookean]   Converged at iteration {} with "
                        "grad_norm={:.6e}",
                        iter,
                        grad_norm);
                }
                break;
            }

            // Update Hessian values in full space
            rzsim_cuda::update_hessian_values_nh_gpu(
                storage.hessian_structure,
                storage.x_new_buffer,
                d_M_diag,
                *storage.volume_adjacency,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.hessian_values);

            // Apply Dirichlet boundary conditions to full-space Hessian
            if (consider_bc && storage.num_bc_dofs > 0) {
                rzsim_cuda::apply_dirichlet_bc_to_hessian_gpu(
                    storage.hessian_structure,
                    storage.bc_dofs_buffer,
                    storage.num_bc_dofs,
                    storage.hessian_values);
            }

            // Project Hessian to reduced space: H_q = J^T * H_x * J
            // Use unmodified Jacobian (BC constraints already in H_x)
            rzsim_cuda::compute_reduced_hessian_gpu(
                storage.hessian_structure,
                storage.hessian_values,
                storage.jacobian,
                num_particles,
                storage.num_basis,
                storage.temp_hessian_buffer,
                storage.hessian_reduced);

            // Solve H_q * p = -grad_q using cuSOLVER dense Cholesky
            // H_q is dense symmetric positive definite [reduced_dof x
            // reduced_dof]
            Ruzino::Solver::SolverConfig solver_config;
            solver_config.tolerance = std::max(1e-8f, grad_norm * 1e-3f);
            solver_config.verbose = false;

            auto solver_result = storage.solver->solveDenseGPU(
                reduced_dof,
                storage.hessian_reduced->get_device_ptr<float>(),
                storage.neg_gradient_reduced->get_device_ptr<float>(),
                storage.newton_direction_reduced->get_device_ptr<float>(),
                solver_config);

            if (!solver_result.converged) {
                spdlog::warn(
                    "[ReducedNeoHookean] Dense solver failed at Newton iter "
                    "{}: {}",
                    iter,
                    solver_result.error_message);
                // Fall back to gradient descent
                storage.newton_direction_reduced->copy_from_device(
                    storage.neg_gradient_reduced.Get());
            }
            else if (substep == 0 && iter < 3) {
                spdlog::debug(
                    "[ReducedNeoHookean]   Solver: {} μs",
                    solver_result.solve_time.count());
            }

            // Line search with energy descent
            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_new_reduced, storage.ro_data, storage.x_new_buffer);

            float E_current = rzsim_cuda::compute_energy_nh_gpu(
                storage.x_new_buffer,
                storage.next_positions_buffer,
                d_M_diag,
                *storage.volume_adjacency,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                density,
                gravity,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.inertial_terms_buffer,
                storage.element_energies_buffer);

            float E_candidate = std::numeric_limits<float>::infinity();
            float alpha = 1.0f;  // Start with full step
            int ls_iter = 0;

            while (E_candidate > E_current && ls_iter < 200) {
                // q_candidate = q_new + alpha * p
                rzsim_cuda::axpy_nh_gpu(
                    alpha,
                    storage.newton_direction_reduced,
                    storage.q_new_reduced,
                    storage.q_candidate_reduced,
                    reduced_dof);

                // Map to full space and compute energy
                rzsim_cuda::map_reduced_to_full_gpu(
                    storage.q_candidate_reduced,
                    storage.ro_data,
                    storage.x_candidate_buffer);

                E_candidate = rzsim_cuda::compute_energy_nh_gpu(
                    storage.x_candidate_buffer,
                    storage.next_positions_buffer,
                    d_M_diag,
                    *storage.volume_adjacency,
                    storage.Dm_inv_buffer,
                    storage.volumes_buffer,
                    mu,
                    lambda,
                    density,
                    gravity,
                    dt_sub,
                    num_particles,
                    storage.num_elements,
                    storage.inertial_terms_buffer,
                    storage.element_energies_buffer);

                // Accept step if energy decreases OR if the increase is within
                // numerical precision
                float energy_tolerance =
                    std::max(1e-6f, std::abs(E_current) * 1e-6f);
                bool accept = (E_candidate <= E_current) ||
                              (E_candidate - E_current < energy_tolerance);
                if (accept) {
                    storage.q_new_reduced->copy_from_device(
                        storage.q_candidate_reduced.Get());
                    break;
                }

                alpha *= 0.5f;
                ls_iter++;
            }

            if (ls_iter >= 200 || alpha < 1e-6f) {
                spdlog::warn(
                    "[ReducedNeoHookean] Line search failed at iter {} "
                    "(ls_iter={}, "
                    "alpha={:.6e})",
                    iter,
                    ls_iter,
                    alpha);
                break;
            }

            // Update max line search iterations
            max_line_search_iterations =
                std::max(max_line_search_iterations, ls_iter);
        }

        // Update max Newton iterations
        max_newton_iterations =
            std::max(max_newton_iterations, newton_iter_count);

        // Check if Newton method converged
        if (!converged) {
            spdlog::warn(
                "[ReducedNeoHookean] Newton method did not converge after {} "
                "iterations",
                max_iterations);
        }

        // Update reduced velocities: q_dot = (q_new - q_old) / dt * damping
        rzsim_cuda::update_reduced_velocities_gpu(
            storage.q_new_reduced,
            storage.q_reduced,
            dt_sub,
            damping,
            storage.num_basis,
            storage.q_dot_reduced);

        // Copy final reduced coordinates back
        storage.q_reduced->copy_from_device(storage.q_new_reduced.Get());
    }

    // Map final reduced coordinates to full space positions
    rzsim_cuda::map_reduced_to_full_gpu(
        storage.q_reduced, storage.ro_data, d_positions);

    // Apply Dirichlet BC to positions (enforce BC vertices at rest pose)
    if (consider_bc && storage.num_bc_dofs > 0) {
        // Save current positions before BC
        cuda::CUDALinearBufferHandle positions_before_bc =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        positions_before_bc->copy_from_device(d_positions.Get());

        // Apply BC: set BC vertices to rest pose
        rzsim_cuda::apply_bc_to_positions_gpu(
            storage.bc_dofs_buffer,
            storage.num_bc_dofs,
            d_positions,
            storage.ro_data.rest_positions,
            num_particles);

        // Now we need to find q_new such that map(q_new) ≈ positions (with BC
        // enforced) Use local linearization: positions ≈ positions_before_bc +
        // J * delta_q Solve: (J^T * J) * delta_q = J^T * (positions -
        // positions_before_bc)

        int reduced_dof = storage.num_basis * 12;

        // Compute Jacobian at current q
        rzsim_cuda::compute_jacobian_gpu(
            storage.q_reduced, storage.ro_data, storage.jacobian);

        // Compute delta_x = positions - positions_before_bc
        cuda::CUDALinearBufferHandle delta_x =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // delta_x = 1.0 * positions - 1.0 * positions_before_bc
        cudaMemcpy(
            delta_x->get_device_ptr<float>(),
            d_positions->get_device_ptr<float>(),
            num_particles * 3 * sizeof(float),
            cudaMemcpyDeviceToDevice);
        rzsim_cuda::axpy_nh_gpu(
            -1.0f, positions_before_bc, delta_x, delta_x, num_particles * 3);

        // Compute rhs = J^T * delta_x
        rzsim_cuda::compute_reduced_gradient_gpu(
            storage.jacobian,
            delta_x,
            num_particles,
            storage.num_basis,
            storage.grad_reduced);

        // Compute J^T * J
        cuda::CUDALinearBufferHandle JtJ_pos =
            cuda::create_cuda_linear_buffer<float>(reduced_dof * reduced_dof);
        rzsim_cuda::compute_jacobian_gram_matrix_gpu(
            storage.jacobian, num_particles, storage.num_basis, JtJ_pos);

        // Solve (J^T * J) * delta_q = J^T * delta_x
        cuda::CUDALinearBufferHandle delta_q =
            cuda::create_cuda_linear_buffer<float>(reduced_dof);

        Ruzino::Solver::SolverConfig solver_config_pos;
        solver_config_pos.tolerance = 1e-9f;
        solver_config_pos.verbose = false;

        auto solver_result_pos = storage.solver->solveDenseGPU(
            reduced_dof,
            JtJ_pos->get_device_ptr<float>(),
            storage.grad_reduced->get_device_ptr<float>(),
            delta_q->get_device_ptr<float>(),
            solver_config_pos);

        if (solver_result_pos.converged) {
            spdlog::info(
                "[ReducedNeoHookean] Position BC projection converged "
                "(iter={}, residual={})",
                solver_result_pos.iterations,
                solver_result_pos.final_residual);

            // Update q: q_new = q + delta_q
            // First copy current q to temp
            cuda::CUDALinearBufferHandle q_temp =
                cuda::create_cuda_linear_buffer<float>(reduced_dof);
            cudaMemcpy(
                q_temp->get_device_ptr<float>(),
                storage.q_reduced->get_device_ptr<float>(),
                reduced_dof * sizeof(float),
                cudaMemcpyDeviceToDevice);

            // q = q + delta_q
            rzsim_cuda::axpy_nh_gpu(
                1.0f, delta_q, q_temp, storage.q_reduced, reduced_dof);

            // Remap to get final positions with BC enforced
            rzsim_cuda::map_reduced_to_full_gpu(
                storage.q_reduced, storage.ro_data, d_positions);
        }
        else {
            spdlog::warn(
                "[ReducedNeoHookean] Position BC projection solver failed: {}",
                solver_result_pos.error_message);
        }
    }

    if (consider_bc) {
        // Project reduced velocities to full space before collision handling
        // v_full = J * q_dot
        rzsim_cuda::compute_jacobian_gpu(
            storage.q_reduced, storage.ro_data, storage.jacobian);

        // Save original q_dot for verification (before collision)
        int reduced_dof = storage.num_basis * 12;

        cuda::CUDALinearBufferHandle q_dot_original =
            cuda::create_cuda_linear_buffer<float>(reduced_dof);
        q_dot_original->copy_from_device(storage.q_dot_reduced.Get());

        rzsim_cuda::map_reduced_velocities_to_full_gpu(
            storage.jacobian,
            storage.q_dot_reduced,
            num_particles,
            storage.num_basis,
            storage.velocities_buffer);

        // Apply Dirichlet BC to full-space velocities (set BC vertices to zero
        // velocity)
        if (storage.num_bc_dofs > 0) {
            rzsim_cuda::apply_dirichlet_bc_to_velocities_gpu(
                storage.bc_dofs_buffer,
                storage.num_bc_dofs,
                storage.velocities_buffer,
                num_particles);
        }

        // Save full-space velocity before collision for verification
        cuda::CUDALinearBufferHandle velocities_before_collision =
            cuda::create_cuda_linear_buffer<glm::vec3>(num_particles);
        velocities_before_collision->copy_from_device(
            storage.velocities_buffer.Get());

        // Handle ground collision in full space
        rzsim_cuda::handle_ground_collision_nh_gpu(
            d_positions, storage.velocities_buffer, restitution, num_particles);

        // Project modified full-space velocities back to reduced space
        // Solve: (J^T * J) * q_dot = J^T * v_full for proper projection
        // First compute rhs = J^T * v_full
        rzsim_cuda::compute_reduced_gradient_gpu(
            storage.jacobian,
            storage.velocities_buffer,
            num_particles,
            storage.num_basis,
            storage.grad_reduced);

        // Compute J^T * J (Gram matrix / reduced "mass matrix")
        cuda::CUDALinearBufferHandle JtJ =
            cuda::create_cuda_linear_buffer<float>(reduced_dof * reduced_dof);

        rzsim_cuda::compute_jacobian_gram_matrix_gpu(
            storage.jacobian, num_particles, storage.num_basis, JtJ);

        // Solve (J^T * J) * q_dot = J^T * v_full using dense Cholesky
        Ruzino::Solver::SolverConfig solver_config_vel;
        solver_config_vel.tolerance = 1e-9f;
        solver_config_vel.verbose = false;

        auto solver_result_vel = storage.solver->solveDenseGPU(
            reduced_dof,
            JtJ->get_device_ptr<float>(),
            storage.grad_reduced->get_device_ptr<float>(),
            storage.q_dot_reduced->get_device_ptr<float>(),
            solver_config_vel);

        if (!solver_result_vel.converged) {
            spdlog::error(
                "[ReducedNeoHookean] Velocity projection solver failed: {}",
                solver_result_vel.error_message);
            // Fall back to simple J^T projection
            storage.q_dot_reduced->copy_from_device(storage.grad_reduced.Get());
        }

        else {
            spdlog::debug(
                "[ReducedNeoHookean] Collision occurred, velocity projection "
                "skipped");
        }
    }
    // Log simulation statistics
    spdlog::info(
        "[ReducedNeoHookean] Simulation complete - Max Newton iterations: "
        "{}, "
        "Max line search iterations: {}",
        max_newton_iterations,
        max_line_search_iterations);

    // Get final positions for geometry update
    auto final_positions = d_positions->get_host_vector<glm::vec3>();

    // Update geometry with new positions
    if (mesh_component) {
        mesh_component->set_vertices(final_positions);

        // Recalculate normals on GPU
        rzsim_cuda::compute_normals_gpu(
            storage.positions_buffer,
            storage.face_vertex_indices_buffer,
            storage.face_counts_buffer,
            flip_normal,
            storage.normals_buffer);

        auto normals = storage.normals_buffer->get_host_vector<glm::vec3>();
        mesh_component->set_normals(normals);
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        points_component->set_vertices(final_positions);
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(reduced_order_neo_hookean_ipc);
NODE_DEF_CLOSE_SCOPE
