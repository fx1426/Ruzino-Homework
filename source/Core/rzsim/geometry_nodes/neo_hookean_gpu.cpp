#include <RHI/cuda.hpp>
#include <glm/glm.hpp>
#include <limits>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "RZSolver/Solver.hpp"
#include "glm/ext/vector_float3.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "rzsim_cuda/neo_hookean.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct NeoHookeanGPUStorage {
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
        float density)
    {
        num_particles = positions.size();

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

        // Initialize velocities to zero
        std::vector<glm::vec3> initial_velocities(
            num_particles, glm::vec3(0.0f));
        velocities_buffer = cuda::create_cuda_linear_buffer(initial_velocities);

        next_positions_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        gradients_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Mass matrix will be computed from density and volumes after reference
        // data is ready
        mass_matrix_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

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
        x_new_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        newton_direction_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        neg_gradient_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        x_candidate_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Allocate temporary buffers for energy computation
        inertial_terms_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        element_energies_buffer =
            cuda::create_cuda_linear_buffer<float>(num_elements);

        // Create solver instance
        solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUDA_CG);

        initialized = true;
    }
};

NODE_DECLARATION_FUNCTION(neo_hookean_gpu)
{
    b.add_input<Geometry>("Geometry");
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
    b.add_input<bool>("Flip Normal").default_val(false);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(neo_hookean_gpu)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<NeoHookeanGPUStorage&>();

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

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
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Convert Young's modulus and Poisson's ratio to Lamé parameters
    float mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    float lambda = youngs_modulus * poisson_ratio /
                   ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));

    // Get mesh component
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
    if (!storage.initialized || storage.num_particles != num_particles) {
        storage.initialize(
            positions, face_vertex_indices, face_counts, density);
    }

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

    // Convert vertex indices to DOF indices (each vertex has 3 DOFs: x, y, z)
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
            "[NeoHookean] Dirichlet BC applied to {} vertices ({} DOFs)",
            bc_vertices.size(),
            storage.num_bc_dofs);
    }
    else {
        spdlog::info("[NeoHookean] No Dirichlet boundary conditions");
    }

    if (!storage.initialized || storage.num_elements == 0) {
        spdlog::warn(
            "[NeoHookean] Neo-Hookean simulation requires tetrahedral mesh. "
            "Skipping simulation.");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;

    // Substep loop
    float dt_sub = dt / substeps;

    // Track statistics
    int max_newton_iterations = 0;
    int max_line_search_iterations = 0;

    for (int substep = 0; substep < substeps; ++substep) {
        // Compute x_tilde = x + dt_sub * v on GPU
        rzsim_cuda::explicit_step_nh_gpu(
            d_positions, d_velocities, dt_sub, num_particles, d_next_positions);

        // Newton's method iterations
        storage.x_new_buffer->copy_from_device(d_next_positions.Get());

        bool converged = false;
        int newton_iter_count = 0;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute negative gradient at current x_new (for Newton's method)
            rzsim_cuda::compute_neg_gradient_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
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
                storage.neg_gradient_buffer);

            float grad_norm = rzsim_cuda::compute_vector_norm_nh_gpu(
                storage.neg_gradient_buffer, num_particles * 3);

            if (!std::isfinite(grad_norm)) {
                spdlog::error(
                    "[NeoHookean] Gradient norm is not finite! Simulation "
                    "unstable.");
                break;
            }

            auto dof = num_particles * 3;
            grad_norm = grad_norm / dof;

            newton_iter_count = iter;

            // Run at least one iteration
            if (iter > 0 && grad_norm < tolerance) {
                converged = true;
                break;
            }

            // Update Hessian values
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

            // Apply Dirichlet boundary conditions to Hessian
            if (storage.num_bc_dofs > 0) {
                rzsim_cuda::apply_dirichlet_bc_to_hessian_gpu(
                    storage.hessian_structure,
                    storage.bc_dofs_buffer,
                    storage.num_bc_dofs,
                    storage.hessian_values);

                // Apply BC to gradient (set to zero for BC DOFs)
                rzsim_cuda::apply_dirichlet_bc_to_gradient_gpu(
                    storage.bc_dofs_buffer,
                    storage.num_bc_dofs,
                    storage.neg_gradient_buffer);
            }

            // Solve H * p = -grad using CUDA CG
            // Use tight tolerance for better symmetry preservation
            float cg_tol = std::max(1e-8f, grad_norm * 1e-3f);

            Ruzino::Solver::SolverConfig solver_config;
            solver_config.tolerance = cg_tol;
            solver_config.max_iterations =
                2000;  // Increased for tighter convergence
            solver_config.use_preconditioner = true;
            solver_config.verbose = false;

            // Zero out the solution buffer before solving
            cudaMemset(
                reinterpret_cast<void*>(
                    storage.newton_direction_buffer->get_device_ptr()),
                0,
                num_particles * 3 * sizeof(float));

            // Solve on GPU (neg_gradient_buffer already contains -gradient)
            auto result = storage.solver->solveGPU(
                storage.hessian_structure.num_rows,
                storage.hessian_structure.nnz,
                reinterpret_cast<const int*>(
                    storage.hessian_structure.row_offsets->get_device_ptr()),
                reinterpret_cast<const int*>(
                    storage.hessian_structure.col_indices->get_device_ptr()),
                reinterpret_cast<const float*>(
                    storage.hessian_values->get_device_ptr()),
                reinterpret_cast<const float*>(
                    storage.neg_gradient_buffer->get_device_ptr()),
                reinterpret_cast<float*>(
                    storage.newton_direction_buffer->get_device_ptr()),
                solver_config);

            if (!result.converged) {
                spdlog::warn(
                    "[NeoHookean] CG solver did not converge in iteration {}",
                    iter);
            }

            // Ensure BC DOFs have zero Newton direction
            if (storage.num_bc_dofs > 0) {
                rzsim_cuda::apply_dirichlet_bc_to_direction_gpu(
                    storage.bc_dofs_buffer,
                    storage.num_bc_dofs,
                    storage.newton_direction_buffer);
            }

            // Line search with energy descent
            float E_current = rzsim_cuda::compute_energy_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
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
            float alpha = 1.0f;  // Start with full Newton step
            int ls_iter = 0;

            while (E_candidate > E_current && ls_iter < 200) {
                // x_candidate = x_new + alpha * p
                rzsim_cuda::axpy_nh_gpu(
                    alpha,
                    storage.newton_direction_buffer,
                    storage.x_new_buffer,
                    storage.x_candidate_buffer,
                    num_particles * 3);

                E_candidate = rzsim_cuda::compute_energy_nh_gpu(
                    storage.x_candidate_buffer,
                    d_next_positions,
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
                // numerical precision This prevents getting stuck when gradient
                // is small and energy changes are negligible
                float energy_tolerance =
                    std::max(1e-6f, std::abs(E_current) * 1e-6f);
                bool accept = (E_candidate <= E_current) ||
                              (E_candidate - E_current < energy_tolerance);
                if (accept) {
                    storage.x_new_buffer->copy_from_device(
                        storage.x_candidate_buffer.Get());
                    break;
                }

                alpha *= 0.5f;
                ls_iter++;
            }

            if (ls_iter >= 200 || alpha < 1e-6f) {
                spdlog::warn(
                    "[NeoHookean]   Line search failed at iter {} (ls_iter={}, "
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
            spdlog::error(
                "[NeoHookean] Newton method FAILED to converge after {} "
                "iterations!",
                max_iterations);
            spdlog::error(
                "[NeoHookean] This indicates a serious problem with the "
                "simulation.");
            // Don't break - let the simulation continue but warn the user
        }

        // Update velocities on GPU: v = (x_new - x_old) / dt * damping
        rzsim_cuda::update_velocities_nh_gpu(
            storage.x_new_buffer,
            d_positions,
            dt_sub,
            damping,
            num_particles,
            d_velocities);

        // Handle ground collision on GPU
        rzsim_cuda::handle_ground_collision_nh_gpu(
            storage.x_new_buffer, d_velocities, restitution, num_particles);

        // Copy final positions back to position buffer
        d_positions->copy_from_device(storage.x_new_buffer.Get());
    }

    // Log simulation statistics
    spdlog::info(
        "[NeoHookean] Simulation complete - Max Newton iterations: {}, Max "
        "line search iterations: {}",
        max_newton_iterations,
        max_line_search_iterations);

    // Update geometry with new positions
    auto final_positions = d_positions->get_host_vector<glm::vec3>();

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

NODE_DECLARATION_UI(neo_hookean_gpu);
NODE_DEF_CLOSE_SCOPE
