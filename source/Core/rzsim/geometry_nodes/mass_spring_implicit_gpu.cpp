#include <RHI/cuda.hpp>
#include <glm/glm.hpp>
#include <limits>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "RZSolver/Solver.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct MassSpringImplicitGPUStorage {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle adjacent_vertices_buffer;
    cuda::CUDALinearBufferHandle vertex_offsets_buffer;
    cuda::CUDALinearBufferHandle rest_lengths_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;
    cuda::CUDALinearBufferHandle f_ext_buffer;

    // Edge offsets for Hessian structure
    cuda::CUDALinearBufferHandle edge_offsets_buffer;

    // NEW: Pre-built CSR structure (built once, reused forever)
    rzsim_cuda::CSRStructure hessian_structure;
    cuda::CUDALinearBufferHandle
        hessian_values;  // Only values change each iteration

    // Temporary buffers for Newton iterations (reused across iterations)
    cuda::CUDALinearBufferHandle x_new_buffer;
    cuda::CUDALinearBufferHandle newton_direction_buffer;
    cuda::CUDALinearBufferHandle neg_gradient_buffer;
    cuda::CUDALinearBufferHandle x_candidate_buffer;

    // Temporary buffers for energy computation (reused)
    cuda::CUDALinearBufferHandle inertial_terms_buffer;
    cuda::CUDALinearBufferHandle spring_energies_buffer;
    cuda::CUDALinearBufferHandle potential_terms_buffer;

    // Reuse solver instance across iterations
    std::unique_ptr<Ruzino::Solver::LinearSolver> solver;

    bool initialized = false;
    int num_particles = 0;

    constexpr static bool has_storage = false;
};

NODE_DECLARATION_FUNCTION(mass_spring_implicit_gpu)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Mass").default_val(1.0f).min(0.01f).max(100.0f);
    b.add_input<float>("Stiffness")
        .default_val(1000.0f)
        .min(1.0f)
        .max(10000.0f);
    b.add_input<float>("Damping").default_val(1.0f).min(0.0f).max(1.0f);
    b.add_input<int>("Substeps").default_val(1).min(1).max(20);
    b.add_input<int>("Newton Iterations").default_val(30).min(1).max(100);
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

NODE_EXECUTION_FUNCTION(mass_spring_implicit_gpu)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<MassSpringImplicitGPUStorage&>();

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    float mass = params.get_input<float>("Mass");
    float stiffness = params.get_input<float>("Stiffness");
    float damping = params.get_input<float>("Damping");
    int substeps = params.get_input<int>("Substeps");
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance = std::max(tolerance, 1e-8f);
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_counts;

    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
        face_counts = mesh_component->get_face_vertex_counts();
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
        // Write positions to GPU buffer
        storage.positions_buffer = cuda::create_cuda_linear_buffer(positions);

        // Initialize velocities to zero
        std::vector<glm::vec3> initial_velocities(
            num_particles, glm::vec3(0.0f));
        storage.velocities_buffer =
            cuda::create_cuda_linear_buffer(initial_velocities);

        storage.next_positions_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.gradients_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.f_ext_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create mass matrix (diagonal with mass value per DOF, matching CPU)
        std::vector<float> mass_diag(num_particles * 3, mass);
        storage.mass_matrix_buffer = cuda::create_cuda_linear_buffer(mass_diag);

        auto face_indices = mesh_component->get_face_vertex_indices();
        auto triangles = cuda::create_cuda_linear_buffer(face_indices);

        // Build adjacency list directly from triangles
        auto [adjacent_vertices, vertex_offsets, rest_lengths] =
            rzsim_cuda::build_adjacency_list_gpu(
                triangles, storage.positions_buffer, num_particles);

        storage.adjacent_vertices_buffer = adjacent_vertices;
        storage.vertex_offsets_buffer = vertex_offsets;
        storage.rest_lengths_buffer = rest_lengths;

        // NEW: Build CSR structure once (this is the key optimization!)
        // Sparsity pattern is fixed, so we only need to update values later
        storage.hessian_structure = rzsim_cuda::build_hessian_structure_gpu(
            storage.adjacent_vertices_buffer,
            storage.vertex_offsets_buffer,
            num_particles);

        // Allocate values buffer (will be filled each iteration)
        storage.hessian_values = cuda::create_cuda_linear_buffer<float>(
            storage.hessian_structure.nnz);

        // Allocate edge_offsets_buffer for Hessian updates
        // This is computed during hessian structure build, we need to recreate
        // it
        auto d_edge_counts =
            cuda::create_cuda_linear_buffer<int>(num_particles);
        // Count edges where j > i for each vertex (simplified host version for
        // initialization)
        std::vector<int> edge_counts_host(num_particles, 0);
        auto vertex_offsets_host =
            storage.vertex_offsets_buffer->get_host_vector<int>();
        auto adjacent_vertices_host =
            storage.adjacent_vertices_buffer->get_host_vector<int>();

        for (int i = 0; i < num_particles; i++) {
            int start = vertex_offsets_host[i];
            int end = vertex_offsets_host[i + 1];
            for (int idx = start; idx < end; idx++) {
                int j = adjacent_vertices_host[idx];
                if (j > i)
                    edge_counts_host[i]++;
            }
        }

        // Compute prefix sum for edge offsets
        std::vector<int> edge_offsets_host(num_particles + 1);
        edge_offsets_host[0] = 0;
        for (int i = 0; i < num_particles; i++) {
            edge_offsets_host[i + 1] =
                edge_offsets_host[i] + edge_counts_host[i];
        }

        storage.edge_offsets_buffer =
            cuda::create_cuda_linear_buffer(edge_offsets_host);

        // Allocate temporary buffers for Newton iterations (reused)
        storage.x_new_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.newton_direction_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.neg_gradient_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.x_candidate_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Allocate temporary buffers for energy computation
        storage.inertial_terms_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        // Spring energies are computed per-vertex (not per-adjacency)
        storage.spring_energies_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles);
        storage.potential_terms_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create solver instance once
        storage.solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUDA_CG);

        storage.initialized = true;
        storage.num_particles = num_particles;
    }

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_rest_lengths = storage.rest_lengths_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;
    auto d_f_ext = storage.f_ext_buffer;

    // Substep loop
    float dt_sub = dt / substeps;
    for (int substep = 0; substep < substeps; ++substep) {
        // Setup external forces on GPU
        rzsim_cuda::setup_external_forces_gpu(
            mass, gravity, num_particles, d_f_ext);

        // Compute x_tilde = x + dt_sub * v on GPU
        rzsim_cuda::explicit_step_gpu(
            d_positions, d_velocities, dt_sub, num_particles, d_next_positions);

        // Newton's method iterations
        // Initialize x_new = x_tilde (predictive position) for better
        // convergence
        storage.x_new_buffer->copy_from_device(d_next_positions.Get());

        bool converged = false;
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute gradient at current x_new
            rzsim_cuda::compute_gradient_gpu(
                storage.x_new_buffer,
                d_next_positions,  // x_tilde (unchanged)
                d_M_diag,
                d_f_ext,
                storage.adjacent_vertices_buffer,
                storage.vertex_offsets_buffer,
                d_rest_lengths,
                stiffness,
                dt_sub,
                num_particles,
                d_gradients);

            // Check gradient norm for convergence (computed on GPU)
            float grad_norm = rzsim_cuda::compute_vector_norm_gpu(
                d_gradients, num_particles * 3);

            // Check for convergence
            if (!std::isfinite(grad_norm)) {
                break;
            }

            auto dof = num_particles * 3;

            grad_norm = grad_norm / dof;

            // Converge when gradient is 1/1000 of initial gradient
            if (iter > 0 && grad_norm < tolerance) {
                converged = true;
                break;
            }

            // NEW: Fast Hessian update (NO SORTING!)
            // Directly fill values into pre-built CSR structure
            rzsim_cuda::update_hessian_values_gpu(
                storage.hessian_structure,
                storage.x_new_buffer,
                d_M_diag,
                storage.adjacent_vertices_buffer,
                storage.vertex_offsets_buffer,
                storage.edge_offsets_buffer,
                d_rest_lengths,
                stiffness,
                dt_sub,
                num_particles,
                storage.hessian_values);

            // Solve H * p = -grad using CUDA CG (reuse solver)
            // Adaptive CG tolerance based on gradient magnitude
            // CG residual should be 0.1% of gradient norm, but not too small
            float cg_tol = std::max(1e-9f, grad_norm * 1e-3f);

            Ruzino::Solver::SolverConfig solver_config;
            solver_config.tolerance = cg_tol;
            solver_config.max_iterations = 1000;
            solver_config.use_preconditioner = true;
            solver_config.verbose = false;

            // Negate gradient for RHS: -grad (do on GPU)
            rzsim_cuda::negate_gpu(
                d_gradients, storage.neg_gradient_buffer, num_particles * 3);

            // Solve on GPU
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
                break;
            }

            // Line search with energy descent
            float E_current = rzsim_cuda::compute_energy_gpu(
                storage.x_new_buffer,
                d_next_positions,  // x_tilde
                d_M_diag,
                d_f_ext,
                storage.adjacent_vertices_buffer,
                storage.vertex_offsets_buffer,
                d_rest_lengths,
                stiffness,
                dt_sub,
                num_particles,
                storage.inertial_terms_buffer,
                storage.spring_energies_buffer,
                storage.potential_terms_buffer);

            float alpha = 1.0f;
            int ls_iter = 0;
            float E_candidate =
                std::numeric_limits<float>::infinity();  // Start with +infinity
                                                         // so first check
                                                         // passes

            while (E_candidate > E_current && ls_iter < 50) {
                // x_candidate = x_new + alpha * p (computed on GPU)
                rzsim_cuda::axpy_gpu(
                    alpha,
                    storage.newton_direction_buffer,
                    storage.x_new_buffer,
                    storage.x_candidate_buffer,
                    num_particles * 3);

                E_candidate = rzsim_cuda::compute_energy_gpu(
                    storage.x_candidate_buffer,
                    d_next_positions,
                    d_M_diag,
                    d_f_ext,
                    storage.adjacent_vertices_buffer,
                    storage.vertex_offsets_buffer,
                    d_rest_lengths,
                    stiffness,
                    dt_sub,
                    num_particles,
                    storage.inertial_terms_buffer,
                    storage.spring_energies_buffer,
                    storage.potential_terms_buffer);

                if (E_candidate <= E_current) {
                    // Accept step - copy result directly on GPU
                    // Copy d_x_candidate to d_x_new on GPU
                    float* x_cand_ptr =
                        storage.x_candidate_buffer->get_device_ptr<float>();
                    float* x_new_ptr =
                        storage.x_new_buffer->get_device_ptr<float>();
                    cudaMemcpy(
                        x_new_ptr,
                        x_cand_ptr,
                        num_particles * 3 * sizeof(float),
                        cudaMemcpyDeviceToDevice);
                    break;
                }

                alpha *= 0.5f;
                ls_iter++;
            }

            if (alpha < 1e-4f) {
                spdlog::warn("Line search failed to find a descent direction.");
            }
        }

        // Update velocities: v = (x_new - x_n) / dt_sub and apply damping
        auto x_new_final = storage.x_new_buffer->get_host_vector<float>();
        auto x_n_host = d_positions->get_host_vector<glm::vec3>();
        std::vector<glm::vec3> v_new(num_particles);
        for (int i = 0; i < num_particles; i++) {
            v_new[i].x =
                (x_new_final[i * 3 + 0] - x_n_host[i].x) / dt_sub * damping;
            v_new[i].y =
                (x_new_final[i * 3 + 1] - x_n_host[i].y) / dt_sub * damping;
            v_new[i].z =
                (x_new_final[i * 3 + 2] - x_n_host[i].z) / dt_sub * damping;
        }

        // Handle ground collision (z = 0)
        int num_collisions = 0;
        for (int i = 0; i < num_particles; i++) {
            if (x_new_final[i * 3 + 2] < 0.0f) {  // Penetrating ground
                // Project position to ground
                x_new_final[i * 3 + 2] = 0.0f;

                // Apply collision response to velocity
                if (v_new[i].z < 0.0f) {  // Moving downward
                    v_new[i].z = -v_new[i].z * restitution;
                    float friction = 0.8f;
                    v_new[i].x *= friction;
                    v_new[i].y *= friction;
                }
                num_collisions++;
            }
        }

        // Convert to output format
        std::vector<glm::vec3> new_positions(num_particles);
        for (int i = 0; i < num_particles; i++) {
            new_positions[i].x = x_new_final[i * 3 + 0];
            new_positions[i].y = x_new_final[i * 3 + 1];
            new_positions[i].z = x_new_final[i * 3 + 2];
        }

        d_velocities->assign_host_vector(v_new);
        d_positions->assign_host_vector(new_positions);
    }  // End substep loop

    // Update geometry with new positions
    auto final_positions = d_positions->get_host_vector<glm::vec3>();
    if (mesh_component) {
        mesh_component->set_vertices(final_positions);

        // Recalculate normals
        std::vector<glm::vec3> normals;
        normals.reserve(face_vertex_indices.size());

        int idx = 0;
        for (int face_count : face_counts) {
            if (face_count >= 3) {
                int i0 = face_vertex_indices[idx];
                int i1 = face_vertex_indices[idx + 1];
                int i2 = face_vertex_indices[idx + 2];

                glm::vec3 edge1 = final_positions[i1] - final_positions[i0];
                glm::vec3 edge2 = final_positions[i2] - final_positions[i0];
                glm::vec3 normal = glm::cross(
                    flip_normal ? edge1 : edge2, flip_normal ? edge2 : edge1);

                float length = glm::length(normal);
                if (length > 1e-8f) {
                    normal = normal / length;
                }
                else {
                    normal = glm::vec3(0.0f, 0.0f, 1.0f);
                }

                for (int i = 0; i < face_count; ++i) {
                    normals.push_back(normal);
                }
            }
            idx += face_count;
        }

        if (!normals.empty()) {
            mesh_component->set_normals(normals);
        }
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        points_component->set_vertices(final_positions);
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(mass_spring_implicit_gpu);
NODE_DEF_CLOSE_SCOPE
