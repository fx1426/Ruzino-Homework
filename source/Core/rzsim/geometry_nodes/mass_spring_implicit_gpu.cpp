#include <RHI/cuda.hpp>
#include <glm/glm.hpp>
#include <set>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "RZSolver/Solver.hpp"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct MassSpringImplicitGPUStorage {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle springs_buffer;
    cuda::CUDALinearBufferHandle rest_lengths_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;
    cuda::CUDALinearBufferHandle f_ext_buffer;

    size_t geom_hash = 0;

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
    b.add_input<float>("Damping").default_val(0.99f).min(0.0f).max(1.0f);
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
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance = std::max(tolerance, 1e-8f);
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    printf(
        "[GPU Params] mass=%.2f, k=%.1f, damp=%.3f, maxIter=%d, tol=%.2e, "
        "g=%.2f, rest=%.2f, dt=%.6f\\n",
        mass,
        stiffness,
        damping,
        max_iterations,
        tolerance,
        gravity,
        restitution,
        dt);

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

    if (input_geom.hash() != storage.geom_hash) {
        // Write positions to GPU buffer
        storage.positions_buffer = cuda::create_cuda_linear_buffer(positions);
        storage.velocities_buffer =
            cuda::create_cuda_linear_buffer<glm::vec3>(num_particles);
        storage.next_positions_buffer =
            cuda::create_cuda_linear_buffer<glm::vec3>(num_particles);
        storage.gradients_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        storage.f_ext_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create mass matrix (diagonal with mass value)
        std::vector<float> mass_diag(num_particles * 3, mass);
        storage.mass_matrix_buffer = cuda::create_cuda_linear_buffer(mass_diag);

        auto face_indices = mesh_component->get_face_vertex_indices();
        auto triangles = cuda::create_cuda_linear_buffer(face_indices);

        storage.springs_buffer =
            rzsim_cuda::build_edge_set_gpu(storage.positions_buffer, triangles);

        // Compute rest lengths from initial positions
        storage.rest_lengths_buffer = rzsim_cuda::compute_rest_lengths_gpu(
            storage.positions_buffer, storage.springs_buffer);

        storage.geom_hash = input_geom.hash();
    }

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_springs = storage.springs_buffer;
    auto d_rest_lengths = storage.rest_lengths_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;
    auto d_f_ext = storage.f_ext_buffer;

    spdlog::info(
        "[GPU] Implicit solver: {} particles, {} springs",
        num_particles,
        storage.springs_buffer->getDesc().element_count / 2);

    // Setup external forces on GPU
    rzsim_cuda::setup_external_forces_gpu(
        mass, gravity, num_particles, d_f_ext);

    // Compute x_tilde = x + dt * v on GPU
    rzsim_cuda::explicit_step_gpu(
        d_positions, d_velocities, dt, num_particles, d_next_positions);

    // Debug: print first 3 particles before simulation
    auto positions_debug = d_positions->get_host_vector<glm::vec3>();
    auto velocities_debug = d_velocities->get_host_vector<glm::vec3>();
    printf(
        "[GPU Before] p[0]=(%.6f, %.6f, %.6f), p[1]=(%.6f, %.6f, %.6f), "
        "p[2]=(%.6f, %.6f, %.6f)\\n",
        positions_debug[0].x,
        positions_debug[0].y,
        positions_debug[0].z,
        positions_debug[1].x,
        positions_debug[1].y,
        positions_debug[1].z,
        positions_debug[2].x,
        positions_debug[2].y,
        positions_debug[2].z);
    printf(
        "[GPU Before] v[0]=(%.6f, %.6f, %.6f), v[1]=(%.6f, %.6f, %.6f)\\n",
        velocities_debug[0].x,
        velocities_debug[0].y,
        velocities_debug[0].z,
        velocities_debug[1].x,
        velocities_debug[1].y,
        velocities_debug[1].z);

    // Compute gradient
    rzsim_cuda::compute_gradient_gpu(
        d_positions,
        d_next_positions,
        d_M_diag,
        d_f_ext,
        d_springs,
        d_rest_lengths,
        stiffness,
        dt,
        num_particles,
        d_gradients);

    // Newton's method iterations
    auto d_x_new = d_next_positions;  // Use next_positions as x_new
    auto d_p = cuda::create_cuda_linear_buffer<float>(num_particles * 3);  // Newton direction
    
    spdlog::info("[GPU] Starting Newton iterations, max_iter={}, tol={:.2e}", max_iterations, tolerance);
    
    bool converged = false;
    for (int iter = 0; iter < max_iterations; iter++) {
        spdlog::debug("[GPU] === Newton iteration {} ===", iter);
        
        // Compute gradient at current x_new
        rzsim_cuda::compute_gradient_gpu(
            d_x_new,
            d_next_positions,  // x_tilde (unchanged)
            d_M_diag,
            d_f_ext,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles,
            d_gradients);
        
        // Check gradient norm for convergence
        auto grad_host = d_gradients->get_host_vector<float>();
        float grad_inf_norm = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            grad_inf_norm = std::max(grad_inf_norm, std::abs(grad_host[i]));
        }
        
        spdlog::info("[GPU] Iteration {}: grad_inf_norm={:.6e}, grad_inf_norm/dt={:.6e}", 
                     iter, grad_inf_norm, grad_inf_norm / dt);
        
        if (!std::isfinite(grad_inf_norm)) {
            spdlog::error("[GPU] Gradient contains NaN/Inf at iteration {}", iter);
            break;
        }
        
        if (grad_inf_norm / dt < tolerance) {
            spdlog::info(
                "[GPU] Converged at iteration {} with grad_norm={:.6e}",
                iter,
                grad_inf_norm / dt);
            converged = true;
            break;
        }
        
        // Assemble Hessian matrix
        auto hessian = rzsim_cuda::assemble_hessian_gpu(
            d_x_new,
            d_M_diag,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles);
        
        if (iter == 0) {
            spdlog::info(
                "[GPU] Hessian ready: {} x {} with {} non-zeros (CSR format)",
                hessian.num_rows,
                hessian.num_cols,
                hessian.nnz);
        }
        
        // Solve H * p = -grad using CUDA CG
        auto solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUDA_CG);
        
        Ruzino::Solver::SolverConfig solver_config;
        solver_config.tolerance = 0.1f;  // Very loose tolerance for testing
        solver_config.max_iterations = 1000;
        solver_config.use_preconditioner = true;
        solver_config.verbose = (iter == 0);  // Verbose only for first iteration
        
        // Negate gradient for RHS: -grad
        auto d_neg_grad = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        std::vector<float> neg_grad_host(num_particles * 3);
        for (int i = 0; i < num_particles * 3; i++) {
            neg_grad_host[i] = -grad_host[i];  // Reuse grad_host from convergence check
        }
        d_neg_grad->assign_host_vector(neg_grad_host);
        
        // Solve on GPU
        auto result = solver->solveGPU(
            hessian.num_rows,
            hessian.nnz,
            reinterpret_cast<const int*>(hessian.row_offsets->get_device_ptr()),
            reinterpret_cast<const int*>(hessian.col_indices->get_device_ptr()),
            reinterpret_cast<const float*>(hessian.values->get_device_ptr()),
            reinterpret_cast<const float*>(d_neg_grad->get_device_ptr()),
            reinterpret_cast<float*>(d_p->get_device_ptr()),
            solver_config);
        
        if (!result.converged) {
            spdlog::warn(
                "[GPU] Newton solve failed at iteration {}: {} (iters={}, residual={:.6e})",
                iter,
                result.error_message,
                result.iterations,
                result.final_residual);
            break;
        }
        
        spdlog::debug(
            "[GPU] Newton iter {}: grad_norm={:.6e}, CG_iters={}, residual={:.6e}",
            iter,
            grad_inf_norm / dt,
            result.iterations,
            result.final_residual);
        
        // Line search with energy descent
        float E_current = rzsim_cuda::compute_energy_gpu(
            d_x_new,
            d_next_positions,  // x_tilde
            d_M_diag,
            d_f_ext,
            d_springs,
            d_rest_lengths,
            stiffness,
            dt,
            num_particles);
        
        float alpha = 1.0f;
        auto d_x_candidate = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        
        int ls_iter = 0;
        while (ls_iter < 20) {
            // x_candidate = x_new + alpha * p
            auto x_new_host = d_x_new->get_host_vector<float>();
            auto p_host = d_p->get_host_vector<float>();
            std::vector<float> x_cand_host(num_particles * 3);
            for (int i = 0; i < num_particles * 3; i++) {
                x_cand_host[i] = x_new_host[i] + alpha * p_host[i];
            }
            d_x_candidate->assign_host_vector(x_cand_host);
            
            float E_candidate = rzsim_cuda::compute_energy_gpu(
                d_x_candidate,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                d_springs,
                d_rest_lengths,
                stiffness,
                dt,
                num_particles);
            
            if (E_candidate <= E_current || alpha < 1e-8f) {
                if (alpha < 1e-8f && E_candidate > E_current) {
                    spdlog::warn("[GPU] Line search failed to reduce energy");
                }
                else {
                    spdlog::debug(
                        "[GPU] Line search: alpha={:.3e}, E: {:.6e} -> {:.6e}",
                        alpha,
                        E_current,
                        E_candidate);
                }
                // Accept step
                auto x_cand_final = d_x_candidate->get_host_vector<float>();
                d_x_new->assign_host_vector(x_cand_final);
                break;
            }
            
            alpha *= 0.5f;
            ls_iter++;
        }
    }
    
    if (!converged) {
        spdlog::warn("[GPU] Newton method did not converge in {} iterations", max_iterations);
    }
    
    // Update velocities: v = (x_new - x_n) / dt
    auto x_new_final = d_x_new->get_host_vector<float>();
    auto x_n_host = d_positions->get_host_vector<float>();
    std::vector<float> v_new(num_particles * 3);
    for (int i = 0; i < num_particles * 3; i++) {
        v_new[i] = (x_new_final[i] - x_n_host[i]) / dt * damping;
    }
    d_velocities->assign_host_vector(v_new);
    d_positions->assign_host_vector(x_new_final);
    
    // Debug print
    auto positions_debug_after = d_positions->get_host_vector<glm::vec3>();
    auto velocities_debug_after = d_velocities->get_host_vector<glm::vec3>();
    printf(
        "[GPU After]  p[0]=(%.6f, %.6f, %.6f), p[1]=(%.6f, %.6f, %.6f), p[2]=(%.6f, %.6f, %.6f)\n",
        positions_debug_after[0].x,
        positions_debug_after[0].y,
        positions_debug_after[0].z,
        positions_debug_after[1].x,
        positions_debug_after[1].y,
        positions_debug_after[1].z,
        positions_debug_after[2].x,
        positions_debug_after[2].y,
        positions_debug_after[2].z);
    printf(
        "[GPU After]  v[0]=(%.6f, %.6f, %.6f), v[1]=(%.6f, %.6f, %.6f)\n",
        velocities_debug_after[0].x,
        velocities_debug_after[0].y,
        velocities_debug_after[0].z,
        velocities_debug_after[1].x,
        velocities_debug_after[1].y,
        velocities_debug_after[1].z);
    
    // Convert to output format
    std::vector<glm::vec3> new_positions = positions_debug_after;
    std::vector<float> positions_out(3 * num_particles);
    std::vector<float> velocities_out(3 * num_particles);

    // new_positions already populated from GPU buffers above

    // Update geometry with new positions
    if (mesh_component) {
        mesh_component->set_vertices(new_positions);

        // Recalculate normals
        std::vector<glm::vec3> normals;
        normals.reserve(face_vertex_indices.size());

        int idx = 0;
        for (int face_count : face_counts) {
            if (face_count >= 3) {
                int i0 = face_vertex_indices[idx];
                int i1 = face_vertex_indices[idx + 1];
                int i2 = face_vertex_indices[idx + 2];

                glm::vec3 edge1 = new_positions[i1] - new_positions[i0];
                glm::vec3 edge2 = new_positions[i2] - new_positions[i0];
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
        points_component->set_vertices(new_positions);
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(mass_spring_implicit_gpu);
NODE_DEF_CLOSE_SCOPE
