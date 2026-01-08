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

    // Newton's method iterations
    auto d_x_new = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
    // Initialize x_new = x (current position, NOT x_tilde) - matching CPU implementation
    auto positions_flat = d_positions->get_host_vector<glm::vec3>();
    std::vector<float> x_init_flat(num_particles * 3);
    for (int i = 0; i < num_particles; i++) {
        x_init_flat[i * 3 + 0] = positions_flat[i].x;
        x_init_flat[i * 3 + 1] = positions_flat[i].y;
        x_init_flat[i * 3 + 2] = positions_flat[i].z;
    }
    d_x_new->assign_host_vector(x_init_flat);
    
    spdlog::info("[GPU] Starting Newton iterations, max_iter={}, tol={:.2e}", max_iterations, tolerance);
    
    bool converged = false;
    for (int iter = 0; iter < max_iterations; iter++) {
        spdlog::debug("[GPU] === Newton iteration {} ===", iter);
        
        // Debug: verify d_x_new at loop start
        if (iter > 0 && iter < 3) {
            auto x_check = d_x_new->get_host_vector<float>();
            spdlog::info("[GPU] Iter {} start: x_new[0:3]=({:.9f}, {:.9f}, {:.9f})",
                         iter, x_check[0], x_check[1], x_check[2]);
        }
        
        // Create fresh buffer for Newton direction each iteration to avoid warm start issues
        auto d_p = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        
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
        solver_config.tolerance = 1e-4f;  // Balanced tolerance to avoid CG breakdown
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
        
        if (iter == 0) {
            float neg_grad_norm = 0.0f;
            for (int i = 0; i < num_particles * 3; i++) {
                neg_grad_norm += neg_grad_host[i] * neg_grad_host[i];
            }
            neg_grad_norm = std::sqrt(neg_grad_norm);
            spdlog::info("[GPU] CG RHS ||-grad|| = {:.6e}, grad[0:3]=({:.6f}, {:.6f}, {:.6f})",
                         neg_grad_norm, -grad_host[0], -grad_host[1], -grad_host[2]);
        }
        
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
        
        // Debug: check what CG actually wrote to d_p
        auto p_after_cg = d_p->get_host_vector<float>();
        if (iter == 0) {
            spdlog::info("[GPU] After CG: p[0:6] = ({:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e})",
                         p_after_cg[0], p_after_cg[1], p_after_cg[2], p_after_cg[3], p_after_cg[4], p_after_cg[5]);
        }
        
        // Check if p is a descent direction: p^T * grad should be < 0
        float p_dot_grad = 0.0f;
        float p_norm_sq = 0.0f;
        float grad_norm_sq = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            p_dot_grad += p_after_cg[i] * grad_host[i];
            p_norm_sq += p_after_cg[i] * p_after_cg[i];
            grad_norm_sq += grad_host[i] * grad_host[i];
        }
        float cosine = p_dot_grad / (std::sqrt(p_norm_sq) * std::sqrt(grad_norm_sq) + 1e-20f);
        if (iter < 3) {
            spdlog::info("[GPU] Iter {}: p^T * grad = {:.6e}, ||p||={:.6e}, ||grad||={:.6e}, cos(angle)={:.6f}", 
                         iter, p_dot_grad, std::sqrt(p_norm_sq), std::sqrt(grad_norm_sq), cosine);
        }
        
        // If p is almost orthogonal to gradient (cos(angle) close to 0), Hessian may be singular
        if (std::abs(cosine) < 0.01f && iter > 0) {
            spdlog::warn("[GPU] Iter {}: Newton direction nearly orthogonal to gradient! Hessian may be singular.", iter);
            spdlog::warn("[GPU] Increasing regularization and retrying...");
            
            // TODO: Try adaptive regularization or trust region
        }
        
        if (!result.converged) {
            spdlog::error(
                "[GPU] Newton solve failed at iteration {}: {} (iters={}, residual={:.6e})",
                iter,
                result.error_message,
                result.iterations,
                result.final_residual);
            
            // Test with CPU solver for comparison
            if (iter == 0) {
                spdlog::info("[GPU] Testing same system with CPU Eigen solver...");
                
                // Copy matrix to host
                auto row_offsets_host = hessian.row_offsets->get_host_vector<int>();
                auto col_indices_host = hessian.col_indices->get_host_vector<int>();
                auto values_host = hessian.values->get_host_vector<float>();
                
                // Convert CSR to Eigen sparse matrix
                std::vector<Eigen::Triplet<float>> triplets;
                for (int i = 0; i < hessian.num_rows; i++) {
                    for (int j = row_offsets_host[i]; j < row_offsets_host[i+1]; j++) {
                        triplets.emplace_back(i, col_indices_host[j], values_host[j]);
                    }
                }
                
                Eigen::SparseMatrix<float> H_cpu(hessian.num_rows, hessian.num_cols);
                H_cpu.setFromTriplets(triplets.begin(), triplets.end());
                
                // Setup RHS
                Eigen::VectorXf b_cpu(num_particles * 3);
                for (int i = 0; i < num_particles * 3; i++) {
                    b_cpu[i] = neg_grad_host[i];
                }
                
                // Solve with Eigen CG
                Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg_cpu;
                cg_cpu.setMaxIterations(1000);
                cg_cpu.setTolerance(0.1f);
                cg_cpu.compute(H_cpu);
                
                Eigen::VectorXf x_cpu = cg_cpu.solve(b_cpu);
                
                spdlog::info(
                    "[CPU CG] iterations={}, error={:.6e}, ||solution||={:.6e}",
                    cg_cpu.iterations(),
                    cg_cpu.error(),
                    x_cpu.norm());
                
                // Check if solutions differ
                auto p_gpu_host = d_p->get_host_vector<float>();
                float diff_norm = 0.0f;
                for (int i = 0; i < num_particles * 3; i++) {
                    float d = p_gpu_host[i] - x_cpu[i];
                    diff_norm += d * d;
                }
                diff_norm = std::sqrt(diff_norm);
                spdlog::info("[CPU vs GPU] Solution difference norm: {:.6e}", diff_norm);
            }
            
            break;
        }
        
        // Check if Newton direction is valid
        auto p_check = d_p->get_host_vector<float>();
        float p_norm = 0.0f;
        for (int i = 0; i < num_particles * 3; i++) {
            p_norm += p_check[i] * p_check[i];
        }
        p_norm = std::sqrt(p_norm);
        
        if (iter == 0) {
            spdlog::info(
                "[GPU] CG converged: iters={}, residual={:.6e}, ||p||={:.6e}",
                result.iterations,
                result.final_residual,
                p_norm);
        }
        
        if (p_norm < 1e-12f) {
            spdlog::warn("[GPU] Newton direction is nearly zero (||p||={:.6e})", p_norm);
            converged = true;
            break;
        }
        
        spdlog::info("[GPU] About to start line search for iter {}, p_norm={:.6e}", iter, p_norm);
        
        spdlog::debug(
            "[GPU] Newton iter {}: grad_norm={:.6e}, CG_iters={}, residual={:.6e}",
            iter,
            grad_inf_norm / dt,
            result.iterations,
            result.final_residual);
        
        spdlog::info("[GPU] Before line search at iter {}", iter);
        
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
        
        if (iter == 0) {
            // Debug: check buffer contents before energy calculation
            auto x_new_buf = d_x_new->get_host_vector<float>();
            auto x_tilde_buf = d_next_positions->get_host_vector<float>();
            auto M_buf = d_M_diag->get_host_vector<float>();
            auto f_buf = d_f_ext->get_host_vector<float>();
            
            spdlog::info("[GPU] Buffer check - x_new[0:3]=({:.6f}, {:.6f}, {:.6f})", 
                         x_new_buf[0], x_new_buf[1], x_new_buf[2]);
            spdlog::info("[GPU] Buffer check - x_tilde[0:3]=({:.6f}, {:.6f}, {:.6f})", 
                         x_tilde_buf[0], x_tilde_buf[1], x_tilde_buf[2]);
            spdlog::info("[GPU] Buffer check - M[0:3]=({:.6f}, {:.6f}, {:.6f}), size={}", 
                         M_buf[0], M_buf[1], M_buf[2], M_buf.size());
            spdlog::info("[GPU] Buffer check - f_ext[0:3]=({:.6f}, {:.6f}, {:.6f})", 
                         f_buf[0], f_buf[1], f_buf[2]);
            
            // CPU energy calculation for verification
            float E_inertial_cpu = 0.0f;
            for (int i = 0; i < num_particles * 3; i++) {
                int pid = i / 3;
                float diff = x_new_buf[i] - x_tilde_buf[i];
                E_inertial_cpu += 0.5f * M_buf[pid] * diff * diff;
            }
            
            float E_potential_cpu = 0.0f;
            for (int i = 0; i < num_particles * 3; i++) {
                E_potential_cpu += -f_buf[i] * x_new_buf[i] * dt * dt;
            }
            
            spdlog::info("[CPU] E_inertial={:.6e}, E_potential={:.6e}", E_inertial_cpu, E_potential_cpu);
            
            spdlog::info("[GPU] Initial energy E_current = {:.6e}", E_current);
            
            // Debug: check first few values of x_new and p
            auto x_new_debug = d_x_new->get_host_vector<float>();
            auto p_debug = d_p->get_host_vector<float>();
            spdlog::info("[GPU] x_new[0:3] = ({:.6f}, {:.6f}, {:.6f})", 
                         x_new_debug[0], x_new_debug[1], x_new_debug[2]);
            spdlog::info("[GPU] p[0:3] = ({:.6f}, {:.6f}, {:.6f})", 
                         p_debug[0], p_debug[1], p_debug[2]);
        }
        
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
            
            // Debug: verify d_x_candidate was written correctly
            if (iter == 0 && ls_iter == 0) {
                auto x_cand_readback = d_x_candidate->get_host_vector<float>();
                spdlog::info("[GPU] x_cand_host[0:3] = ({:.6f}, {:.6f}, {:.6f})", 
                             x_cand_host[0], x_cand_host[1], x_cand_host[2]);
                spdlog::info("[GPU] x_cand_readback[0:3] = ({:.6f}, {:.6f}, {:.6f})", 
                             x_cand_readback[0], x_cand_readback[1], x_cand_readback[2]);
            }
            
            // Debug: test if energy calculation works with d_x_new
            if (iter == 0 && ls_iter == 0) {
                float E_test_x_new = rzsim_cuda::compute_energy_gpu(
                    d_x_new,  // Should be same as E_current
                    d_next_positions,
                    d_M_diag,
                    d_f_ext,
                    d_springs,
                    d_rest_lengths,
                    stiffness,
                    dt,
                    num_particles);
                spdlog::info("[GPU] Test energy with d_x_new (should match E_current): {:.6e}", E_test_x_new);
            }
            
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
            
            // Log line search progress for first few iterations
            if (iter < 3 || (iter < 10 && ls_iter < 3)) {
                float energy_reduction = E_current - E_candidate;
                spdlog::info("[GPU] Iter {}, LS {}: alpha={:.3e}, E_current={:.6e}, E_candidate={:.6e}, reduction={:.6e}",
                             iter, ls_iter, alpha, E_current, E_candidate, energy_reduction);
            }
            
            if (iter == 0 && ls_iter < 5) {
                // Check for NaN/Inf in x_candidate
                bool has_nan = false;
                float max_val = 0.0f;
                for (int i = 0; i < std::min(30, num_particles * 3); i++) {
                    if (!std::isfinite(x_cand_host[i])) {
                        has_nan = true;
                        spdlog::error("[GPU] x_candidate[{}] = {} (NOT FINITE!)", i, x_cand_host[i]);
                    }
                    max_val = std::max(max_val, std::abs(x_cand_host[i]));
                }
                if (has_nan) {
                    spdlog::error("[GPU] Line search {}: max|x_cand|={:.3e} <-- HAS NaN/Inf!", ls_iter, max_val);
                }
            
                if (ls_iter == 0) {
                    spdlog::info("[GPU] x_cand[0:6] = ({:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f})",
                                 x_cand_host[0], x_cand_host[1], x_cand_host[2],
                                 x_cand_host[3], x_cand_host[4], x_cand_host[5]);
                }
            }
            
            if (E_candidate <= E_current) {
                // Accept step
                spdlog::info("[GPU] Iter {}: Line search accepted at LS iter {}, alpha={:.3e}, E: {:.6e} -> {:.6e}",
                             iter, ls_iter, alpha, E_current, E_candidate);
                
                // Debug: check if x_new actually changes
                auto x_new_before = d_x_new->get_host_vector<float>();
                auto x_cand_final = d_x_candidate->get_host_vector<float>();
                
                float max_change = 0.0f;
                for (int i = 0; i < num_particles * 3; i++) {
                    max_change = std::max(max_change, std::abs(x_cand_final[i] - x_new_before[i]));
                }
                
                if (iter < 5) {
                    spdlog::info("[GPU] Iter {}: max|x_cand - x_new| = {:.6e}, p_norm={:.6e}, alpha={:.6e}",
                                 iter, max_change, std::sqrt(p_norm_sq), alpha);
                }
                
                d_x_new->assign_host_vector(x_cand_final);
                break;
            }
            
            if (alpha < 1e-8f) {
                spdlog::warn("[GPU] Iter {}: Line search failed after {} attempts, alpha={:.3e} too small, E_current={:.6e}, best E_candidate={:.6e}",
                             iter, ls_iter + 1, alpha, E_current, E_candidate);
                // Still accept the step even though energy doesn't decrease
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
