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
#include "rzsim_cuda/neo_hookean.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct NeoHookeanGPUStorage {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle tetrahedra_buffer;
    cuda::CUDALinearBufferHandle Dm_inv_buffer;
    cuda::CUDALinearBufferHandle volumes_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;
    cuda::CUDALinearBufferHandle f_ext_buffer;

    // Mesh topology buffers (cached)
    cuda::CUDALinearBufferHandle face_vertex_indices_buffer;
    cuda::CUDALinearBufferHandle face_counts_buffer;
    cuda::CUDALinearBufferHandle normals_buffer;

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

    // Reuse solver instance
    std::unique_ptr<Ruzino::Solver::LinearSolver> solver;

    bool initialized = false;
    int num_particles = 0;
    int num_elements = 0;

    constexpr static bool has_storage = false;

    // Extract tetrahedra from mesh faces
    std::vector<glm::ivec4> extract_tetrahedra_from_faces(
        const std::vector<int>& face_vertex_indices,
        const std::vector<int>& face_counts)
    {
        std::vector<glm::ivec4> tetrahedra;
        
        spdlog::error("[NeoHookean] Tetrahedral mesh extraction from surface mesh is not yet implemented.");
        spdlog::error("[NeoHookean] Please provide a proper tetrahedral mesh (.tet format) or use TetGen.");
        spdlog::error("[NeoHookean] For now, returning empty tetrahedra list.");
        
        // The current surface-to-tet conversion creates degenerate elements
        // which cause matrix inversion failures in the FEM simulation.
        // A proper implementation would need:
        // 1. Use TetGen or similar library to generate interior tetrahedra
        // 2. Or load a pre-generated tetrahedral mesh from file
        // 3. Or use a constrained Delaunay tetrahedralization
        
        return tetrahedra;
    }

    // Initialize all GPU buffers and structures
    void initialize(
        const std::vector<glm::vec3>& positions,
        const std::vector<int>& face_vertex_indices,
        const std::vector<int>& face_counts,
        float mass)
    {
        num_particles = positions.size();

        // Extract tetrahedra from faces (or load from tetrahedral mesh)
        auto tetrahedra = extract_tetrahedra_from_faces(face_vertex_indices, face_counts);
        num_elements = tetrahedra.size();

        if (num_elements == 0) {
            spdlog::error("No tetrahedral elements found! Neo-Hookean requires volumetric mesh.");
            return;
        }

        // Write positions to GPU buffer
        positions_buffer = cuda::create_cuda_linear_buffer(positions);

        // Initialize velocities to zero
        std::vector<glm::vec3> initial_velocities(num_particles, glm::vec3(0.0f));
        velocities_buffer = cuda::create_cuda_linear_buffer(initial_velocities);

        next_positions_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        gradients_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        f_ext_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create mass matrix (diagonal with mass value per DOF)
        std::vector<float> mass_diag(num_particles * 3, mass);
        mass_matrix_buffer = cuda::create_cuda_linear_buffer(mass_diag);

        // Upload tetrahedra
        std::vector<int> tet_flat(num_elements * 4);
        for (int i = 0; i < num_elements; i++) {
            tet_flat[i * 4 + 0] = tetrahedra[i].x;
            tet_flat[i * 4 + 1] = tetrahedra[i].y;
            tet_flat[i * 4 + 2] = tetrahedra[i].z;
            tet_flat[i * 4 + 3] = tetrahedra[i].w;
        }
        tetrahedra_buffer = cuda::create_cuda_linear_buffer(tet_flat);

        // Compute reference shape matrices and volumes
        auto [Dm_inv, volumes] = rzsim_cuda::compute_reference_data_gpu(
            positions_buffer, tetrahedra_buffer, num_elements);
        Dm_inv_buffer = Dm_inv;
        volumes_buffer = volumes;

        // Cache face topology buffers
        face_vertex_indices_buffer = cuda::create_cuda_linear_buffer(face_vertex_indices);
        face_counts_buffer = cuda::create_cuda_linear_buffer(face_counts);
        normals_buffer = cuda::create_cuda_linear_buffer<glm::vec3>(face_vertex_indices.size());

        // Build CSR structure once
        hessian_structure = rzsim_cuda::build_hessian_structure_nh_gpu(
            tetrahedra_buffer, num_particles, num_elements);

        // Allocate values buffer
        hessian_values = cuda::create_cuda_linear_buffer<float>(hessian_structure.nnz);

        // Allocate temporary buffers for Newton iterations
        x_new_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        newton_direction_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        neg_gradient_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        x_candidate_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Allocate temporary buffers for energy computation
        inertial_terms_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        element_energies_buffer = cuda::create_cuda_linear_buffer<float>(num_elements);
        potential_terms_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create solver instance
        solver = Ruzino::Solver::SolverFactory::create(Ruzino::Solver::SolverType::CUDA_CG);

        initialized = true;
    }
};

NODE_DECLARATION_FUNCTION(neo_hookean_gpu)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Mass").default_val(1.0f).min(0.01f).max(100.0f);
    b.add_input<float>("Young's Modulus").default_val(1e6f).min(1e3f).max(1e9f);
    b.add_input<float>("Poisson's Ratio").default_val(0.45f).min(0.0f).max(0.49f);
    b.add_input<float>("Damping").default_val(0.99f).min(0.0f).max(1.0f);
    b.add_input<int>("Substeps").default_val(1).min(1).max(20);
    b.add_input<int>("Newton Iterations").default_val(30).min(1).max(100);
    b.add_input<float>("Newton Tolerance").default_val(1e-2f).min(1e-8f).max(1e-1f);
    b.add_input<float>("Gravity").default_val(-9.81f).min(-20.0f).max(0.0f);
    b.add_input<float>("Ground Restitution").default_val(0.3f).min(0.0f).max(1.0f);
    b.add_input<bool>("Flip Normal").default_val(false);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(neo_hookean_gpu)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<NeoHookeanGPUStorage&>();

    spdlog::info("[NeoHookean] Starting execution");

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    float mass = params.get_input<float>("Mass");
    float youngs_modulus = params.get_input<float>("Young's Modulus");
    float poisson_ratio = params.get_input<float>("Poisson's Ratio");
    float damping = params.get_input<float>("Damping");
    int substeps = params.get_input<int>("Substeps");
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance = std::max(tolerance, 1e-8f);
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Convert Young's modulus and Poisson's ratio to Lamé parameters
    float mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    float lambda = youngs_modulus * poisson_ratio / ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_counts;

    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
        face_counts = mesh_component->get_face_vertex_counts();
    } else {
        auto points_component = input_geom.get_component<PointsComponent>();
        if (!points_component) {
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }
        positions = points_component->get_vertices();
    }

    int num_particles = positions.size();
    spdlog::info("[NeoHookean] num_particles = {}", num_particles);
    if (num_particles == 0) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Initialize buffers only once or when particle count changes
    if (!storage.initialized || storage.num_particles != num_particles) {
        spdlog::info("[NeoHookean] Initializing storage...");
        storage.initialize(positions, face_vertex_indices, face_counts, mass);
        spdlog::info("[NeoHookean] Storage initialized: num_elements = {}", storage.num_elements);
    }

    if (!storage.initialized || storage.num_elements == 0) {
        spdlog::warn("[NeoHookean] Neo-Hookean simulation requires tetrahedral mesh. Skipping simulation.");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }
    spdlog::info("[NeoHookean] Starting simulation: dt={}, substeps={}", dt, substeps);

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;
    auto d_f_ext = storage.f_ext_buffer;

    // Substep loop
    float dt_sub = dt / substeps;
    for (int substep = 0; substep < substeps; ++substep) {
        spdlog::info("[NeoHookean] Substep {}/{}", substep + 1, substeps);
        // Setup external forces on GPU
        spdlog::info("[NeoHookean] Setting up external forces...");
        rzsim_cuda::setup_external_forces_nh_gpu(mass, gravity, num_particles, d_f_ext);

        // Compute x_tilde = x + dt_sub * v on GPU
        spdlog::info("[NeoHookean] Computing explicit step...");
        rzsim_cuda::explicit_step_nh_gpu(
            d_positions, d_velocities, dt_sub, num_particles, d_next_positions);

        // Newton's method iterations
        spdlog::info("[NeoHookean] Starting Newton iterations...");
        storage.x_new_buffer->copy_from_device(d_next_positions.Get());

        bool converged = false;
        for (int iter = 0; iter < max_iterations; iter++) {
            spdlog::info("[NeoHookean] Newton iteration {}/{}", iter + 1, max_iterations);
            // Compute gradient at current x_new
            spdlog::info("[NeoHookean] Computing gradient...");
            rzsim_cuda::compute_gradient_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                storage.tetrahedra_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                d_gradients);
            spdlog::info("[NeoHookean] Gradient computed");

            // Check gradient norm for convergence
            spdlog::info("[NeoHookean] Computing gradient norm...");
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                spdlog::error("[NeoHookean] CUDA error before norm: {}", cudaGetErrorString(err));
            }
            float grad_norm = rzsim_cuda::compute_vector_norm_nh_gpu(
                d_gradients, num_particles * 3);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                spdlog::error("[NeoHookean] CUDA error after norm: {}", cudaGetErrorString(err));
            }
            spdlog::info("[NeoHookean] Gradient norm: {}", grad_norm);

            if (!std::isfinite(grad_norm)) {
                spdlog::error("[NeoHookean] Gradient norm is not finite! Simulation unstable.");
                break;
            }

            auto dof = num_particles * 3;
            grad_norm = grad_norm / dof;
            spdlog::info("[NeoHookean] Normalized gradient norm: {}", grad_norm);

            if (iter > 0 && grad_norm < tolerance) {
                spdlog::info("[NeoHookean] Converged!");
                converged = true;
                break;
            }

            // Update Hessian values
            spdlog::info("[NeoHookean] Updating Hessian values...");
            rzsim_cuda::update_hessian_values_nh_gpu(
                storage.hessian_structure,
                storage.x_new_buffer,
                d_M_diag,
                storage.tetrahedra_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.hessian_values);
            spdlog::info("[NeoHookean] Hessian updated");

            // Solve H * p = -grad using CUDA CG
            float cg_tol = std::max(1e-9f, grad_norm * 1e-3f);

            Ruzino::Solver::SolverConfig solver_config;
            solver_config.tolerance = cg_tol;
            solver_config.max_iterations = 1000;
            solver_config.use_preconditioner = true;
            solver_config.verbose = false;

            // Negate gradient for RHS
            spdlog::info("[NeoHookean] Negating gradient...");
            rzsim_cuda::negate_nh_gpu(
                d_gradients, storage.neg_gradient_buffer, num_particles * 3);

            // Solve on GPU
            spdlog::info("[NeoHookean] Solving linear system with CG...");
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
                spdlog::warn("[NeoHookean] CG solver did not converge in iteration {}", iter);
            } else {
                spdlog::info("[NeoHookean] CG solver converged in {} iterations", result.iterations);
            }

            // Line search with energy descent
            spdlog::info("[NeoHookean] Starting line search...");
            float E_current = rzsim_cuda::compute_energy_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                storage.tetrahedra_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.inertial_terms_buffer,
                storage.element_energies_buffer,
                storage.potential_terms_buffer);

            float alpha = 1.0f;
            int ls_iter = 0;
            float E_candidate = std::numeric_limits<float>::infinity();

            while (E_candidate > E_current && ls_iter < 50) {
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
                    d_f_ext,
                    storage.tetrahedra_buffer,
                    storage.Dm_inv_buffer,
                    storage.volumes_buffer,
                    mu,
                    lambda,
                    dt_sub,
                    num_particles,
                    storage.num_elements,
                    storage.inertial_terms_buffer,
                    storage.element_energies_buffer,
                    storage.potential_terms_buffer);

                if (E_candidate <= E_current) {
                    storage.x_new_buffer->copy_from_device(storage.x_candidate_buffer.Get());
                    break;
                }

                alpha *= 0.5f;
                ls_iter++;
            }

            if (alpha < 1e-4f) {
                spdlog::warn("Line search failed, alpha too small");
                break;
            }
        }

        // Update velocities: v = (x_new - x_n) / dt_sub and apply damping
        spdlog::info("[NeoHookean] Updating velocities...");
        auto x_new_final = storage.x_new_buffer->get_host_vector<float>();
        auto x_n_host = d_positions->get_host_vector<glm::vec3>();
        std::vector<glm::vec3> v_new(num_particles);
        for (int i = 0; i < num_particles; i++) {
            v_new[i].x = (x_new_final[i * 3 + 0] - x_n_host[i].x) / dt_sub * damping;
            v_new[i].y = (x_new_final[i * 3 + 1] - x_n_host[i].y) / dt_sub * damping;
            v_new[i].z = (x_new_final[i * 3 + 2] - x_n_host[i].z) / dt_sub * damping;
        }

        // Handle ground collision (z = 0)
        for (int i = 0; i < num_particles; i++) {
            float z_new = x_new_final[i * 3 + 2];
            if (z_new < 0.0f) {
                x_new_final[i * 3 + 2] = 0.0f;
                
                // Reflect velocity with restitution
                if (v_new[i].z < 0.0f) {
                    v_new[i].z = -v_new[i].z * restitution;
                }
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
        spdlog::info("[NeoHookean] Substep {} completed", substep + 1);
    }

    // Update geometry with new positions
    if (mesh_component) {
        auto final_positions = d_positions->get_host_vector<glm::vec3>();
        mesh_component->set_vertices(final_positions);

        // Note: For proper rendering, you'd want to recompute normals
        // For now, we'll keep the original normals or recompute them from surface
    } else {
        auto points_component = input_geom.get_component<PointsComponent>();
        auto final_positions = d_positions->get_host_vector<glm::vec3>();
        points_component->set_vertices(final_positions);
    }

    spdlog::info("[NeoHookean] Execution completed successfully");
    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(neo_hookean_gpu);
NODE_DEF_CLOSE_SCOPE
