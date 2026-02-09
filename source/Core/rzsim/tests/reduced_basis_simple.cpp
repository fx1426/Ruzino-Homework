#include <GCore/Components/MeshComponent.h>
#include <GCore/GOP.h>
#include <GCore/algorithms/tetgen_algorithm.h>
#include <GCore/create_geom.h>
#include <GCore/util_openmesh_bind.h>
#include <rzsim/reduced_order_basis.h>

#include <iomanip>
#include <iostream>

using namespace Ruzino;

// Helper to print eigenvalues
void print_eigenmode_info(
    const ReducedOrderedBasis& rob,
    const std::string& name)
{
    std::cout << "\n=== " << name << " ===\n";
    std::cout << "Number of modes: " << rob.basis.size() << std::endl;
    std::cout << "Laplacian size: " << rob.laplacian_matrix_.rows() << " x "
              << rob.laplacian_matrix_.cols() << std::endl;
    std::cout << "Laplacian non-zeros: " << rob.laplacian_matrix_.nonZeros()
              << std::endl;

    int num_to_print = std::min(10, static_cast<int>(rob.eigenvalues.size()));
    std::cout << "\nFirst " << num_to_print << " eigenvalues:\n";
    for (int i = 0; i < num_to_print; i++) {
        std::cout << "  λ[" << i << "] = " << std::setprecision(8)
                  << rob.eigenvalues[i] << std::endl;
    }
}

void test_tetrahedral_mesh()
{
    std::cout << "\n========== Tetrahedral Mesh Test ==========\n"
              << std::flush;

    // Create a sphere (already triangulated)
    Geometry sphere = create_ico_sphere(2, 1.0f);

    geom_algorithm::TetgenParams params;
    params.max_volume = 0.05f;
    params.quality_ratio = 2.0f;

    Geometry mesh = geom_algorithm::tetrahedralize(sphere, params);

    std::cout << "Mesh created\n" << std::flush;

    // Test custom implementation
    {
        ReducedOrderedBasis rob(mesh, 10, 3, false);
        print_eigenmode_info(rob, "Custom 3D Implementation");
    }

    // Test libigl implementation
    {
        ReducedOrderedBasis rob_libigl(mesh, 10, 3, true);
        print_eigenmode_info(rob_libigl, "libigl 3D Implementation");
    }

    // Compare
    ReducedOrderedBasis rob_custom(mesh, 10, 3, false);
    ReducedOrderedBasis rob_libigl(mesh, 10, 3, true);

    std::cout << "\n=== Comparison (custom vs libigl) ===\n";
    for (int i = 0;
         i < std::min(10, static_cast<int>(rob_custom.eigenvalues.size()));
         i++) {
        double custom_val = rob_custom.eigenvalues[i];
        double libigl_val = rob_libigl.eigenvalues[i];
        double diff = std::abs(custom_val - libigl_val);
        std::cout << "Mode " << i << ": custom=" << std::setprecision(6)
                  << custom_val << " libigl=" << libigl_val << " diff=" << diff
                  << std::endl;
    }
}

int main()
{
    std::cout << "Starting tests...\n" << std::flush;
    try {
        test_tetrahedral_mesh();
        std::cout << "Tests completed successfully!\n" << std::flush;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Test failed with unknown exception\n";
        return 1;
    }
    return 0;
}
