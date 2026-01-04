#include <GCore/Components/MeshComponent.h>
#include <GCore/GOP.h>
#include <gtest/gtest.h>
#include <rzsim/rzsim.h>


// Forward declarations for CUDA initialization
namespace Ruzino {
namespace cuda {
    extern int cuda_init();
    extern int cuda_shutdown();
}  // namespace cuda
}  // namespace Ruzino
#include <iostream>

using namespace Ruzino;

TEST(AdjacencyMap, SimpleTriangle)
{
    // Create a simple triangle mesh
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Triangle vertices: 0, 1, 2
    std::vector<glm::vec3> vertices = { glm::vec3(0.0f, 0.0f, 0.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f),
                                        glm::vec3(0.0f, 1.0f, 0.0f) };

    std::vector<int> faceVertexCounts = { 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    // Test GPU version
    auto adjacencyGPU = get_adjcency_map_gpu(mesh);
    ASSERT_NE(adjacencyGPU, nullptr);

    // Test CPU version (just downloads from GPU)
    auto adjacencyCPU = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);

    std::cout << "Adjacency map for triangle:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << std::endl;

    // For a single triangle [0, 1, 2]:
    // - Edge 0-1: 0->1, 1->0
    // - Edge 1-2: 1->2, 2->1
    // - Edge 2-0: 2->0, 0->2
    // Vertex 0 connects to: 1, 2
    // Vertex 1 connects to: 0, 2
    // Vertex 2 connects to: 1, 0
    std::vector<unsigned> expected = { 2, 1, 2, 2, 0, 2, 2, 1, 0 };
    ASSERT_EQ(adjacencyCPU.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(adjacencyCPU[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(AdjacencyMap, Quad)
{
    // Create a quad mesh
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Quad vertices: 0--1
    //                |  |
    //                3--2
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 1.0f, 0.0f),  // 0
        glm::vec3(1.0f, 1.0f, 0.0f),  // 1
        glm::vec3(1.0f, 0.0f, 0.0f),  // 2
        glm::vec3(0.0f, 0.0f, 0.0f)   // 3
    };

    std::vector<int> faceVertexCounts = { 4 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 3 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto adjacencyCPU = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);

    std::cout << "Adjacency map for quad:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << std::endl;

    // For quad [0, 1, 2, 3]:
    // - Vertex 0 connects to 1 and 3
    // - Vertex 1 connects to 0 and 2
    // - Vertex 2 connects to 1 and 3
    // - Vertex 3 connects to 2 and 0
    std::vector<unsigned> expected = { 2, 1, 3, 2, 0, 2, 2, 1, 3, 2, 2, 0 };
    ASSERT_EQ(adjacencyCPU.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(adjacencyCPU[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(AdjacencyMap, TwoTriangles)
{
    // Create two triangles sharing an edge
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Vertices:  0
    //           /|\
    //          / | \
    //         1--2--3
    // Faces: [0,1,2] and [0,2,3]
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.5f, 1.0f, 0.0f),  // 0
        glm::vec3(0.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.5f, 0.0f, 0.0f),  // 2
        glm::vec3(1.0f, 0.0f, 0.0f)   // 3
    };

    std::vector<int> faceVertexCounts = { 3, 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 0, 2, 3 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto adjacencyCPU = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);

    std::cout << "Adjacency map for two triangles:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << std::endl;

    // From face [0,1,2]: edges 0-1, 1-2, 2-0
    // From face [0,2,3]: edges 0-2, 2-3, 3-0
    // V0: from (0,1), (1,0), (0,2), (2,0), (3,0), (0,3) but no duplicates in
    // count = 4 neighbors Note: The order of neighbors might vary due to atomic
    // operations being unordered V0: neighbors could be [1, 2, 2, 3] or any
    // permutation with these values V1: [0, 2] V2: [0, 3, 1, 0] or similar
    // (order may vary) V3: [2, 0] Total size: (4+1) + (2+1) + (4+1) + (2+1) = 5
    // + 3 + 5 + 3 = 16
    std::vector<unsigned> expected = { 4, 1, 2, 2, 3, 2, 0, 2,
                                       4, 0, 3, 1, 0, 2, 2, 0 };
    ASSERT_EQ(adjacencyCPU.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(adjacencyCPU[i], expected[i]) << "Mismatch at index " << i;
    }
}

int main(int argc, char** argv)
{
    // Initialize CUDA - use forward declaration to avoid namespace conflicts
    Ruzino::cuda::cuda_init();

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // Cleanup
    Ruzino::cuda::cuda_shutdown();

    return result;
}
