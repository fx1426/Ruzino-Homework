#include <exception>
#include <string>

#include "nodes/core/def/node_def.hpp"
#include "polyscope/curve_network.h"
#include "polyscope/pick.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/utilities.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(visualize_vertices_path)
{
    b.add_input<std::string>("Mesh Name");
    b.add_input<std::list<size_t>>("Vertex Indices");
}

NODE_EXECUTION_FUNCTION(visualize_vertices_path)
{
    auto structure_name = params.get_input<std::string>("Mesh Name");
    structure_name = std::string(structure_name.c_str());
    auto vertex_indices = params.get_input<std::list<size_t>>("Vertex Indices");

    if (!polyscope::hasStructure("Surface Mesh", structure_name)) {
        std::cerr << "The mesh name is not found." << std::endl;
        return false;
    }

    if (vertex_indices.size() < 2) {
        std::cerr << "The number of vertices is less than 2." << std::endl;
        return false;
    }

    auto structure = polyscope::getStructure("Surface Mesh", structure_name);
    auto mesh = dynamic_cast<polyscope::SurfaceMesh*>(structure);

    std::list<std::array<float, 3>> vertices;
    for (auto i : vertex_indices) {
        if (i >= mesh->nVertices()) {
            std::cerr << "The vertex index is out of range." << std::endl;
            return false;
        }
        auto pos = mesh->vertexPositions.getValue(i);
        vertices.push_back({ pos.x, pos.y, pos.z });
    }

    auto curve_network =
        polyscope::registerCurveNetworkLine("Path", vertices)->setEnabled(true);
    curve_network->setTransform(mesh->getTransform());

    return true;
}

NODE_DECLARATION_UI(visualize_vertices_path);
NODE_DECLARATION_REQUIRED(visualize_vertices_path);
NODE_DEF_CLOSE_SCOPE
